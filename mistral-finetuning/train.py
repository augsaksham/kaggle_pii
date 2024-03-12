from datetime import datetime
from modal import Secret
from transformers import TrainerCallback

from common import stub, BASE_MODEL, MODEL_PATH, VOLUME_CONFIG

WANDB_PROJECT = "hf-mistral7b-finetune"

# Callback function to store model checkpoints in modal.Volume
class CheckpointCallback(TrainerCallback):
    def __init__(self, volume):
        self.volume = volume

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print("running commit on modal.Volume after model checkpoint")
            self.volume.commit()
            

# Download training dataset from Hugging Face and push to modal.Volume. 
# You can load in your own dataset to push to a Volume here or push local data files 
# using `modal volume put VOLUME_NAME [LOCAL_PATH] [REMOTE_PATH]` in your CLI.
@stub.function(volumes=VOLUME_CONFIG)
def download_dataset():
    import os
    from datasets import load_dataset
    import pandas as pd
    from huggingface_hub import login
    login(token='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')

    if not os.path.exists('/training_data/data_train.jsonl'):
        def format_instruction(sample):
            PROMPT_TEMPLATE = """[INST] <<SYS>>
            #Mission
            You are a student which is tasked with writing an essay for a random Challenge using a given TOOL, also you need to use the provided PII in your essay.

            #Context
            - TOOL is a methodology used to write an essay.
            - PII this contains a dictionary of personal information the dictionary can include the keys : ['MAIL', 'ID_NUM', 'NAME_STUDENT', 'PHONE_NUM', 'STREET_ADDRESS', 'URL_PERSONAL', 'USERNAME']
            - Challenge is a random topic around which the essay is formulated.

            #instructions
            Step 0 : Analyze the provided Tool and the PII to get a gist of the type of essay that needs to be written. Note the keys of PII can have multiple values associated with them make sure to include all of them in the final essay.
            Step 1 : Use the analysis in Step 0 to come up with a random challenge suiting the TOOL and the PII.
            Step 2 : Write an essay in simple vocabulary on the chosen Challenge in Step1 using the TOOL and the PII make sure to include all the information present in the PII in the essay you can also repeat some of the information.
            <</SYS>>

            Input:
            TOOL: {tool} 
            PII: {pii} [/INST]

            Essay: {essay}"""
            # PROMPT_TEMPLATE = "[INST] <<SYS>>\nUse the provided tool and the to write an essay about any random challenge. While generating the essay include the provided personal information in the essay.\n<</SYS>>\n\nInput:\n Tool: {tool} \n Personal_Information: {pii} [/INST]\n\nEssay: {essay}"
            return {"text": PROMPT_TEMPLATE.format(tool=sample["TOOL"],pii=sample['PII'],essay=sample["full_text"])}
        
        # downloading data from hugging face
        train_dataset = load_dataset('augsaksham/full_train', split='train')
        val_dataset = load_dataset('augsaksham/full_train', split='validation')


        train_dataset = train_dataset.map(format_instruction, remove_columns=['document', 'PII', 'TOOL','full_text','is_valid'])
        val_dataset = val_dataset.map(format_instruction, remove_columns=['document', 'PII', 'TOOL','full_text','is_valid'])

        # writing data to Volume mounted at "/training-data" in container
        train_dataset.to_json(f"/training_data/data_train.jsonl")
        val_dataset.to_json(f"/training_data/data_val.jsonl")
        
    stub.training_data_volume.commit()


@stub.function(
    gpu="A100",
    secrets=[
        Secret.from_name("my-wandb-secret") if WANDB_PROJECT else None,
        Secret.from_name("my-huggingface-secret")
    ],
    timeout=60 * 60 * 4,
    volumes=VOLUME_CONFIG,
)
def finetune(
    model_name: str, 
    run_id: str = "",
    wandb_project: str = "", 
    resume_from_checkpoint: str = None  # path to checkpoint in Volume (e.g. "/results/checkpoint-300/")
):
    import os
    import torch
    import transformers
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config)  # Load and quantize the pretrained model baked into our image
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    def epochs_to_max_steps(num_epochs, total_train_samples, train_batch_size, gradient_accumulation_steps):
        """Convert the number of epochs to the maximum number of training steps."""
        steps_per_epoch = total_train_samples // (train_batch_size * gradient_accumulation_steps)
        max_steps = num_epochs * steps_per_epoch
        return max_steps
    

    def tokenize(sample, cutoff_len=512, add_eos_token=True):
        prompt = sample["text"]
        result = tokenizer.__call__(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()

        return result
    
    # Load datasets from training data Volume
    train_dataset = load_dataset('json', data_files='/training_data/data_train.jsonl', split="train")
    eval_dataset = load_dataset('json', data_files='/training_data/data_val.jsonl', split="train")

    tokenized_train_dataset = train_dataset.map(tokenize)
    tokenized_val_dataset = eval_dataset.map(tokenize)
    
    # Train Config
    num_epochs = 3
    total_train_samples = len(train_dataset) 
    train_batch_size = 8
    gradient_accumulation_steps = 4
    max_steps = epochs_to_max_steps(num_epochs, total_train_samples, train_batch_size, gradient_accumulation_steps)
    save_steps = max_steps//5
    print("Maximum number of steps:", max_steps)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if len(wandb_project) > 0:
        # Set environment variables if Weights and Biases is enabled
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_WATCH"] = "gradients"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        callbacks=[CheckpointCallback(stub.results_volume)], # Callback function for committing a checkpoint to Volume when reached
        args=transformers.TrainingArguments(
            output_dir=f"/results/{run_id}",  # Must also set this to write into results Volume's mount location
            warmup_steps=5,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,  # Feel free to tweak to correct for under/overfitting
            learning_rate=2e-5, # ~10x smaller than Mistral's learning rate
            bf16=True,
            optim="adamw_8bit",
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=save_steps,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=save_steps,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb" if wandb_project else "",         
            run_name=run_id if wandb_project else ""     
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # Silence the warnings. Re-enable for inference!
    trainer.train()  # Run training
    
    model.save_pretrained(f"/results/{run_id}") 
    stub.results_volume.commit()


@stub.local_entrypoint()
def main(run_id: str = "", resume_from_checkpoint: str = None):
    print("Downloading data from Hugging Face and syncing to volume.")
    download_dataset.remote()
    print("Finished syncing data.")

    if not run_id:
        run_id = f"mistral7b-finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    print(f"Starting training run {run_id=}.")
    finetune.remote(model_name=BASE_MODEL, run_id=run_id, wandb_project=WANDB_PROJECT, resume_from_checkpoint=resume_from_checkpoint)
    print(f"Completed training run {run_id=}")
    print("To test your trained model, run `modal run inference.py --run_id <run_id>`")
    
    
# PROMPT_TEMPLATE = "[INST] <<SYS>>\nUse the provided tool and the to write an essay about any random challenge. While generating the essay include the provided personal information in the essay.\n<</SYS>>\n\nInput:\n Tool: {tool} \n Personal_Information: {pii} [/INST]\n\nEssay: {essay}"