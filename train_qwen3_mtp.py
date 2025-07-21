import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from mtp_qwen3 import Qwen3ForCausalLMWithMTP
from torch.optim import AdamW

class DetailedMonitoringCallback(TrainerCallback):
    """Monitor MTP training and provide detailed loss breakdown"""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if hasattr(model, 'loss_details') and model.loss_details:
            details = model.loss_details
            
            # Check participation of all MTP heads dynamically
            mtp_participated = any(key in details for key in ['mtp1_loss', 'mtp2_loss', 'mtp3_loss'])
            
            if state.is_local_process_zero and args.logging_steps > 0 and state.global_step % args.logging_steps == 0:
                status_msg = f"Step {state.global_step}: "
                
                if 'total_loss' in details and details['total_loss'] is not None:
                    status_msg += f"Loss={details['total_loss']:.4f} ("
                    loss_parts = []
                    # Dynamically build loss components to support arbitrary number of MTP heads
                    for i in range(1, 4):  # Check MTP1, MTP2, MTP3
                        loss_key = f'mtp{i}_loss'
                        if loss_key in details and details[loss_key] > 0:
                            loss_parts.append(f"mtp{i}={details[loss_key]:.4f}")
                    status_msg += ", ".join(loss_parts) + ")"
                
                status_msg += f", MTP={'✓' if mtp_participated else '✗'}"
                print(status_msg)
            
            if logs is not None:
                logs.update(details) 
                logs["train/mtp_participated"] = 1.0 if mtp_participated else 0.0
        
        return control

def construct_unified_prompt(passage, question):
    """Construct unified prompt template"""

    template = """Read the passage and answer the question with 'true' or 'false'. 
    After your answer, explain your reasoning.
    Passage: {passage}
    Question: {question}
    Answer: 
    """
    return template.format(passage=passage, question=question)

def preprocess_function(examples, tokenizer, max_length=1024):
    """Data preprocessing function - fix label masking errors caused by truncation"""
    prompts = [construct_unified_prompt(p, q) for p, q in zip(examples["passage"], examples["question"])]
    answers = examples["answer"]
    full_texts = [p + a for p, a in zip(prompts, answers)]
    
    # 1. Tokenize prompts again with same truncation config to get correct prompt lengths
    prompts_tokenized = tokenizer(prompts, padding=False, truncation=True, max_length=max_length)
    prompt_lens = [len(x) for x in prompts_tokenized["input_ids"]]

    # 2. Tokenize full texts for model input
    full_texts_tokenized = tokenizer(
        full_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    )
    input_ids = full_texts_tokenized["input_ids"]
    attention_mask = full_texts_tokenized["attention_mask"]

    # 3. Create labels and mask using correct prompt lengths
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i, length in enumerate(prompt_lens):
        # Use correct prompt length after truncation for masking
        labels[i, :length] = -100

    position_ids = (torch.cumsum(attention_mask, dim=1) - 1)
    position_ids.masked_fill_(attention_mask == 0, 0)
    
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "position_ids": position_ids.tolist(),
        "labels": labels.tolist(),
    }

def data_collator(features, tok):
    """Data collator function"""
    batch = {}
    first = features[0]
    for k, v in first.items():
        batch[k] = torch.stack([torch.tensor(f[k]) for f in features])
    return batch

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(42)
    
    # TODO: Replace with your own data file paths
    data_files = {
        "train": "path/to/your/train_data",
        "validation": "path/to/your/validation_data"
    }
    dataset = load_dataset("json", data_files=data_files)
    
    train_dataset_raw = dataset['train'].shuffle(seed=42)
    val_dataset_raw = dataset['validation'].shuffle(seed=42)
    
    model_name = "Qwen/Qwen3-4B"
    print(f"Loading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Step 1: Filter samples with too short answers
    print("(1/2) Filtering samples with short answers...")
   
    min_answer_tokens = 4
    def is_long_enough(example):
        return len(tokenizer(example['answer'], add_special_tokens=False)['input_ids']) >= min_answer_tokens
    
    original_train_count = len(train_dataset_raw)
    original_val_count = len(val_dataset_raw)
    train_dataset_raw = train_dataset_raw.filter(is_long_enough, num_proc=4, desc="Filtering short answers in train set")
    val_dataset_raw = val_dataset_raw.filter(is_long_enough, num_proc=4, desc="Filtering short answers in val set")
    print(f"Filtering complete. Train: {original_train_count} -> {len(train_dataset_raw)} | Val: {original_val_count} -> {len(val_dataset_raw)}")

    config = AutoConfig.from_pretrained(model_name)
    
    print("Loading MTP model...")
    model = Qwen3ForCausalLMWithMTP.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    print("MTP model loaded successfully")
    
    model.freeze_main_model()
    
    def tokenize_function(batch):
        return preprocess_function(batch, tokenizer, max_length=1024)
    
    print("(2/2) Processing and tokenizing data...")
    train_dataset = train_dataset_raw.map(
        tokenize_function, batched=True, batch_size=16, 
        remove_columns=["question", "answer", "passage"], desc="Tokenizing training samples"
    )
    val_dataset = val_dataset_raw.map(
        tokenize_function, batched=True, batch_size=16,
        remove_columns=["question", "answer", "passage"], desc="Tokenizing validation samples"
    )

    # Step 2: Filter samples that lost all labels due to truncation
    print("(2/2) Filtering samples that lost all labels due to truncation...")
    def has_valid_labels(example):
        return any(label != -100 for label in example['labels'])

    original_train_count = len(train_dataset)
    original_val_count = len(val_dataset)
    train_dataset = train_dataset.filter(has_valid_labels, desc="Filtering invalid labels in train set")
    val_dataset = val_dataset.filter(has_valid_labels, desc="Filtering invalid labels in val set")
    print(f"Filtering complete. Train: {original_train_count} -> {len(train_dataset)} | Val: {original_val_count} -> {len(val_dataset)}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            # Output directory for 3-head model
            output_dir="./qwen3-mtp-3-heads",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            bf16=True,
            eval_strategy="steps", 
            eval_steps=1000,
            logging_steps=50,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            learning_rate=5e-5,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            logging_dir="./qwen3-mtp-3-heads/logs",
            remove_unused_columns=False,
            gradient_checkpointing=False,
            # Use traditional PyTorch save method to avoid safetensors strict checks
            save_safetensors=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=lambda features: data_collator(features, tokenizer),
        callbacks=[DetailedMonitoringCallback()],
    )
    
    print("Starting MTP training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join("./qwen3-mtp-3-heads", "final_model")

    model.save_pretrained(final_model_path, safe_serialization=False)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 