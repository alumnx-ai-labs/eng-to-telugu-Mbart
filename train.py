import os
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split

# Constants
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DATASET_URL = "hf://datasets/Shreya3095/TeluguTranslator/data/train-00000-of-00001.parquet"
OUTPUT_DIR = "./mbart-finetuned-en-te"

def prepare_dataset():
    """
    Downloads and prepares the Telugu-English dataset.
    """
    print("info: loading dataset from parquet...")
    df = pd.read_parquet(DATASET_URL)
    
    # Rename and clean
    df = df.rename(columns={'question': 'english', 'answer': 'telugu'})
    df = df.dropna()
    df['english'] = df['english'].astype(str).str.strip()
    df['telugu'] = df['telugu'].astype(str).str.strip()
    
    # Basic filtering
    df = df[(df['english'].str.len() > 3) & (df['english'].str.len() < 200)]
    df = df[(df['telugu'].str.len() > 3) & (df['telugu'].str.len() < 300)]
    
    print(f"info: dataset size: {len(df)}")
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Convert to HF Dataset
    train_ds = Dataset.from_pandas(train_df[['english', 'telugu']])
    val_ds = Dataset.from_pandas(val_df[['english', 'telugu']])
    
    return train_ds, val_ds

def train(epochs=10, batch_size=8, gradient_accumulation_steps=1):
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"info: using device: {device}")
    
    # Load Model & Tokenizer
    print("info: loading model and tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "te_IN"
    
    # Prepare Data
    train_ds, val_ds = prepare_dataset()
    
    def preprocess_function(examples):
        inputs = tokenizer(examples["english"], max_length=128, truncation=True, padding="max_length")
        targets = tokenizer(examples["telugu"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs
    
    print("info: tokenizing data...")
    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["english", "telugu"])
    tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=["english", "telugu"])
    
    # Training Config
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=(device == "cuda"),  # Enable mixed precision on GPU
        logging_dir="./logs",
        logging_steps=50,
        dataloader_num_workers=4,
        report_to="none" # Disable wandb/mlflow unless configured
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("info: starting training...")
    trainer.train()
    
    print(f"info: saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("info: training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size)
