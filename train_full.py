"""
Full fine-tuning (all weights, no LoRA, no quantization) of mBART-50 for English -> Telugu.

Memory: roughly 12-14 GB on a 16 GB T4 (fp32 weights + gradients + AdamW states + activations),
so run it with the GPU to yourself. On CUDA OOM try, in this order:
  --batch_size 2 --grad_accum 8
  --grad_ckpt            (gradient checkpointing, slower but much less activation memory)
  --optim adafactor      (much smaller optimizer state than AdamW)
"""
import argparse
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from train import prepare_dataset, MODEL_NAME

OUTPUT_DIR = "./mbart-full-finetuned-en-te"

def train_full(epochs=5, batch_size=4, grad_accum=2, lr=5e-5, optim="adamw_torch", grad_ckpt=False):
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"info: using device: {device}")

    # Load Model & Tokenizer
    print("info: loading model and tokenizer...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "te_IN"

    # All weights are trained
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"info: total trainable params: {total_params:,}")

    # The KV cache is useless during training (and conflicts with gradient checkpointing)
    model.config.use_cache = False

    # Prepare Data
    train_ds, val_ds = prepare_dataset()

    def preprocess_function(examples):
        inputs = tokenizer(examples["english"], max_length=128, truncation=True, padding="max_length")
        targets = tokenizer(text_target=examples["telugu"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = [[(x if x != tokenizer.pad_token_id else -100) for x in ids] for ids in targets["input_ids"]]
        return inputs

    print("info: tokenizing data...")
    tokenized_train = train_ds.map(preprocess_function, batched=True, remove_columns=["english", "telugu"])
    tokenized_val = val_ds.map(preprocess_function, batched=True, remove_columns=["english", "telugu"])

    # Training Config
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints-full",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        weight_decay=0.01,
        optim=optim,
        gradient_checkpointing=grad_ckpt,
        eval_strategy="epoch",
        save_strategy="no",  # skip checkpoints (each would be ~7 GB with optimizer state); final model is saved below
        predict_with_generate=False,
        fp16=(device == "cuda"),  # Enable mixed precision on GPU
        logging_steps=20,
        dataloader_num_workers=2,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("info: starting training...")
    trainer.train()

    # Re-enable the cache for fast generation at inference time
    model.config.use_cache = True

    print(f"info: saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if device == "cuda":
        print(f"info: peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("info: training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer (use 'adafactor' for lower memory)")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing (less memory, slower)")
    args = parser.parse_args()

    train_full(epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
               lr=args.lr, optim=args.optim, grad_ckpt=args.grad_ckpt)
