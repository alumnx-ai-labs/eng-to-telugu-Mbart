"""
QLoRA fine-tuning of mBART-50 for English -> Telugu: the base model is loaded in 4-bit (nf4)
and frozen, only small LoRA adapters are trained. Needs a CUDA GPU and bitsandbytes
(pip install -r requirements-quant.txt).
"""
import argparse
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from train import prepare_dataset, MODEL_NAME

OUTPUT_DIR = "./mbart-qlora-en-te"

def train_qlora(epochs=5, batch_size=8):
    if not torch.cuda.is_available():
        raise SystemExit("error: QLoRA needs a CUDA GPU (4-bit loading is not supported on CPU).")
    print(f"info: using device: cuda ({torch.cuda.get_device_name(0)})")

    # Load Model & Tokenizer (base model quantized to 4-bit)
    print("info: loading tokenizer and 4-bit model...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "te_IN"

    # Prepare the quantized model for training (also enables gradient checkpointing)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # Apply LoRA (same config as train.py)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        output_dir="./checkpoints-qlora",
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        predict_with_generate=False,
        fp16=True,
        logging_steps=50,
        dataloader_num_workers=4,
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

    print(f"info: saving adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"info: peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("info: training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    args = parser.parse_args()

    train_qlora(epochs=args.epochs, batch_size=args.batch_size)
