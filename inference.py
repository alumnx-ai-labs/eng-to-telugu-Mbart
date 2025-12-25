import torch
import argparse
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

MODEL_DIR = "./mbart-finetuned-en-te"
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"

def load_model(model_path):
    print(f"info: loading model from {model_path}...")
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
    except OSError:
        print(f"warning: fine-tuned model not found at {model_path}. loading base model '{BASE_MODEL}' instead.")
        tokenizer = MBart50TokenizerFast.from_pretrained(BASE_MODEL)
        model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer.src_lang = "en_XX"
    return model, tokenizer, device

def translate(text, model, tokenizer, device, target_lang="te_IN"):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
    
    # Generate translation
    generated_ids = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Text to translate (if not provided, enters interactive mode)")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Path to the model directory")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model_dir)
    
    if args.text:
        print(f"\nOriginal: {args.text}")
        translation = translate(args.text, model, tokenizer, device)
        print(f"Telugu:   {translation}\n")
    else:
        print("\n--- English to Telugu Translator (Interactive) ---")
        print("Type 'q' or 'quit' to exit.\n")
        while True:
            text = input("Enter English text: ")
            if text.lower() in ['q', 'quit', 'exit']:
                break
            translation = translate(text, model, tokenizer, device)
            print(f"Telugu: {translation}\n")
