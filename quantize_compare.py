"""
Compare the BASE model in fp16, 8-bit and 4-bit (nf4): weight memory, peak memory and
translation speed, plus the outputs side by side. Needs a CUDA GPU and bitsandbytes
(pip install -r requirements-quant.txt).
"""
import gc
import time
import argparse
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, BitsAndBytesConfig
from train import MODEL_NAME

SENTENCES = [
    "How are you?",
    "What is your name?",
    "I will go to the temple daily",
    "Thank you very much",
    "Where is the nearest railway station?",
    "The weather is very nice today",
    "Please give me a glass of water",
    "My brother works in a big company in Hyderabad",
]

GB = 1024 ** 3

def load_mode(mode, model_name):
    if mode == "fp16":
        model = MBartForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        return model.to("cuda").eval()
    if mode == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = MBartForConditionalGeneration.from_pretrained(
        model_name, quantization_config=bnb_config, device_map={"": 0}
    )
    return model.eval()

def translate(text, model, tokenizer):
    # Same generation settings as inference.translate
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("te_IN"),
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def run_mode(mode, model_name, tokenizer, sentences):
    print(f"\ninfo: === {mode} ===")
    torch.cuda.reset_peak_memory_stats()
    model = load_mode(mode, model_name)
    weights_gb = torch.cuda.memory_allocated() / GB
    print(f"info: weights memory: {weights_gb:.2f} GB")

    # Warm-up (CUDA kernels, caches) so it does not distort the timing
    translate(sentences[0], model, tokenizer)

    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = [translate(s, model, tokenizer) for s in sentences]
    torch.cuda.synchronize()
    sec_per_sentence = (time.perf_counter() - start) / len(sentences)

    peak_gb = torch.cuda.max_memory_allocated() / GB
    print(f"info: peak memory: {peak_gb:.2f} GB, {sec_per_sentence:.3f} sec/sentence")

    # Free the model before the next mode
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, weights_gb, peak_gb, sec_per_sentence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Base model name or path")
    parser.add_argument("--text", type=str, help="Translate this single sentence instead of the built-in list")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("error: this comparison needs a CUDA GPU.")
    print(f"info: using device: cuda ({torch.cuda.get_device_name(0)})")

    sentences = [args.text] if args.text else SENTENCES
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model)
    tokenizer.src_lang = "en_XX"

    modes = ["fp16", "8bit", "4bit"]
    results = {}
    for mode in modes:
        results[mode] = run_mode(mode, args.model, tokenizer, sentences)

    print("\n--- Outputs ---")
    for i, sentence in enumerate(sentences):
        print(f"\nEnglish: {sentence}")
        for mode in modes:
            print(f"  {mode:>5}: {results[mode][0][i]}")

    print("\n--- Summary ---")
    print(f"{'mode':<8}{'weights GB':>12}{'peak GB':>10}{'sec/sentence':>15}")
    for mode in modes:
        _, weights_gb, peak_gb, sec = results[mode]
        print(f"{mode:<8}{weights_gb:>12.2f}{peak_gb:>10.2f}{sec:>15.3f}")
