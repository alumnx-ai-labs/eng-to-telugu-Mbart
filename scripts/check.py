import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import transformers
print("transformers", transformers.__version__)
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
t = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
t.src_lang = "en_XX"; t.tgt_lang = "te_IN"
print("te_IN id:", t.convert_tokens_to_ids("te_IN"))
print("labels:", t(text_target="నమస్తే")["input_ids"])
print("OK")
