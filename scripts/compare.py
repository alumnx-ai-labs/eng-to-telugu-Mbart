"""Translate a list of sentences with the fine-tuned adapter and the base model, side by side.
Run from the repo root:  PYTHONPATH=. python scripts/compare.py
"""
import torch
from inference import load_model, translate

sents = [
    "Good morning everyone", "Thank you very much", "Where is the nearest hospital?",
    "I am very hungry, please give me some food", "I went to school yesterday",
    "She is reading a book", "We will meet tomorrow at five o'clock",
    "How much does this cost?", "What time does the train leave?",
    "The farmers are waiting for the rain because the crops need water.",
    "My brother works in a software company in Hyderabad.",
    "I will go to the temple daily", "How are you?",
]
out = {}
for name, path in [("ADAPTER", "./mbart-finetuned-en-te"), ("BASE", "/nonexistent")]:
    m, t, d = load_model(path)
    out[name] = [translate(s, m, t, d) for s in sents]
    del m
    torch.cuda.empty_cache()

for i, s in enumerate(sents):
    print("\nEN     :", s)
    print("ADAPTER:", out["ADAPTER"][i])
    print("BASE   :", out["BASE"][i])
