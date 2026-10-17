"""Score base vs adapter on held-out validation sentences (chrF / BLEU).
Needs:  pip install sacrebleu
Run from the repo root:  PYTHONPATH=. python scripts/eval_chrf.py
"""
import sacrebleu, torch
from train import prepare_dataset
from inference import load_model, translate

_, val = prepare_dataset()
val = val.select(range(min(100, len(val))))
src = [x["english"] for x in val]
ref = [x["telugu"] for x in val]

for name, path in [("base", "/nonexistent"), ("adapter", "./mbart-finetuned-en-te")]:
    m, t, d = load_model(path)
    hyp = [translate(s, m, t, d) for s in src]
    print("\n==", name, "chrF", round(sacrebleu.corpus_chrf(hyp, [ref]).score, 1),
          "BLEU", round(sacrebleu.corpus_bleu(hyp, [ref]).score, 1))
    for s, h, rf in list(zip(src, hyp, ref))[:5]:
        print(" EN: ", s, "\n HYP:", h, "\n REF:", rf)
    del m
    torch.cuda.empty_cache()
