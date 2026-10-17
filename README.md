# English → Telugu mBART (LoRA): Training and Inference on AWS g4dn.xlarge

Repo: https://github.com/alumnx-ai-labs/eng-to-telugu-Mbart

This guide runs `train.py` (LoRA fine-tuning of `facebook/mbart-large-50-many-to-many-mmt`) and `inference.py` on one g4dn.xlarge (1x NVIDIA T4 16 GB, 4 vCPU, 16 GB RAM, Ubuntu 22.04).

The original repo needed a few fixes to work on a fresh install. **They are already applied in this repo** (Step 4 lists them):

| Fix | Why |
|---|---|
| Pin `transformers<5` | transformers 5.x needs PyTorch >= 2.6; the cu121 PyTorch build is 2.5.1 and the import crashes |
| `tokenizer=` → `processing_class=` | `Seq2SeqTrainer` API change |
| `lang_code_to_id` → `convert_tokens_to_ids` | removed in newer transformers |
| Labels: `text_target=` and pad → `-100` | without it the labels start with the English token and padding dominates the loss (loss stays near 10, the model learns nothing) |
| LoRA (r=8, q_proj/v_proj), lr 2e-4, 5 epochs | small trainable adapter, fits the T4 easily |
| Adapter-aware `inference.py` | loads base model, attaches adapter, merges |

---

## 0. What you need

- An AWS account that can launch `g4dn.xlarge` (if launch fails with a vCPU limit error, request a quota increase for "Running On-Demand G and VT instances" to 4+ vCPUs).
- Internet access from the instance (Hugging Face downloads the model and the dataset).
- The dataset is `Shreya3095/TeluguTranslator` (about 1,100 usable pairs after cleaning).

## 1. Launch the instance

1. EC2 → Launch instance.
2. AMI: **Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)**. A plain Ubuntu 22.04 image also works if the NVIDIA driver is installed (`nvidia-smi` must work).
3. Instance type: `g4dn.xlarge`.
4. Storage: **60 GB gp3 or more**. The default 8 to 20 GB is too small: the venv needs about 7 GB and the base model about 2.4 GB.
5. Security group: allow SSH (22) from your IP only.
6. Connect with SSH (`ssh -i key.pem ubuntu@<public-ip>`) or EC2 Instance Connect from the console.

## 2. Check GPU and disk

```bash
nvidia-smi        # must list a Tesla T4
df -h /           # need ~20 GB free
python3 --version # 3.10 on Ubuntu 22.04
```

`nvidia-smi` should show nearly 0 MiB used. If another process holds the GPU (for example Ollama or vLLM on a shared machine), training will run out of memory. Free the GPU first (see Troubleshooting).

## 3. Create a work folder and virtual environment

```bash
sudo apt update && sudo apt install -y python3.10-venv tmux git
# If a "Daemons using outdated libraries" dialog appears: leave boxes unchecked, Tab to <Ok>, Enter.

mkdir -p ~/mbart && cd ~/mbart
git clone https://github.com/alumnx-ai-labs/eng-to-telugu-Mbart.git code
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch first, with the CUDA 12.1 build:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then the rest (transformers is pinned below 5 in requirements.txt):
cd code && pip install -r requirements.txt && cd ..

python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
# expected: 2.5.1+cu121 True
```

If it prints `False`, check `nvidia-smi` first, then reinstall torch with the command above.

## 4. Patches (already included in this repo)

Nothing to apply. Verify they are present:

```bash
cd ~/mbart/code
grep -n "peft\|processing_class\|text_target\|-100\|convert_tokens\|learning_rate\|default=5" train.py inference.py
```

You should see matching lines in both files (`text_target` and `-100` on two lines in `train.py`).

## 5. Pre-flight check (about 1 minute, no training)

```bash
cd ~/mbart/code
python scripts/check.py
```

Expected: `torch 2.5.1+cu121 cuda True`, `transformers 4.57.x`, `te_IN id: 250045`, and labels whose **first number is 250045**, then `OK`.

## 6. Train

Run inside tmux so a dropped SSH or browser session does not kill the job.

```bash
tmux new -s train
```

Inside tmux (a new shell, so source the venv again):

```bash
cd ~/mbart/code
source ../venv/bin/activate
python train.py --epochs 5 --batch_size 8
```

- Detach: **Ctrl+B**, then **D**. Reattach: `tmux attach -t train`.
- Watch GPU memory from a second connection: `watch -n 2 nvidia-smi`.
- If you see `CUDA out of memory`, rerun with `--batch_size 4`.
- The first run downloads the base model (about 2.4 GB) and the dataset.
- Training on about 1,000 pairs for 5 epochs took about 3 minutes in our run.

What the log should look like:

```
trainable params: 1,179,648 || all params: 612,059,136 || trainable%: 0.1927
info: dataset size: 1101
{'loss': ..., 'epoch': 0.4}
{'eval_loss': ..., 'epoch': 1.0}
...
info: saving model to ./mbart-finetuned-en-te...
info: training complete.
```

**Health check on the loss.** The loss should start in the low single digits and the `eval_loss` should fall over the epochs. If the loss stays near **10** and the eval loss is flat, the labels fix was not applied. Re-check `grep -n "text_target" train.py`.

Output: the adapter is saved to `~/mbart/code/mbart-finetuned-en-te/` (tens of MB).

## 7. Inference

Run from the `code` folder (the model path is relative):

```bash
cd ~/mbart/code
source ../venv/bin/activate

python inference.py --text "How are you?"     # one sentence
python inference.py                            # interactive; type q to quit
```

You should see `info: loading model from ./mbart-finetuned-en-te...`. If you see "fine-tuned model not found", the adapter folder is missing and the script is using the base model.

Compare with the base model (a non-existent folder forces the fallback):

```bash
python inference.py --text "I will go to the temple daily"
python inference.py --model_dir /nonexistent --text "I will go to the temple daily"
```

Many sentences at once, adapter vs base, side by side:

```bash
PYTHONPATH=. python scripts/compare.py
```

Score both on held-out validation data (needs `pip install sacrebleu`):

```bash
PYTHONPATH=. python scripts/eval_chrf.py
```

Inference loads the merged model in fp32 (about 2.4 GB of GPU memory). It fits easily on a free T4.

## 8. Save your work, then stop paying

The adapter is small. Copy it off the instance before you stop or terminate:

```bash
cd ~/mbart/code
tar czf ~/mbart-adapter.tgz mbart-finetuned-en-te
# from your own PC:
#   scp -i key.pem ubuntu@<public-ip>:~/mbart-adapter.tgz .
# or, with the AWS CLI configured on the instance:
#   aws s3 cp ~/mbart-adapter.tgz s3://<your-bucket>/
```

- **Stop** the instance to pause compute billing (EBS storage still bills).
- **Terminate** it when you are completely done, and delete any extra EBS volumes you created.
- Note: the g4dn's built-in 125 GB NVMe instance store is wiped on stop/terminate. Never keep the only copy of the adapter there.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NameError: name 'nn' is not defined` while importing peft/transformers; log says "PyTorch >= 2.6 is required but found 2.5.1" | transformers 5.x installed | `pip install "transformers>=4.48,<5" "datasets>=3.2,<5" "huggingface_hub<1"` |
| `PermissionError ... /data/hf-cache` | A machine-wide `TRANSFORMERS_CACHE` is set (typical on shared servers) and overrides `HF_HOME` | In every new shell/tmux: `unset TRANSFORMERS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE` then `export HF_HOME=~/mbart/hf_cache` |
| `CUDA out of memory` at start | Another process (Ollama, vLLM) holds the GPU, or batch too big | Run `nvidia-smi`; free the GPU or use `--batch_size 4` |
| Terminal shows a `>` prompt and ignores commands | An unclosed quote or heredoc | Press Ctrl+C. If stuck, type `"` and Enter, then Ctrl+C, or reconnect |
| Loss stays near 10, eval loss flat | Label fix missing | Check `grep -n "text_target" train.py`; use the version from this repo and retrain |
| `inference.py` says "fine-tuned model not found" | Wrong working directory or missing adapter folder | `cd ~/mbart/code` and run `ls mbart-finetuned-en-te` |
| `no space left on device` | Root volume too small | Grow the EBS volume, then `sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1` |

### Using a shared GPU server

If another service already uses the GPU:
1. Get the owner's okay before stopping anything.
2. Check what restarts it: `crontab -l` and the scripts it calls.
3. Unload or stop it for the run (`ollama stop <model>`, `sudo systemctl stop ollama`), confirm with `nvidia-smi`.
4. Afterwards restore it (`sudo systemctl start ollama`, re-enable any cron line you paused).

If you cannot get the GPU, launch your own g4dn.xlarge as in Step 1.

## Quick reference

```bash
# every new shell:
cd ~/mbart/code && source ../venv/bin/activate
unset TRANSFORMERS_CACHE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE
export HF_HOME=~/mbart/hf_cache

python train.py --epochs 5 --batch_size 8
python inference.py --text "How are you?"
```

## Other methods (one script each)

| Method | Command | Output folder |
|---|---|---|
| Full fine-tuning (all weights) | `python train_full.py --epochs 5` (options: `--batch_size`, `--grad_accum`, `--lr`, `--optim adafactor`, `--grad_ckpt`) | `./mbart-full-finetuned-en-te` |
| LoRA | `python train.py --epochs 5 --batch_size 8` | `./mbart-finetuned-en-te` |
| QLoRA (4-bit base + LoRA) | `pip install -r requirements-quant.txt` then `python train_qlora.py --epochs 5` | `./mbart-qlora-en-te` |
| Quantization comparison (fp16 / 8-bit / 4-bit, base model) | `pip install -r requirements-quant.txt` then `python quantize_compare.py` | none (prints a table) |

Run a saved model with `inference.py`:

```bash
python inference.py --model_dir ./mbart-full-finetuned-en-te --text "How are you?"
python inference.py --model_dir ./mbart-finetuned-en-te --text "How are you?"
python inference.py --model_dir ./mbart-qlora-en-te --text "How are you?"
```

Full fine-tuning needs roughly 12-14 GB on a 16 GB T4, so run it with the GPU to yourself. On CUDA OOM use `--batch_size 2 --grad_accum 8`, `--grad_ckpt`, or `--optim adafactor`.

Note: `train_full.py`, `train_qlora.py` and `quantize_compare.py` have not yet been run on the T4.
