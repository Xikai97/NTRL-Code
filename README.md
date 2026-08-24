# NTRL-Code

## News
- **[2026-08-20]** NTRL-Code has been accepted as a Main Conference paper in [EMNLP2026](https://2026.emnlp.org/)!

---

## Introduction

NTRL-Code extends [verl](https://github.com/volcengine/verl) for test-time reinforcement learning on noisy code-generation benchmarks (HumanEval, LeetCode, MBPP).

Noisy prompts are produced with the scripts in [`./noise_simulation/`](./noise_simulation/ADD_NOISE.md).

## Environment setup

You can use a custom conda env as below, or follow the official [verl](https://github.com/volcengine/verl) installation guide (dependencies, vLLM, Ray, etc.) and then install this repo in editable mode.

```bash
cd /path/to/NTRL_Code/verl

conda create -n ntrl-code python=3.10
conda activate ntrl-code

pip install -e .
```

---

## Step 1 — Prepare verl training data

All data-prep scripts live under `verl/data/`. Run them from that directory.

### 1a. Place benchmark JSONL files

Put clean or noisy JSONL under `verl/data/` (filename must match the prepare script):

| Benchmark | JSONL filename |
|-----------|----------------|
| HumanEval | `humaneval-python.jsonl` |
| LeetCode | `20240121-Jul.jsonl` |
| MBPP | `MBPP.jsonl` |

Example (after running `noise_simulation`):

```bash
cp /path/to/noisy_outputs/humaneval-python_noisy_key1_syn0_para0.jsonl \
   /path/to/NTRL_Code/verl/data/data/humaneval-python.jsonl
```

### 1b. JSON → NTRL format (`train.json` / `test.json`)

```bash
cd /path/to/NTRL_Code/verl/data

# HumanEval → ./humaneval-python/{train,test}.json
python ttrl_data_prepare_HumanEval.py \
  --saved_dir . \
  --test_filename humaneval-python.jsonl \
  --language python

# LeetCode → ./Leecode-20240121-Jul/{train,test}.json
python ttrl_data_prepare_LeeCode.py \
  --saved_dir . \
  --test_filename 20240121-Jul.jsonl

# MBPP → ./MBPP/{train,test}.json
python ttrl_data_prepare_MBPP.py \
  --saved_dir . \
  --test_filename MBPP.jsonl
```

Each script reads from `verl/data/data/<test_filename>` and writes under `verl/data/<source>/`, where `<source>` is derived from the filename (see table above).

### 1c. JSON → parquet for verl

Edit `data_source` in `preprocess_code.py` to match the output folder name, then run:

```bash
cd /path/to/NTRL_Code/verl/data

# Examples:
#   data_source = 'humaneval-python'
#   data_source = 'Leecode-20240121-Jul'
#   data_source = 'MBPP'
python preprocess_code.py
```

This creates `train.parquet` and `test.parquet` under `verl/data/<data_source>/`.

The training script expects parquet at `$DATA_LOCAL_DIR/$TASK/`, e.g. `verl/data/humaneval-python/train.parquet` when `TASK=humaneval-python`.

---

## Step 2 — NTRL-Code training

Entry point: **`verl/run_NTRL_code.sh`** (calls `examples/ntrl_code/Deepseek-Coder-1.3B/humaneval_ntrl_code.sh`).

```bash
cd /path/to/NTRL_Code/verl
bash run_NTRL_code.sh
```

Before running, edit paths in `examples/ntrl_code/Deepseek-Coder-1.3B/humaneval_ntrl_code.sh`:

| Variable | Description |
|----------|-------------|
| `DATA_LOCAL_DIR` | Default `/path/to/NTRL_Code/verl/data` |
| `BACKBONE_PATH` | Default `/path/to/models/deepseek-coder-1.3b` |
| `TASK` | Parquet subfolder name, e.g. `humaneval-python` |


To add more runs, extend `run_NTRL_code.sh` with additional `run_step` lines pointing at other scripts under `examples/ntrl_code/`.

---

## Step 3 — Merge FSDP checkpoints to Hugging Face format

```bash
cd /path/to/NTRL_Code/verl
bash test_verl_merge.sh
```

Edit `--local_dir` and `--target_dir` in `test_verl_merge.sh` to your checkpoint paths.

---

## Evaluation

Inference and pass@k evaluation follow the [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder) evaluation pipeline. Use merged checkpoints with your benchmark eval scripts.
