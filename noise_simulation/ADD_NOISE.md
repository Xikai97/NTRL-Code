# Noise Simulation

This directory contains scripts to inject controlled noise into code-generation benchmark prompts (HumanEval, LeetCode, MBPP). Three noise types are supported; each can be applied independently via command-line flags.

## Noise Types

| Type | Flag | Description | Requirements |
|------|------|-------------|--------------|
| **Synonyms** | `--synonyms 1` | Word-level synonym replacement via [nlpaug](https://github.com/makcedward/nlpaug) | NLTK data (see below) |
| **Keyboard typos** | `--keyboard_typos 1` | Character-level keyboard-adjacent typos | nlpaug |
| **Paraphrasing** | `--paraphrasing 1` | LLM-based rephrasing of problem descriptions | Remote Qwen3-Coder-480B API (no GPU) |

Noise intensities are passed as integers (`1` = one application pass). Only flags with value `> 0` are applied. Multiple types can be combined in one run by setting several flags.

## Directory Layout

```
noise_simulation/
├── ADD_NOISE.md                      # Usage guide (this file)
├── real_world_noisy_prompts.jsonl    # Requirement-level Noise
├── script.sh                         # Example batch commands for all benchmarks
├── add_noise_HumanEval.py
├── add_noise_LeetCode.py
└── add_noise_MBPP.py
```

## Prerequisites

```bash
pip install nlpaug nltk
```

Paraphrasing no longer needs `vllm` or any local GPU — it calls a hosted
Qwen3-Coder-480B endpoint over HTTP (`http.client`, stdlib only).

For **synonym** augmentation, download NLTK WordNet (or point to your local copy):

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Update the NLTK paths at the top of each script if needed:

```python
nltk.data.path.append('/path/to/nltk_models')
os.environ["MODEL_DIR"] = '/path/to/nltk_models'
```

### Paraphrasing via remote API

Because there is no local Qwen3-Coder-480B deployment, `--paraphrasing 1`
delegates rephrasing to a hosted **Qwen3-Coder-480B** endpoint using an
OpenAI-compatible `chat/completions` API (same style as `check_api.py`).
Requests within a batch are issued concurrently through a thread pool, with
exponential-backoff retries; on persistent failure the original text is kept
unchanged so no sample is dropped.

All endpoint settings are constants at the top of each script and can be
overridden via environment variables:

| Env var | Default | Meaning |
|---------|---------|---------|
| `PARAPHRASE_API_HOST` | `xxx.ai` | API host |
| `PARAPHRASE_API_PATH` | `/v1/chat/completions` | API path |
| `PARAPHRASE_API_TOKEN` | *(baked-in key)* | Bearer token |
| `PARAPHRASE_API_MODEL` | `qwen3-coder-480b-a35b-instruct` | Model name |
| `PARAPHRASE_API_WORKERS` | `8` | Concurrent requests per batch |

No `CUDA_VISIBLE_DEVICES` is required for any noise type anymore.

## Default Data Paths

Each script defines anonymous placeholder paths. Override them with `--input` and `--output`, or edit the constants at the top of the file.

| Script | Input (default) | Field augmented |
|--------|-----------------|-----------------|
| `add_noise_HumanEval.py` | `.../HumanEval/data/humaneval-python.jsonl` | `prompt` |
| `add_noise_LeetCode.py` | `.../LeetCode/data/20240121-Jul.jsonl` | `prompt_sft` |
| `add_noise_MBPP.py` | `.../MBPP/data/mbpp.jsonl` | `text` |

Base path prefix (replace with your installation):

```
/path/to/deepseek-coder-noisy/Evaluation/<Benchmark>/data/
```

## Usage

Run from this directory. Example for a single noise type on MBPP:

```bash
python add_noise_MBPP.py --synonyms 1
```

### Output naming

The output filename is derived from `--output` by inserting noise flags before `.jsonl`:

```
mbpp-noisy.jsonl  →  mbpp-noisy_key0_syn1_para0.jsonl
```

Pattern: `_noisy_key{K}_syn{S}_para{P}.jsonl`

## Running script

No noise type requires a GPU anymore (paraphrasing uses a remote API):

```bash
# LeetCode
python add_noise_LeetCode.py --synonyms 1
python add_noise_LeetCode.py --keyboard_typos 1
python add_noise_LeetCode.py --paraphrasing 1

# HumanEval
python add_noise_HumanEval.py --synonyms 1
python add_noise_HumanEval.py --keyboard_typos 1
python add_noise_HumanEval.py --paraphrasing 1

# MBPP
python add_noise_MBPP.py --synonyms 1
python add_noise_MBPP.py --keyboard_typos 1
python add_noise_MBPP.py --paraphrasing 1
```

## Illustrative Examples

---

### Case  — HumanEval [Python/5] (`intersperse`)

**Original**

```python
from typing import List

def intersperse(numbers: List[int],  delimeter: int) -> List[int]:
	"""
	Insert a number 'delimeter' between every consecutive elements of input list 'numbers'
	>>> intersperse([], 4)
	[]
	>>> intersperse([1,2,3], 4)
	[1,4,2,4,3]
	"""
```

**Keyboard typos** (`--keyboard_typos 1`; `key1_syn0_para0`)

```python
from typing import List

def intersperse(numbers: List[int],  delimeter: int) -> List[int]:
	"""
	Insert a number 'delimeter' veY#een every consecutive elements of input list 'numbers'
	>>> inHersoe4s3([], 4)
	[]
	>>> 8ntersoFrEe([1,2,3], 4)
	[1,4,2,4,3]
	"""
```

**Synonyms** (`--synonyms 1`; `key0_syn1_para0`)

```python
from typing import List

def intersperse(numbers: List[int],  delimeter: int) -> List[int]:
	"""
	Insert a number 'delimeter' between every sequent elements of input list 'numbers'
	>>> intersperse([], 4)
	[]
	>>> intersperse([1,ii,3], 4)
	[1,4,2,4,3]
	"""
```

**Paraphrasing** (`--paraphrasing 1`; `key0_syn0_para1`)

```python
from typing import List

def intersperse(numbers: List[int],  delimeter: int) -> List[int]:
	"""
	Given a list of numbers called 'numbers' and a value 'delimeter', construct a new list by
	placing the 'delimeter' between each pair of adjacent elements in the original list.
	>>> intersperse([], 4)
	[]
	>>> intersperse([1,2,3], 4)
	[1,4,2,4,3]
	"""
```

