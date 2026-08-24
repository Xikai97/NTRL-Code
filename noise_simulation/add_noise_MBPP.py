import json
import argparse
import random
import time
import http.client
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import re
import os
import nltk
nltk.data.path.append('/path/to/nltk_data')
os.environ["MODEL_DIR"] = '/path/to/nltk_data'

DATASET_PATH = '/path/to/deepseek-coder-noisy/Evaluation/MBPP/data/mbpp.jsonl'
OUTPUT_PATH = '/path/to/deepseek-coder-noisy/Evaluation/MBPP/data/mbpp-noisy.jsonl'

# --- Paraphrasing via remote API (Qwen3-Coder-480B) ---
# The 480B model is not deployed locally, so paraphrasing is delegated to a
# hosted Qwen3-Coder-480B endpoint (OpenAI-compatible chat/completions API).
API_HOST = os.environ.get("PARAPHRASE_API_HOST", "xxxx.ai")
API_PATH = os.environ.get("PARAPHRASE_API_PATH", "/v1/chat/completions")
API_TOKEN = os.environ.get("PARAPHRASE_API_TOKEN", "sk-xxxxx")
API_MODEL = os.environ.get("PARAPHRASE_API_MODEL", "qwen3-coder-480b-a35b-instruct")
API_MAX_WORKERS = int(os.environ.get("PARAPHRASE_API_WORKERS", "8"))
API_MAX_RETRIES = 4
PARAPHRASE_SYSTEM_PROMPT = "You are a helpful assistant that rephrases programming problem descriptions."

keyboard_aug = None
synonym_aug = None

PARAPHRASING_PROMPT = """
You are tasked with rephrasing the following problem description for clarity and variety, as a form of noise enhancement. Ensure that:
The specific variable names (e.g., word) remain unchanged.
The provided examples, including their inputs, outputs, and explanations, are kept intact and unaltered.
The technical meaning and constraints of the problem are preserved, ensuring no ambiguity in interpretation.
Maintain the logical structure of the description while varying sentence structures, vocabulary, and rephrasing.

Here is the text to paraphrase:
\"{text}\"
Output ONLY the paraphrased text, with no preamble, quotes, or explanation.
"""


def find_different_words(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    unique_in_text1 = words1 - words2
    unique_in_text2 = words2 - words1
    
    return {
        "unique_in_text1": unique_in_text1,
        "unique_in_text2": unique_in_text2
    }
    

def _strip_wrapping_quotes(text):
    """Remove a single pair of surrounding quotes the model sometimes adds."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1].strip()
    return text


def call_paraphrase_api(text):
    """Call the hosted Qwen3-Coder-480B endpoint to paraphrase a single text.

    Retries on transient failures; on persistent failure returns the original
    text unchanged so the pipeline never drops a sample.
    """
    user_prompt = PARAPHRASING_PROMPT.format(text=text)
    payload = json.dumps({
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "top_p": 1,
    })
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(API_MAX_RETRIES):
        conn = None
        try:
            conn = http.client.HTTPSConnection(API_HOST, timeout=180)
            conn.request("POST", API_PATH, payload, headers)
            res = conn.getresponse()
            raw = res.read().decode("utf-8")
            if res.status >= 400:
                raise RuntimeError(f"HTTP {res.status} {res.reason}: {raw[:300]}")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return _strip_wrapping_quotes(content)
        except Exception as e:  # noqa: BLE001 - transient network/api errors
            last_err = e
            time.sleep(2 ** attempt)
        finally:
            if conn is not None:
                conn.close()
    print(f"[WARN] paraphrase failed after {API_MAX_RETRIES} retries, keeping original. Last error: {last_err}")
    return text


def add_keyboard_typos(text, intensity=1):
    text_ori = text
    text_segments = text.split('\n')
    for i in range(len(text_segments)):
        if len(text_segments[i]) > 20:
            for _ in range(intensity):
                text_aug_tmp = keyboard_aug.augment(text_segments[i])
                text_segments[i] = text_aug_tmp[0]
                text_segments[i] = re.sub(r'"(.*?)"', lambda m: f'"{m.group(1).strip()}"', text_segments[i])
    text = '\n'.join(text_segments)
    return text


def add_synonym(text, intensity=1):
    text_ori = text
    text_segments = text.split('\n')
    for i in range(len(text_segments)):
        if len(text_segments[i]) > 20:
            for _ in range(intensity):
                text_aug_tmp = synonym_aug.augment(text_segments[i])
                text_segments[i] = text_aug_tmp[0]
                text_segments[i] = re.sub(r'"(.*?)"', lambda m: f'"{m.group(1).strip()}"', text_segments[i])
        text = '\n'.join(text_segments)
    return text


def paraphrase_text_batch(prompts, intensity=1):
    """Using Qwen3-Coder-480B (remote API) to paraphrase a batch of texts.

    Requests within a batch run concurrently via a thread pool. `intensity`
    applies the paraphrasing pass that many times sequentially.
    """
    for _ in range(intensity):
        with ThreadPoolExecutor(max_workers=API_MAX_WORKERS) as executor:
            prompts = list(executor.map(call_paraphrase_api, prompts))
    return prompts


def process_dataset(input_path, output_path, noise_config, batch_size=32):
    noisy_data = []
    batch_prompts = []
    batch_indices = []
    paraphrasing_results = []
    
    with open(input_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
        
    for idx, item in enumerate(data):
        if "text" in item:
            prompt = item["text"]    
        if "keyboard_typos" in noise_config:
            item["text"] = add_keyboard_typos(prompt, intensity=noise_config["keyboard_typos"])
            prompt = item["text"]
        if "synonyms" in noise_config:
            item["text"] = add_synonym(prompt, intensity=noise_config["synonyms"])
            prompt = item["text"]
        if "paraphrasing" in noise_config:
            batch_prompts.append(prompt)
            batch_indices.append(idx)
            
        noisy_data.append(item)
        
        if len(batch_prompts) >= batch_size:
            paraphrasing_results.extend(paraphrase_text_batch(batch_prompts, intensity=noise_config["paraphrasing"]))
            batch_prompts = []
            batch_indices = []
    
    if "paraphrasing" in noise_config:
        # processing remaining prompts
        if batch_prompts:
            paraphrasing_results.extend(paraphrase_text_batch(batch_prompts, intensity=noise_config["paraphrasing"]))
        idx_list = list(range(len(paraphrasing_results)))
        for idx, result in zip(idx_list, paraphrasing_results):
            noisy_data[idx]["text"] = result
    
    with open(output_path, "w", encoding="utf-8") as file:
        for item in noisy_data:
            file.write(json.dumps(item) + "\n")
            
            
def main():
    parser = argparse.ArgumentParser(description="Add noise to HumanEval dataset")
    parser.add_argument("--input", type=str, default=DATASET_PATH, help="Path to the input JSONL dataset")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Path to save the noisy JSONL dataset")
    parser.add_argument("--keyboard_typos", type=int, default=0, help="Whether to add keyboard typos noise")
    parser.add_argument("--keyboard_typos_char", type=int, default=5, help="Intensity of max aug char")
    parser.add_argument("--keyboard_typos_word", type=int, default=1, help="Intensity of max aug word")
    parser.add_argument("--synonyms", type=int, default=0, help="Intensity of synonym noise")
    parser.add_argument("--paraphrasing", type=int, default=0, help="Intensity of paraphrasing noise")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for paraphrasing")
    args = parser.parse_args()
    
    noise_config = {
        "keyboard_typos": args.keyboard_typos,
        "synonyms": args.synonyms,
        "paraphrasing": args.paraphrasing
    }
    # filter out zero intensity noises
    noise_config = {k: v for k, v in noise_config.items() if v > 0}
    
    if "keyboard_typos" in noise_config.keys():
        global keyboard_aug
        keyboard_aug = nac.KeyboardAug(aug_char_max=args.keyboard_typos_char, aug_word_max=args.keyboard_typos_word)
    if "synonyms" in noise_config.keys():
        global synonym_aug
        synonym_aug = naw.SynonymAug(aug_p=0.1,
                                     aug_min=0,
                                     aug_max=2,
                                     stopwords=['key'])
    if "paraphrasing" in noise_config.keys():
        print(f"Paraphrasing via remote API: model={API_MODEL} host={API_HOST}")
    
    args.output = args.output.replace('.jsonl', f'_noisy_key{args.keyboard_typos}_syn{args.synonyms}_para{args.paraphrasing}.jsonl')
    
    print("Processing dataset with the following configurations:")
    print(f" Input path: {args.input}")
    print(f" Output path: {args.output}")
    print(f" Noise config: {noise_config}")
    print(f" Paraphrasing batch size: {args.batch_size}")
    
    process_dataset(args.input, args.output, noise_config, batch_size=args.batch_size)
    print(f"Dataset with noise added has been saved to {args.output}")
    

if __name__ == "__main__":
    main()
    