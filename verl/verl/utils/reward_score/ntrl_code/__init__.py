"""Provides a code-generation grading function with static rewards only.
No sandbox execution. Rewards are: format extractability, non-echo, similarity to ground truth.
"""

import traceback
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union

# ------------------------
# Utilities
# ------------------------

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code_from_response(passage: str) -> Optional[str]:
    if not passage:
        return None
    m = CODE_FENCE_RE.findall(passage)
    if m:
        # If multiple fenced blocks exist, use the first non-empty one
        for seg in m:
            seg = seg.strip()
            if seg:
                return seg
    # Fallback: heuristically extract code-like text
    lines = [ln for ln in passage.splitlines() if ln.strip()]
    if not lines:
        return None
    kept = []
    score = 0
    for ln in lines:
        kept.append(ln)
        if re.search(r"\b(def|class|import|from|return|if|for|while|try|except|with|lambda)\b", ln):
            score += 1
        if re.match(r"\s{2,}\S", ln):
            score += 0.2
    code = "\n".join(kept).strip()
    return code if (code and score >= 1) else None

def normalize_for_similarity(s: str) -> str:
    # Coarse normalization: strip comments/whitespace and lowercase
    s = re.sub(r"#.*", "", s)
    s = re.sub(r"//.*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r'"""[\s\S]*?"""', "", s)
    s = re.sub(r"'''[\s\S]*?'''", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def levenshtein(a: str, b: str) -> int:
    # Lightweight implementation, sufficient for scoring
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    # Space optimized to O(min(len(a), len(b)))
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def jaccard_tokens(a: str, b: str) -> float:
    # Coarse tokenization over identifiers/keywords
    toks_a = set(re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:]", a))
    toks_b = set(re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:]", b))
    if not toks_a and not toks_b:
        return 1.0
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    uni = len(toks_a | toks_b)
    return inter / max(1, uni)

def compute_similarity_score(pred_code: str, gt_code: Optional[str]) -> float:
    if not gt_code:
        return 0.0
    pred_norm = normalize_for_similarity(pred_code)
    gt_norm = normalize_for_similarity(gt_code)
    if not pred_norm and not gt_norm:
        return 1.0
    # Normalized edit-distance similarity
    d = levenshtein(pred_norm, gt_norm)
    max_len = max(1, max(len(pred_norm), len(gt_norm)))
    sim_lev = 1.0 - min(1.0, d / max_len)
    # Token Jaccard
    sim_jac = jaccard_tokens(pred_norm, gt_norm)
    # Combine scores
    return 0.5 * sim_lev + 0.5 * sim_jac

def compute_non_echo_score(code: str, prompt: Optional[str]) -> float:
    """
    Simple non-echo heuristic:
    - Return 1.0 if no prompt/context is provided
    - Penalize overlap between prompt and generated code (higher overlap -> lower score)
    """
    if not prompt:
        return 1.0

    def normalize(s: str) -> str:
        s = re.sub(r"#.*", "", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    code_norm = normalize(code)
    prompt_norm = normalize(prompt)

    if not code_norm or not prompt_norm:
        return 1.0

    # Sliding-window overlap count
    matches = 0
    window = max(10, min(80, len(prompt_norm) // 10 + 10))
    step = max(5, window // 2)
    for i in range(0, max(0, len(prompt_norm) - window + 1), step):
        chunk = prompt_norm[i:i + window]
        if chunk in code_norm:
            matches += 1
    overlap_ratio = matches / (1 + len(prompt_norm) // step)

    # Detect echoing the problem statement via print()
    echo_print = 1 if re.search(r'print\(.{0,40}(problem|prompt|question|题目|问题).{0,40}\)', code_norm) else 0

    score = 1.0 - min(1.0, overlap_ratio + 0.3 * echo_print)
    return max(0.0, min(1.0, score))

def parse_ground_truth_code(ground_truth: Union[str, dict, list]) -> Optional[str]:
    """
    Extract reference code from ground_truth for static comparison:
    - dict with 'code' field -> use it
    - str -> prefer fenced code; otherwise use the string as reference
    - other types -> None
    """
    if isinstance(ground_truth, dict):
        if "code" in ground_truth and isinstance(ground_truth["code"], str):
            return ground_truth["code"]
        # Also support {"reference": "..."} style fields
        for k in ["reference", "gt_code", "solution", "answer"]:
            if k in ground_truth and isinstance(ground_truth[k], str):
                return ground_truth[k]
        # Try code fences inside serialized dict
        text_blob = json.dumps(ground_truth, ensure_ascii=False)
        m = CODE_FENCE_RE.findall(text_blob)
        if m:
            for seg in m:
                if seg.strip():
                    return seg.strip()
        return None
    elif isinstance(ground_truth, str):
        m = CODE_FENCE_RE.findall(ground_truth)
        if m:
            for seg in m:
                if seg.strip():
                    return seg.strip()
        return ground_truth.strip()
    else:
        return None

# ------------------------
# Scoring (static only)
# ------------------------

def compute_code_score(model_response: str, gt_spec: Union[str, dict, list], fast: bool = True, extra_info: Optional[dict]=None):
    """
    Static-only scoring:
      - format_score: whether code was extracted (0/1)
      - non_echo_score: penalty for echoing prompt/context ([0,1], higher is better)
      - similarity_score: text similarity to ground-truth code ([0,1])
    Total: score = w_format*format + w_echo*non_echo + w_sim*similarity
    Default weights: format=0.25, non_echo=0.25, similarity=0.5; override via extra_info["weights"].
    """
    code = extract_code_from_response(model_response)

    # Weights
    weights = {"format": 0.25, "non_echo": 0.25, "similarity": 0.5}
    prompt_for_echo = None
    if isinstance(extra_info, dict):
        w = extra_info.get("weights")
        if isinstance(w, dict):
            weights["format"] = float(w.get("format", weights["format"]))
            weights["non_echo"] = float(w.get("non_echo", weights["non_echo"]))
            weights["similarity"] = float(w.get("similarity", weights["similarity"]))
        prompt_for_echo = extra_info.get("prompt") or extra_info.get("context")

    gt_code = parse_ground_truth_code(gt_spec)

    format_score = 1.0 if (code is not None and code.strip()) else 0.0
    if format_score == 0.0:
        sub = {"format_score": 0.0, "non_echo_score": 0.0, "similarity_score": 0.0}
        
        # total = weights["format"] * sub["format_score"] + weights["non_echo"] * sub["non_echo_score"] + weights["similarity"] * sub["similarity_score"]
        score = weights["similarity"] * sub["similarity_score"]
        format_score = weights["format"] * sub["format_score"] + weights["non_echo"] * sub["non_echo_score"]
        
        return {
            # "score": float(total),
            # "format_score": 0.0,
            "score": score,
            "format_score": format_score,
            "acc": False,  # No execution; acc does not mean "all tests passed"
            # "extracted_gt": {"gt_code": gt_code, "sub_rewards": sub},
            "extracted_gt": gt_code,
            "pred": "",
        }

    non_echo_score = compute_non_echo_score(code, prompt_for_echo)
    similarity_score = compute_similarity_score(code, gt_code) if gt_code else 0.0

    sub = {
        "format_score": float(format_score),
        "non_echo_score": float(non_echo_score),
        "similarity_score": float(similarity_score),
    }
    # total = weights["format"] * sub["format_score"] + weights["non_echo"] * sub["non_echo_score"] + weights["similarity"] * sub["similarity_score"]
    score = weights["similarity"] * sub["similarity_score"] + weights["format"] * sub["format_score"] + weights["non_echo"] * sub["non_echo_score"]
    # score = weights["similarity"] * sub["similarity_score"]
    format_score = weights["format"] * sub["format_score"] + weights["non_echo"] * sub["non_echo_score"]
    
    return {
        # "score": float(total),
        # "format_score": 1.0,
        "score": score,
        "format_score": format_score,
        "acc": False,  # No tests run; cannot define pass@all-tests
        # "extracted_gt": {"gt_code": gt_code, "sub_rewards": sub},
        "extracted_gt": gt_code,
        "pred": code,
    }

# ------------------------
# Public API: keep signature and returns
# ------------------------

def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    Static reward:
    - Scores code extractability, non-echo vs prompt, and text similarity to reference
    - Does not execute code or run tests
    - Tune weights via extra_info["weights"]; pass original problem via extra_info["prompt"] or ["context"]
    """
    try:
        gt_spec = ground_truth  # Static scoring does not merge tests, etc.
        res = compute_code_score(solution_str, gt_spec, fast=True, extra_info=extra_info)

        if isinstance(res, dict):
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise