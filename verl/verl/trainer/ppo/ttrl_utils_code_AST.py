import re
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional
import numpy as np
import ast
import hashlib

# === AST fingerprinting utilities ===

class _NormalizeNames(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.var_map = 0
        self.func_map = 0
        self.cls_map = 0
        self._var_id = 0
        self._func_id = 0
        self._cls_id = 0
        
    def _name(self, name, table, counter_attr, prefix):
        if name not in table:
            setattr(self, counter_attr, getattr(self, counter_attr) + 1)
            table[name] = f"{prefix}{getattr(self, counter_attr)}"
        return table[name]
    
    def visit_Name(self, node):
        node = self.generic_visit(node)
        node.id = self._name(getattr(node, 'id', ""), self.var_map, '_var_id', 'VAR')
        return node
    
    def visit_arg(self, node):
        node = self.generic_visit(node)
        node.arg = self._name(getattr(node, 'arg', ""), self.var_map, '_var_id', 'VAR')
        return node
    
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.attr, str):
            node.attr = "ATTR"
        return node
    
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        node.name = self._name(getattr(node, 'name', ""), self.func_map, '_func_id', 'FUNC')
        return node
    
    def visit_AsyncFunctionDef(self, node):
        node = self.generic_visit(node)
        node.name = self._name(getattr(node, 'name', ""), self.func_map, '_func_id', 'FUNC')
        return node
    
    def visit_ClassDef(self, node):
        node = self.generic_visit(node)
        node.name = self._name(getattr(node, 'name', ""), self.cls_map, '_cls_id', 'CLS')
        return node
    
    def visit_Constant(self, node):
        val = getattr(node, 'value', None)
        if isinstance(val, bool):
            rep = "CONST_BOOL"
        elif isinstance(val, (int, float, complex)):
            rep = "CONST_NUM"
        elif isinstance(val, str):
            rep = "CONST_STR"
        elif val is None:
            rep = "CONST_NONE"
        else:
            rep = "CONST_OTHER"
        return ast.copy_location(ast.Name(id=rep, ctx=ast.Load()), node)
    

def _ast_fingerprint_python(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tree = _NormalizeNames().visit(tree)
    ast.fix_missing_locations(tree)
    dumped = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.md5(dumped.encode('utf-8')).hexdigest()


def _ast_fingerprint(code: str, language: str = "python") -> Optional[str]:
    if language.lower() == "python":
        return _ast_fingerprint_python(code)
    else:
        return None
    

def _cluster_codes_by_fingerprint_then_jaccard(
    raw_codes: List[str],
    norm_codes: List[str],
    language: str = "python",
    jaccard_threshold: float = 0.92,
) -> Tuple[List[int], List[str]]:
    
    assert len(raw_codes) == len(norm_codes)
    n = len(raw_codes)
    
    fp_to_cluster: Dict[str, int] = {}
    assign: List[int] = [-1] * n
    centers_raw: List[str] = []
    centers_norm: List[str] = []
    
    for i, (raw, norm) in enumerate(zip(raw_codes, norm_codes)):
        fp = _ast_fingerprint(raw, language=language)
        if fp is not None:
            if fp in fp_to_cluster:
                cid = fp_to_cluster[fp]
            else:
                cid = len(centers_raw)
                fp_to_cluster[fp] = cid
                centers_raw.append(raw)
                centers_norm.append(norm)
            assign[i] = cid
            
    from math import inf
    for i, (raw, norm) in enumerate(zip(raw_codes, norm_codes)):
        if assign[i] != -1:
            continue
        found = False
        for cid, center_norm in enumerate(centers_norm):
            if center_norm:
                if _codes_are_similar(norm, center_norm, jaccard_threshold=jaccard_threshold):
                    assign[i] = cid
                    found = True
                    break
        if not found:
            cid = len(centers_raw)
            centers_raw.append(raw)
            centers_norm.append(norm)
            assign[i] = cid
    
    return assign, centers_raw




# Code-fence extraction (same pattern as above)
CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", re.DOTALL)

def extract_code_from_response(passage: str) -> Optional[str]:
    if not passage:
        return None
    m = CODE_FENCE_RE.findall(passage)
    if m:
        for seg in m:
            seg = seg.strip()
            if seg:
                return seg
    # Fallback heuristic: keep lines that look like Python code
    lines = [ln for ln in passage.splitlines() if ln.strip()]
    kept = []
    hit = 0
    for ln in lines:
        kept.append(ln)
        if re.search(r"\b(def|class|import|from|return|if|for|while|try|except|with|lambda)\b", ln):
            hit += 1
        if re.match(r"\s{2,}\S", ln):
            hit += 0.2
    code = "\n".join(kept).strip()
    return code if (code and hit >= 1) else None

def normalize_code_for_vote(code: str, language: str = "python") -> str:
    """
    Lightweight normalization that preserves semantics while removing surface differences:
      - Strip trailing/extra whitespace and normalize newlines
      - Collapse consecutive blank lines
      - Remove comments (#, //, /* */, triple-quoted docstrings)
      - Optional lowercasing (disabled by default to avoid losing identifier information)
    """
    s = code

    # Remove multiline comments (Python triple quotes, C-style)
    s = re.sub(r'"""[\s\S]*?"""', "", s)
    s = re.sub(r"'''[\s\S]*?'''", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)

    # Remove single-line comments (Python/C/C++/JS)
    s = re.sub(r"#.*", "", s)
    s = re.sub(r"//.*", "", s)

    # Strip trailing whitespace and extra blank lines
    s = "\n".join([ln.rstrip() for ln in s.splitlines()])
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s).strip()

    return s

def _codes_are_similar(a: str, b: str, jaccard_threshold: float = 0.92) -> bool:
    """
    Token-level Jaccard similarity; treats near-duplicate implementations as one vote.
    Default threshold is high (0.92) to merge only minor surface differences.
    """
    fa = _ast_fingerprint(a)
    fb = _ast_fingerprint(b)
    if fa is not None and fb is not None:
        return fa == fb
    
    toks_a = set(re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:;]", a))
    toks_b = set(re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:;]", b))
    if not toks_a and not toks_b:
        return True
    if not toks_a or not toks_b:
        return False
    inter = len(toks_a & toks_b)
    uni = len(toks_a | toks_b)
    jac = inter / max(1, uni)
    return jac >= jaccard_threshold

# === TTRL ground-truth application (same interface/flow as the math version) ===

def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer, language: str = "python", jaccard_threshold: float = 0.92):
    """
    Write code-majority vote results back into batch as the new ground_truth.
    Input/output contract matches the original function.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n, language=language, jaccard_threshold=jaccard_threshold)

    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch

def _batch_majority_vote(model_outputs: List[str], n: int, language: str = "python", jaccard_threshold: float = 0.92) -> Tuple[List[str], List[float]]:
    """
    Batch interface aligned with the math version, adapted for code:
      - Extract code from responses
      - Normalize
      - Optionally merge similar solutions
      - Majority vote
    """
    majority_gt_list: List[str] = []
    majority_ratio_list: List[float] = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs, language=language, jaccard_threshold=jaccard_threshold)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)

    return majority_gt_list, majority_ratio_list

def _majority_vote(model_outputs: List[str], language: str = "python", jaccard_threshold: float = 0.92) -> Tuple[str, float]:
    assert len(model_outputs) > 0
    
    extracted = [extract_code_from_response(resp) for resp in model_outputs]
    
    vaild_pairs = [(c, normalize_code_for_vote(c, language=language)) for c in extracted if c is not None and c.strip()]

    if len(vaild_pairs) == 0:
        return "None", 0.0

    raw_codes = [p[0] for p in vaild_pairs]
    norm_codes = [p[1] for p in vaild_pairs]
    
    assign, centers_raw = _cluster_codes_by_fingerprint_then_jaccard(
        raw_codes,
        norm_codes,
        language=language,
        jaccard_threshold=jaccard_threshold,
    )
    
    counter = Counter(assign)
    majority_cluster, majority_count = counter.most_common(1)[0]
    
    majority_code = centers_raw[majority_cluster]
    majority_ratio = majority_count / len(model_outputs)
    
    return majority_code, float(majority_ratio)


# === Metrics Computation ===


# ===== Code-similarity-based grader =====

def _normalize_for_similarity(s: str) -> str:
    # Strip comments and whitespace; lowercase
    s = re.sub(r'"""[\s\S]*?"""', "", s)     # Python multiline strings
    s = re.sub(r"'''[\s\S]*?'''", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)     # C/JS block comments
    s = re.sub(r"#.*", "", s)                # Python line comments
    s = re.sub(r"//.*", "", s)               # C/JS line comments
    s = re.sub(r"[ \t]+", " ", s)            # Collapse internal whitespace
    s = "\n".join(ln.rstrip() for ln in s.splitlines())
    s = re.sub(r"\n{3,}", "\n\n", s)         # Limit extra blank lines
    return s.strip().lower()

def _levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def _jaccard_tokens(a: str, b: str) -> float:
    toks_a = set(re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:;]", a))
    toks_b = set(re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|[-+*/%(){}\[\].,:;]", b))
    if not toks_a and not toks_b:
        return 1.0
    if not toks_a or not toks_b:
        return 0.0
    inter = len(toks_a & toks_b)
    uni = len(toks_a | toks_b)
    return inter / max(1, uni)

def code_similarity(code_a: str, code_b: str) -> float:
    """
    Similarity in [0, 1] between two code snippets:
    - 0.6 * normalized edit distance + 0.4 * token Jaccard
    """
    if code_a is None or code_b is None:
        return 0.0
    a = _normalize_for_similarity(code_a)
    b = _normalize_for_similarity(code_b)
    if not a and not b:
        return 1.0
    d = _levenshtein(a, b)
    max_len = max(1, max(len(a), len(b)))
    sim_lev = 1.0 - min(1.0, d / max_len)
    sim_jac = _jaccard_tokens(a, b)
    return 0.6 * sim_lev + 0.4 * sim_jac

# Note: grade() signature unchanged; comparison is similarity-based internally
def grade(model_code: str, gt_code: str, fast: bool = True, threshold: float = None, extra_info: Dict = None) -> bool:
    """
    Similarity-based code "equivalence" check:
    - Default threshold 0.9; override via extra_info["code_grade_threshold"] or threshold
    - fast is kept for backward compatibility and does not affect logic
    """
    thr = 0.9
    if threshold is not None:
        thr = float(threshold)
    elif isinstance(extra_info, dict) and "code_grade_threshold" in extra_info:
        thr = float(extra_info["code_grade_threshold"])
    sim = code_similarity(model_code, gt_code)
    return sim >= thr

# === Metrics Computation ===

def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


from collections import Counter

def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    # Similarity-based grade; inject custom threshold via extra_info or closure if needed
    hit_rate = 1.0 if grade(majority_label, gt_label, fast=True) else 0.0

    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric