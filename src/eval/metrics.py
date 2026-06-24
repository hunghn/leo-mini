"""
Vietnamese VQA evaluation metrics.

Metrics (per paper requirements):
  ANLS  — Average Normalized Levenshtein Similarity (main metric)
  EM    — Exact Match (%)
  F1    — Token-level F1 (%)

Vietnamese normalization keeps tone marks (dấu), because meaning changes
completely without them (e.g. "ma" / "má" / "mà" / "mả" / "mã" / "mạ").
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

ANLS_THRESHOLD = 0.5  # standard TextVQA threshold


# ---------------------------------------------------------------------------
# Levenshtein edit distance  (O(n) space)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Vietnamese-aware normalization
# ---------------------------------------------------------------------------

def normalize_vi(text: str) -> str:
    """
    Normalize Vietnamese answer text.
    - NFC Unicode normalization FIRST (must be before regex to prevent combining
      diacritical marks U+0300–U+036F from being stripped as non-word characters;
      NFD "á" = a + combining-acute would lose its tone mark without this step)
    - Lowercase
    - Remove punctuation (keep letters, digits, spaces, Vietnamese chars)
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Per-question scores
# ---------------------------------------------------------------------------

def anls_score(
    prediction:   str,
    ground_truths: List[str],
    threshold:    float = ANLS_THRESHOLD,
) -> float:
    """ANLS for one question: max over all GT answers."""
    pred = normalize_vi(prediction)
    best = 0.0
    for gt in ground_truths:
        gt_n = normalize_vi(gt)
        max_len = max(len(pred), len(gt_n))
        if max_len == 0:
            nl = 0.0
        else:
            nl = _edit_distance(pred, gt_n) / max_len
        score = 1.0 - nl if nl < threshold else 0.0
        if score > best:
            best = score
    return best


def exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    """EM: 1.0 if normalized prediction matches any normalized GT answer."""
    pred = normalize_vi(prediction)
    return 1.0 if any(normalize_vi(gt) == pred for gt in ground_truths) else 0.0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Token-level F1: max over GT answers (bag-of-words overlap)."""
    pred_toks = normalize_vi(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_toks = normalize_vi(gt).split()
        if not pred_toks and not gt_toks:
            return 1.0
        if not pred_toks or not gt_toks:
            continue
        # Token overlap using multiset intersection
        from collections import Counter
        pred_cnt = Counter(pred_toks)
        gt_cnt   = Counter(gt_toks)
        common   = sum((pred_cnt & gt_cnt).values())
        if common == 0:
            continue
        prec = common / len(pred_toks)
        rec  = common / len(gt_toks)
        f1   = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

def compute_dataset_metrics(
    predictions:   List[str],
    ground_truths: List[List[str]],
    threshold:     float = ANLS_THRESHOLD,
    question_ids:  Optional[List[str]] = None,
) -> Dict:
    """
    Aggregate ANLS, EM, F1 over a full split.

    Returns:
        {
            "anls":            float (0–1),
            "em":              float (0–1),
            "f1":              float (0–1),
            "anls_pct":        float (0–100),
            "em_pct":          float (0–100),
            "f1_pct":          float (0–100),
            "n_samples":       int,
            "per_sample":      [ {"qid", "pred", "anls", "em", "f1"}, ... ]
        }
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truths ({len(ground_truths)}) must match"
        )
    if not predictions:
        return {"anls": 0.0, "em": 0.0, "f1": 0.0, "anls_pct": 0.0, "em_pct": 0.0, "f1_pct": 0.0, "n_samples": 0, "per_sample": []}

    per_sample = []
    anls_sum = em_sum = f1_sum = 0.0

    for idx, (pred, gts) in enumerate(zip(predictions, ground_truths)):
        a = anls_score(pred, gts, threshold)
        e = exact_match_score(pred, gts)
        f = f1_score(pred, gts)
        anls_sum += a
        em_sum   += e
        f1_sum   += f
        record = {
            "idx":  idx,
            "pred": pred,
            "gts":  gts,
            "anls": round(a, 4),
            "em":   round(e, 4),
            "f1":   round(f, 4),
        }
        if question_ids is not None:
            record["qid"] = question_ids[idx]
        per_sample.append(record)

    n = len(predictions)
    anls = anls_sum / n
    em   = em_sum   / n
    f1   = f1_sum   / n

    return {
        "anls":      round(anls, 4),
        "em":        round(em,   4),
        "f1":        round(f1,   4),
        "anls_pct":  round(anls * 100, 2),
        "em_pct":    round(em   * 100, 2),
        "f1_pct":    round(f1   * 100, 2),
        "n_samples": n,
        "per_sample": per_sample,
    }
