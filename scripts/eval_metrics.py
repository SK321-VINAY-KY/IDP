"""
File: eval_metrics.py
Purpose: Standardized evaluation metrics for IDP evaluation harness (Phase 3).
         Provides robust, hardened evaluation functions:
         1. field_match & evaluate_fields: Schema-level field extraction accuracy with
            fuzzy matching, numeric/currency normalization, date formatting resilience,
            and token overlap heuristics.
         2. evidence_groundedness: Verification of graph memory node citations,
            measuring what percentage of canonical nodes trace back to concrete
            document evidence / bounding boxes.
         3. calculate_cer_wer: Levenshtein-distance Character Error Rate (CER) and
            Word Error Rate (WER) for transcript / OCR evaluation when reference
            transcripts are available.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# String & Value Normalization Utilities
# ---------------------------------------------------------------------------

def normalize_text(val: Any) -> str:
    """Strip whitespace, accents, and punctuation for robust comparison."""
    if val is None:
        return ""
    text = str(val).strip()
    text = unicodedata.normalize("NFKD", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def normalize_numeric(val: Any) -> Optional[float]:
    """Extract numeric value stripped of currency symbols, commas, and formatting."""
    if val is None:
        return None
    s = str(val).strip()
    # If it is a valid date, don't treat as single number
    if normalize_date(s) is not None:
        return None
    # Strip common currency symbols, commas, trailing /-
    s = re.sub(r"[₹$€£,/\-]|(?:inr|rs\.?|usd)", "", s, flags=re.IGNORECASE).strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def normalize_date(val: Any) -> Optional[str]:
    """Normalize common date formats (YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY) into YYYY-MM-DD."""
    if not val:
        return None
    s = str(val).strip()
    
    # Check YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", s)
    if m_iso:
        y, m, d = m_iso.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # Check DD/MM/YYYY or DD-MM-YYYY
    m_dmy = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", s)
    if m_dmy:
        d, m, y = m_dmy.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    return None


# ---------------------------------------------------------------------------
# 1. Field-Level Matching & Evaluation
# ---------------------------------------------------------------------------

def field_match(
    expected: Any,
    actual: Any,
    strict: bool = False,
    fuzzy_threshold: float = 0.85,
) -> bool:
    """
    Evaluates whether an extracted field value matches the ground truth.
    
    If strict=True:
      Applies exact normalized string, gender abbreviation, normalized date,
      and normalized numeric/currency equivalence only.
      
    If strict=False (fuzzy-tolerant):
      Applies the full evaluation chain including high-threshold Levenshtein
      similarity (>= 0.85) and multi-token Jaccard overlap (>= 0.65).
      Incomplete single-token fragments (e.g. 'Sharma' for 'Rahul Sharma')
      are rejected.
    """
    if expected is None or actual is None:
        return expected == actual

    exp_str = str(expected).strip()
    act_str = str(actual).strip()

    if not exp_str and not act_str:
        return True
    if not exp_str or not act_str:
        return False

    # 1. Exact normalized match
    norm_exp = normalize_text(exp_str)
    norm_act = normalize_text(act_str)
    if norm_exp == norm_act:
        return True

    # 2. Gender / common single-letter abbreviations
    gender_map = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
    }
    if norm_exp in gender_map and norm_act in gender_map:
        return gender_map[norm_exp] == gender_map[norm_act]

    # 3. Date comparison
    date_exp = normalize_date(exp_str)
    date_act = normalize_date(act_str)
    if date_exp and date_act:
        return date_exp == date_act

    # 4. Numeric / currency comparison
    num_exp = normalize_numeric(exp_str)
    num_act = normalize_numeric(act_str)
    if num_exp is not None and num_act is not None:
        return abs(num_exp - num_act) < 1e-3

    # If strict mode requested, stop here (no fuzzy / substring / token overlap)
    if strict:
        return False

    # -----------------------------------------------------------------------
    # Fuzzy-Tolerant Matching Chain (strict=False)
    # -----------------------------------------------------------------------

    # If both values are numeric representations, strict numeric check was authoritative
    if num_exp is not None and num_act is not None:
        return False

    # Clean punctuation
    clean_exp = re.sub(r"[^\w\s]", " ", norm_exp).strip()
    clean_act = re.sub(r"[^\w\s]", " ", norm_act).strip()
    clean_exp = re.sub(r"\s+", " ", clean_exp)
    clean_act = re.sub(r"\s+", " ", clean_act)

    if clean_exp == clean_act:
        return True

    # 5. High-threshold Fuzzy sequence matching (e.g. 'Chander Kochar' vs 'CHANDER KOCHHAR')
    ratio = SequenceMatcher(None, norm_exp, norm_act).ratio()
    if ratio >= fuzzy_threshold:
        return True

    # 6. Multi-token Jaccard check for multi-word entities (requires >= 2 tokens on both sides)
    exp_tokens = set(clean_exp.split())
    act_tokens = set(clean_act.split())
    if len(exp_tokens) >= 2 and len(act_tokens) >= 2:
        overlap = exp_tokens.intersection(act_tokens)
        jaccard = len(overlap) / len(exp_tokens.union(act_tokens))
        if jaccard >= 0.65:
            return True

    return False


def evaluate_fields(
    ground_truth: Dict[str, Any],
    result_dict: Dict[str, Any],
    fuzzy_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Evaluates result dictionary against ground truth dictionary across both
    strict (exact/numeric/date) and fuzzy-tolerant (full chain) tiers.
    
    Returns:
    {
        "strict_pct": float,
        "strict_matched_count": int,
        "fuzzy_pct": float,
        "fuzzy_matched_count": int,
        "pct": float,  # alias for fuzzy_pct (backward-compatibility)
        "total_fields": int,
        "field_results": Dict[str, bool],         # fuzzy-tolerant verdicts
        "strict_field_results": Dict[str, bool],  # strict verdicts
        "field_details": Dict[str, Dict[str, Any]],
    }
    """
    gt_non_empty = {k: str(v).strip() for k, v in ground_truth.items() if str(v).strip()}
    if not gt_non_empty:
        return {
            "strict_pct": 100.0,
            "strict_matched_count": 0,
            "fuzzy_pct": 100.0,
            "fuzzy_matched_count": 0,
            "pct": 100.0,
            "total_fields": 0,
            "field_results": {},
            "strict_field_results": {},
            "field_details": {},
        }

    strict_matches: Dict[str, bool] = {}
    fuzzy_matches: Dict[str, bool] = {}
    details: Dict[str, Dict[str, Any]] = {}
    strict_correct = 0
    fuzzy_correct = 0

    for field, expected in gt_non_empty.items():
        actual = result_dict.get(field, "")
        s_match = field_match(expected, actual, strict=True)
        f_match = field_match(expected, actual, strict=False, fuzzy_threshold=fuzzy_threshold)

        strict_matches[field] = s_match
        fuzzy_matches[field] = f_match

        if s_match:
            strict_correct += 1
        if f_match:
            fuzzy_correct += 1

        details[field] = {
            "expected": expected,
            "actual": actual,
            "strict_matched": s_match,
            "fuzzy_matched": f_match,
        }

    total = len(gt_non_empty)
    strict_pct = round((strict_correct / total) * 100, 1)
    fuzzy_pct = round((fuzzy_correct / total) * 100, 1)

    return {
        "strict_pct": strict_pct,
        "strict_matched_count": strict_correct,
        "fuzzy_pct": fuzzy_pct,
        "fuzzy_matched_count": fuzzy_correct,
        "pct": fuzzy_pct,
        "total_fields": total,
        "field_results": fuzzy_matches,
        "strict_field_results": strict_matches,
        "field_details": details,
    }


# ---------------------------------------------------------------------------
# 2. Graph Evidence Groundedness
# ---------------------------------------------------------------------------

def node_has_evidence(node: Union[Dict[str, Any], Any]) -> bool:
    """Inspects a GraphNode dict or object to determine if it has grounded evidence."""
    if isinstance(node, dict):
        props = node.get("properties", {}) or {}
        # Direct evidence field on properties or node
        if props.get("evidence") or node.get("evidence"):
            return True
        # Check observations list
        for obs in props.get("observations", []):
            if isinstance(obs, dict) and obs.get("evidence"):
                return True
        # Check merge_audit evidence
        for audit in props.get("merge_audit", []):
            if isinstance(audit, dict) and audit.get("evidence"):
                return True
        # Check source_pages
        if node.get("source_pages") or props.get("source_pages") or node.get("source_page"):
            # Has source page tracing
            return True
        return False
    else:
        # Object attribute access (Pydantic / Dataclass)
        if getattr(node, "evidence", None):
            return True
        props = getattr(node, "properties", {}) or {}
        if isinstance(props, dict) and (props.get("evidence") or props.get("observations")):
            return True
        if getattr(node, "source_pages", None):
            return True
        return False


def evidence_groundedness(nodes: List[Union[Dict[str, Any], Any]]) -> Dict[str, Any]:
    """
    Computes evidence groundedness score across graph nodes.
    Returns:
    {
        "grounded_pct": float,
        "grounded_count": int,
        "total_nodes": int,
        "unreferenced_node_ids": List[str],
    }
    """
    if not nodes:
        return {
            "grounded_pct": 100.0,
            "grounded_count": 0,
            "total_nodes": 0,
            "unreferenced_node_ids": [],
        }

    grounded_count = 0
    unreferenced: List[str] = []

    for node in nodes:
        node_id = node.get("id", "") if isinstance(node, dict) else getattr(node, "id", "")
        if node_has_evidence(node):
            grounded_count += 1
        else:
            unreferenced.append(str(node_id))

    grounded_pct = round((grounded_count / len(nodes)) * 100, 1)

    return {
        "grounded_pct": grounded_pct,
        "grounded_count": grounded_count,
        "total_nodes": len(nodes),
        "unreferenced_node_ids": unreferenced,
    }


# ---------------------------------------------------------------------------
# 3. CER & WER (Character & Word Error Rates)
# ---------------------------------------------------------------------------

def levenshtein_distance(seq1: Union[str, List[str]], seq2: Union[str, List[str]]) -> int:
    """Standard dynamic programming Levenshtein distance between two sequences."""
    len1, len2 = len(seq1), len(seq2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[len1][len2]


def calculate_cer_wer(
    reference: str,
    hypothesis: str,
    case_sensitive: bool = False,
) -> Dict[str, float]:
    """
    Computes Character Error Rate (CER) and Word Error Rate (WER).
    CER = Levenshtein(reference_chars, hypothesis_chars) / len(reference_chars)
    WER = Levenshtein(reference_words, hypothesis_words) / len(reference_words)
    """
    ref_clean = reference if case_sensitive else reference.lower()
    hyp_clean = hypothesis if case_sensitive else hypothesis.lower()

    # Character level
    ref_chars = list(ref_clean)
    hyp_chars = list(hyp_clean)
    char_dist = levenshtein_distance(ref_chars, hyp_chars)
    cer = char_dist / max(len(ref_chars), 1)

    # Word level
    ref_words = re.findall(r"\b\w+\b", ref_clean)
    hyp_words = re.findall(r"\b\w+\b", hyp_clean)
    word_dist = levenshtein_distance(ref_words, hyp_words)
    wer = word_dist / max(len(ref_words), 1)

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "cer_pct": round(cer * 100, 2),
        "wer_pct": round(wer * 100, 2),
        "char_distance": char_dist,
        "ref_char_count": len(ref_chars),
        "word_distance": word_dist,
        "ref_word_count": len(ref_words),
    }
