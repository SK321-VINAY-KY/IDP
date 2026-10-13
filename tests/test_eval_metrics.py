"""
File: test_eval_metrics.py
Purpose: Unit tests for scripts/eval_metrics.py covering field_match,
         evaluate_fields, evidence_groundedness, and CER/WER calculations.
"""
import pytest
from scripts.eval_metrics import (
    field_match,
    evaluate_fields,
    evidence_groundedness,
    calculate_cer_wer,
    normalize_date,
    normalize_numeric,
    normalize_text,
)


class TestFieldMatching:
    def test_exact_match(self):
        assert field_match("Rahul Sharma", "Rahul Sharma")
        assert field_match("Rahul Sharma", "rahul sharma")
        assert field_match("  Rahul Sharma  ", "Rahul Sharma")

    def test_numeric_currency_normalization(self):
        assert field_match("10,000.00", "10000")
        assert field_match("10000.00", "10,000.00")
        assert field_match("Rs-93553/-", "93553")
        assert field_match("INR 87,500", "87500.00")
        assert field_match("$450.00", "450")
        assert not field_match("10000", "20000")

    def test_date_normalization(self):
        assert field_match("2025-02-24", "24/02/2025")
        assert field_match("2025-02-24", "24-02-2025")
        assert field_match("12/08/2026", "2026-08-12")

    def test_fuzzy_and_substring_match(self):
        assert field_match("CHANDER KOCHHAR", "Chander Kochar")
        assert field_match(
            "P. D. HINDUJA HOSPITAL & MEDICAL RESEARCH CENTRE",
            "P. D. Hinduja Hospital & Medical Research Centre"
        )
        assert not field_match("Female", "Male")

    def test_original_bug_still_fails(self):
        """Single surname token 'Sharma' must NOT match full name 'Rahul Sharma'."""
        assert field_match("Rahul Sharma", "Sharma") is False
        assert field_match("Sharma", "Rahul Sharma") is False
        assert field_match("Rahul Sharma", "Sharma", strict=True) is False
        assert field_match("Rahul Sharma", "Sharma", strict=False) is False


class TestEvaluateFields:
    def test_evaluate_fields_contract(self):
        gt = {
            "patient_name": "CHANDER KOCHHAR",
            "admission_date": "2025-02-24",
            "advance_deposit_amount": "10,000.00",
            "gender": "Female",
        }
        res = {
            "patient_name": "Chander Kochar",  # matches under fuzzy-tolerant (typo), fails strict
            "admission_date": "24/02/2025",    # matches strict (date normalization)
            "advance_deposit_amount": "10000.00", # matches strict (numeric normalization)
            "gender": "Female",                # matches strict (exact)
        }
        eval_res = evaluate_fields(gt, res)
        assert eval_res["fuzzy_pct"] == 100.0
        assert eval_res["strict_pct"] == 75.0  # 3/4 match strictly
        assert eval_res["strict_matched_count"] == 3
        assert eval_res["fuzzy_matched_count"] == 4
        assert eval_res["total_fields"] == 4
        assert eval_res["strict_field_results"]["patient_name"] is False
        assert eval_res["field_results"]["patient_name"] is True

    def test_evaluate_fields_partial(self):
        gt = {
            "patient_name": "Rahul Sharma",
            "age": "45",
        }
        res = {
            "patient_name": "Rahul Sharma",
            "age": "91",
        }
        eval_res = evaluate_fields(gt, res)
        assert eval_res["strict_pct"] == 50.0
        assert eval_res["fuzzy_pct"] == 50.0
        assert eval_res["strict_matched_count"] == 1
        assert eval_res["fuzzy_matched_count"] == 1
        assert eval_res["field_results"]["patient_name"] is True
        assert eval_res["field_results"]["age"] is False


class TestEvidenceGroundedness:
    def test_groundedness_calculation(self):
        nodes = [
            {
                "id": "node_1",
                "label": "Patient",
                "properties": {
                    "evidence": [{"text": "Patient Name: Rahul", "confidence": 0.99}],
                },
            },
            {
                "id": "node_2",
                "label": "Hospital",
                "source_pages": [1],
            },
            {
                "id": "node_3",
                "label": "OrphanNode",
                "properties": {},
            },
        ]
        res = evidence_groundedness(nodes)
        assert res["total_nodes"] == 3
        assert res["grounded_count"] == 2
        assert res["grounded_pct"] == 66.7
        assert "node_3" in res["unreferenced_node_ids"]


class TestCerWer:
    def test_perfect_match(self):
        res = calculate_cer_wer("Hello World", "Hello World")
        assert res["cer"] == 0.0
        assert res["wer"] == 0.0

    def test_errors(self):
        res = calculate_cer_wer("The quick brown fox", "The quik brown dog")
        assert res["cer"] > 0.0
        assert res["wer"] > 0.0
        assert res["char_distance"] == 3  # 'c' deleted (1) + 'fox' -> 'dog' ('f'->'d', 'x'->'g' = 2)
