"""
End-to-End Integration Test for Layer 3 Navigation-First Pipeline.
Uses the navigation POC's Section 11 experiment table against the Chander Kochhar 76-page bundle.

Acceptance Criteria:
1. Segmentation isolates ~5 real sub-documents (+/- 1 page).
2. Navigation for Identifier and Amount recovers matches from both page 1 and page 6 (not just first hit).
3. Page 4's OCR-garbled pharmacy bill still segments correctly as [4, 4].
4. "Dr. Abhay Raut" and "Dr Arhay Raut" resolve to one Person node.
5. Actual LLM call counts get logged against the current exhaustive-scan baseline per the cost model in Section 9.
"""
import json
from pathlib import Path
import pytest

from src.ai.layer3_extraction.page_loader import load_pages_from_fixture
from src.ai.layer3_extraction.pipeline import run_navigation_extraction_pipeline
from src.config.ontology import load_ontology


class MockPipelineLLM:
    """Mock LLM delivering realistic navigation and summarization for the test bundle."""
    def __init__(self):
        self.summary_calls = 0
        self.nav_calls = 0
        self.extraction_calls = 0
        self.key_findings_calls = 0

    def summarize_segment(self, segment_text: str, page_range: list[int]) -> dict:
        self.summary_calls += 1
        p_start, p_end = page_range
        text_upper = segment_text[:1000].upper()
        if "DISCHARGE SUMMARY" in text_upper:
            return {"doc_type_hint": "discharge summary", "one_line_summary": "Discharge summary for Chander Kochhar from Hinduja Hospital"}
        elif "CHEMIST" in text_upper:
            return {"doc_type_hint": "pharmacy bill", "one_line_summary": "Hayaat Chemist pharmacy bill for patient with prescribed medicines"}
        elif "AMBULANCE" in text_upper:
            return {"doc_type_hint": "ambulance bill", "one_line_summary": "Jeevan Ambulance transport bill for patient"}
        elif "DEPOSIT RECEIPT" in text_upper:
            return {"doc_type_hint": "deposit receipt", "one_line_summary": "In-patient deposit receipt for Rs 10000"}
        elif "AUTHORIZATION" in text_upper:
            return {"doc_type_hint": "authorization letter", "one_line_summary": "ICICI Lombard cashless authorization guarantee letter"}
        elif "FINAL BILL" in text_upper:
            return {"doc_type_hint": "final bill", "one_line_summary": "Hospital final inpatient bill"}
        elif "INVOICE" in text_upper:
            return {"doc_type_hint": "invoice", "one_line_summary": "Hospital tax invoice"}
        return {"doc_type_hint": "document", "one_line_summary": f"Document segment for pages {p_start}-{p_end}"}

    def navigate_category(self, segment_summaries, category, category_description, category_examples=None, extraction_focus=None):
        self.nav_calls += 1
        # Navigation logic per POC Section 6 & Experiment B:
        # Narrows extraction to key sub-document segments rather than scanning all 76 pages
        if category == "Amount":
            # Pages 4, 5, 6, 7, 10-12
            return ["seg_02", "seg_03", "seg_04", "seg_05", "seg_08"]
        elif category == "Identifier":
            # Recovers both discharge summary (seg_01, page 1) and deposit receipt (seg_04, page 6)
            return ["seg_01", "seg_02", "seg_03", "seg_04", "seg_05", "seg_08"]
        elif category in ("Person", "Organization"):
            # Recovers discharge summary, pharmacy, ambulance, deposit receipt, authorization letter
            return ["seg_01", "seg_02", "seg_03", "seg_04", "seg_05"]
        elif category in ("Date", "Document/Record"):
            return ["seg_01", "seg_02", "seg_03", "seg_04", "seg_05"]
        else:
            return ["seg_01"]

    def extract(self, content, schema):
        return schema()

    def summarize_page(self, page_md, max_words=50):
        return ""

    def navigate(self, page_summaries, schema_fields):
        return {}

    def check_page_for_fields(self, page_md, schema_fields, page_number=0, total_pages=0):
        return []

    def extract_page_ontology(self, page_md, page_number=0, extraction_focus=None):
        self.extraction_calls += 1
        if page_number == 1:
            return {
                "nodes": [
                    {"category": "Person", "label": "Chander Kochhar", "importance": "high", "importance_reason": "Primary hospitalized patient", "subtype": "Patient"},
                    {"category": "Person", "label": "Dr. Abhay Raut", "importance": "high", "importance_reason": "Attending consultant surgeon", "subtype": "Doctor"},
                    {"category": "Organization", "label": "P. D. Hinduja Hospital", "importance": "high", "importance_reason": "Issuing medical hospital", "subtype": "Hospital"},
                    {"category": "Identifier", "label": "HS No: 192673", "importance": "high", "importance_reason": "Hospital identification number", "subtype": "HS No"},
                    {"category": "Identifier", "label": "Adm No: 68049", "importance": "medium", "importance_reason": "Inpatient admission record", "subtype": "Admission No"},
                    {"category": "Document/Record", "label": "Discharge Summary", "importance": "medium", "importance_reason": "Clinical discharge summary", "subtype": "Discharge Summary"},
                ],
                "edges": [
                    {"source_label": "Chander Kochhar", "target_label": "P. D. Hinduja Hospital", "relationship": "ADMITTED_TO"},
                    {"source_label": "Dr. Abhay Raut", "target_label": "Chander Kochhar", "relationship": "TREATED"},
                ]
            }
        elif page_number == 4:
            return {
                "nodes": [
                    {"category": "Person", "label": "Chander Kochhar", "importance": "high", "importance_reason": "Patient purchasing medicines", "subtype": "Patient"},
                    {"category": "Person", "label": "Dr Arhay Raut", "importance": "high", "importance_reason": "Doctor prescribing medicines", "subtype": "Doctor"},
                    {"category": "Organization", "label": "Hayaat Chemist", "importance": "medium", "importance_reason": "Dispensing retail chemist", "subtype": "Pharmacy"},
                    {"category": "Amount", "label": "INR 1485.75", "importance": "high", "importance_reason": "Total pharmacy bill charge", "subtype": "Total Amount"},
                    {"category": "Identifier", "label": "Phone: 9892462685", "importance": "low", "importance_reason": "Chemist contact phone", "subtype": "Phone"},
                ],
                "edges": [
                    {"source_label": "Dr Arhay Raut", "target_label": "Chander Kochhar", "relationship": "PRESCRIBED_FOR"},
                ]
            }
        elif page_number == 5:
            return {
                "nodes": [
                    {"category": "Person", "label": "Chander Kochhar", "importance": "high", "importance_reason": "Patient transported in ambulance", "subtype": "Patient"},
                    {"category": "Organization", "label": "Jeevan Ambulance Service", "importance": "high", "importance_reason": "Emergency transport company", "subtype": "Ambulance Provider"},
                    {"category": "Amount", "label": "INR 4500", "importance": "high", "importance_reason": "Ambulance transit fee", "subtype": "Total Amount"},
                    {"category": "Identifier", "label": "Bill No: 20570", "importance": "medium", "importance_reason": "Ambulance receipt voucher", "subtype": "Bill No"},
                    {"category": "Identifier", "label": "Phone: 98924625", "importance": "low", "importance_reason": "Emergency dispatch phone", "subtype": "Phone"},
                ],
                "edges": []
            }
        elif page_number == 6:
            return {
                "nodes": [
                    {"category": "Person", "label": "Chander Kochhar", "importance": "high", "importance_reason": "Patient for hospital deposit", "subtype": "Patient"},
                    {"category": "Organization", "label": "P. D. Hinduja Hospital", "importance": "high", "importance_reason": "Hospital receipt issuer", "subtype": "Hospital"},
                    {"category": "Amount", "label": "INR 10000", "importance": "high", "importance_reason": "Hospital in-patient deposit amount", "subtype": "Deposit Amount"},
                    {"category": "Identifier", "label": "HS No: 192673", "importance": "high", "importance_reason": "Patient hospital record identifier", "subtype": "HS No"},
                    {"category": "Document/Record", "label": "Deposit Receipt", "importance": "medium", "importance_reason": "Proof of deposit receipt", "subtype": "Deposit Receipt"},
                ],
                "edges": []
            }
        elif page_number == 7:
            return {
                "nodes": [
                    {"category": "Person", "label": "Chander Kochhar", "importance": "high", "importance_reason": "Insured patient covered", "subtype": "Patient"},
                    {"category": "Organization", "label": "ICICI Lombard", "importance": "high", "importance_reason": "Insurer issuing authorization", "subtype": "Insurer"},
                    {"category": "Amount", "label": "INR 80698", "importance": "high", "importance_reason": "Cashless approved guarantee amount", "subtype": "Authorized Amount"},
                    {"category": "Identifier", "label": "AL Number: 110100834860", "importance": "high", "importance_reason": "Insurance pre-authorization number", "subtype": "AL Number"},
                    {"category": "Document/Record", "label": "Authorization Letter", "importance": "medium", "importance_reason": "Cashless pre-auth guarantee letter", "subtype": "Authorization Letter"},
                ],
                "edges": []
            }
        return {"nodes": [], "edges": []}

    def extract_key_findings(self, segments, candidate_nodes, extraction_focus=None):
        self.key_findings_calls += 1
        return {
            "key_findings": [
                {
                    "label": "Primary Patient",
                    "value": "Chander Kochhar",
                    "importance": "high",
                    "reason": "Hospitalized patient subject of all medical bills",
                    "source_pages": [1, 2, 4, 6],
                    "confidence": 0.98,
                },
                {
                    "label": "Primary Hospital",
                    "value": "P. D. Hinduja Hospital",
                    "importance": "high",
                    "reason": "Issuing medical center for inpatient admission",
                    "source_pages": [1, 6],
                    "confidence": 0.98,
                }
            ]
        }


def test_full_navigation_pipeline_chander_kochhar():
    pages = load_pages_from_fixture("chander_kochhar")
    assert len(pages) == 76

    # Load OCR metadata
    schema_ref_path = Path("tests/fixtures/chander_kochhar.schema_ref.json")
    if schema_ref_path.exists():
        with open(schema_ref_path, encoding="utf-8") as f:
            ref = json.load(f)
        ref_map = {p["page_number"]: p for p in ref.get("pages", [])}
        for p in pages:
            meta = ref_map.get(p["page_number"], {})
            p["confidence"] = meta.get("confidence", 1.0)
            p["engines_used"] = meta.get("engines_used", [])
            p["chars"] = meta.get("chars", len(p["markdown"]))

    mock_llm = MockPipelineLLM()
    ontology = load_ontology()

    result = run_navigation_extraction_pipeline(
        pages=pages,
        llm=mock_llm,
        ontology=ontology,
    )

    # -------------------------------------------------------------
    # 1. Acceptance Criterion 1 & 3:
    # Segmentation isolates ~5 real sub-documents (+/- 1 page)
    # Page 4's OCR-garbled pharmacy bill still segments correctly as [4, 4]
    # -------------------------------------------------------------
    spans = [s.page_range for s in result.segments]
    assert spans[0] == [1, 3], f"Expected discharge summary [1, 3], got {spans[0]}"
    assert spans[1] == [4, 4], f"Expected page 4 pharmacy bill isolated as [4, 4], got {spans[1]}"
    assert spans[2] == [5, 5], f"Expected ambulance bill [5, 5], got {spans[2]}"
    assert spans[3] == [6, 6], f"Expected deposit receipt [6, 6], got {spans[3]}"
    assert spans[4] == [7, 7], f"Expected authorization letter [7, 7], got {spans[4]}"

    # Verify doc_type_hints
    assert result.segments[0].doc_type_hint == "discharge summary"
    assert result.segments[1].doc_type_hint == "pharmacy bill"
    assert result.segments[2].doc_type_hint == "ambulance bill"
    assert result.segments[3].doc_type_hint == "deposit receipt"
    assert result.segments[4].doc_type_hint == "authorization letter"

    # -------------------------------------------------------------
    # 2. Acceptance Criterion 2:
    # Navigation for Identifier/Amount recovers matches from both page 1 and page 6
    # (does NOT stop after the first hit)
    # -------------------------------------------------------------
    id_pages = result.navigation_map.category_pages.get("Identifier", [])
    amt_pages = result.navigation_map.category_pages.get("Amount", [])

    assert 1 in id_pages, "Identifier category must include page 1"
    assert 6 in id_pages, "Identifier category must include page 6 (recovers both, doesn't stop)"
    assert 6 in amt_pages, "Amount category must include page 6 (deposit receipt)"
    assert 7 in amt_pages, "Amount category must include page 7 (authorization amount)"

    # -------------------------------------------------------------
    # 3. Acceptance Criterion 4:
    # "Dr. Abhay Raut" and "Dr Arhay Raut" resolve to one Person node
    # -------------------------------------------------------------
    doctor_entities = [
        e for e in result.resolved_entities
        if "ABHAY" in e.canonical_label.upper() or "ARHAY" in e.canonical_label.upper()
    ]
    assert len(doctor_entities) == 1, f"Expected 1 resolved doctor entity, got {len(doctor_entities)}"
    resolved_doctor = doctor_entities[0]
    assert 1 in resolved_doctor.source_pages
    assert 4 in resolved_doctor.source_pages


    # Also check patient "Chander Kochhar" resolved across pages 1, 4, 5, 6
    patient_entities = [
        e for e in result.resolved_entities
        if "CHANDER" in e.canonical_label.upper() or "CHAMDAR" in e.canonical_label.upper()
    ]
    assert len(patient_entities) == 1
    assert 1 in patient_entities[0].source_pages
    assert 4 in patient_entities[0].source_pages

    # -------------------------------------------------------------
    # 4. Acceptance Criterion 5:
    # Actual LLM call counts logged against exhaustive-scan baseline
    # Cost model per Section 9: S + 9 + N vs P + 1
    # -------------------------------------------------------------
    cost = result.cost_summary
    assert cost.total_pages_P == 76
    assert cost.baseline_exhaustive_calls == 77  # P + 1
    assert cost.navigation_llm_calls == 9        # Exactly 9 calls (1 per ontology category)
    assert cost.segmentation_llm_calls == len(result.segments)
    assert cost.actual_llm_calls_total == (
        cost.segmentation_llm_calls + cost.navigation_llm_calls + cost.extraction_llm_calls + cost.expectation_check_rerun_calls + cost.key_findings_llm_calls
    )
    # The navigation pipeline achieves significant call reduction
    assert cost.reduction_percentage > 0.0

    # Verify mock call counts match CostSummary
    assert mock_llm.summary_calls == cost.segmentation_llm_calls
    assert mock_llm.nav_calls == cost.navigation_llm_calls
    assert mock_llm.key_findings_calls == cost.key_findings_llm_calls
    assert mock_llm.extraction_calls >= cost.extraction_llm_calls

    # Verify candidate nodes have populated importance, importance_reason, and subtype
    assert any(n.importance == "high" for n in result.candidate_nodes)
    assert any(n.importance_reason != "" for n in result.candidate_nodes)
    assert any(n.subtype == "Doctor" for n in result.candidate_nodes)

    # -------------------------------------------------------------
    # 5. Key Findings Check
    # -------------------------------------------------------------
    assert len(result.key_findings) > 0
    patient_finding = next((kf for kf in result.key_findings if "Patient" in kf.label), None)
    assert patient_finding is not None
    assert "Kochhar" in patient_finding.value
    assert patient_finding.importance == "high"

    print("\n" + "=" * 60)
    print("COST MODEL REPORT (Section 9):")
    print(f"Total document pages (P)        : {cost.total_pages_P}")
    print(f"Total sub-document segments (S) : {cost.total_segments_S}")
    print(f"Segmentation LLM calls (S)      : {cost.segmentation_llm_calls}")
    print(f"Navigation LLM calls (9 fixed)  : {cost.navigation_llm_calls}")
    print(f"Navigated pages extracted (N)   : {cost.navigated_pages_N}")
    print(f"Extraction LLM calls (N)        : {cost.extraction_llm_calls}")
    print(f"Expectation check re-run calls  : {cost.expectation_check_rerun_calls}")
    print(f"Key findings LLM calls          : {cost.key_findings_llm_calls}")
    print(f"ACTUAL LLM CALLS TOTAL (S+9+N+1): {cost.actual_llm_calls_total}")
    print(f"BASELINE EXHAUSTIVE CALLS (P+1) : {cost.baseline_exhaustive_calls}")
    print(f"CALL REDUCTION                  : {cost.reduction_percentage}%")
    print("=" * 60)


def test_pipeline_with_extraction_focus():
    """Verify that extraction_focus is accepted end-to-end without errors and yields results."""
    pages = load_pages_from_fixture("chander_kochhar")[:7]
    mock_llm = MockPipelineLLM()

    result = run_navigation_extraction_pipeline(
        pages=pages,
        llm=mock_llm,
        extraction_focus=["Focus on pharmacy bills", "Check for Dr. Abhay Raut prescriptions"],
    )
    assert len(result.segments) > 0
    assert len(result.candidate_nodes) > 0
    assert len(result.key_findings) > 0
    assert mock_llm.extraction_calls > 0
    assert mock_llm.key_findings_calls == 1


def test_type_other_ignored_by_identity_resolution_and_matching():
    """Verify identity resolution and identifier matching ignore type='Other' nodes without error."""
    from src.ai.layer3_extraction.extractor import CandidateNode
    from src.ai.layer3_extraction.identity_resolution import run_scoped_identity_resolution
    from src.ai.layer3_extraction.identifier_matching import run_identifier_matching

    nodes = [
        CandidateNode(
            node_id="n_001",
            type="Person",
            label="Dr. Abhay Raut",
            source_page=1,
            evidence="Doctor: Dr. Abhay Raut",
            importance="high",
        ),
        CandidateNode(
            node_id="n_002",
            type="Person",
            label="Dr. Arhay Raut",
            source_page=4,
            evidence="Doctor: Dr. Arhay Raut",
            importance="high",
        ),
        CandidateNode(
            node_id="n_003",
            type="Other",
            label="Acute Bronchitis",
            source_page=1,
            evidence="Diagnosis: Acute Bronchitis",
            importance="high",
            subtype="Diagnosis",
        ),
        CandidateNode(
            node_id="n_004",
            type="Other",
            label="Acute Bronchitis with wheeze",
            source_page=2,
            evidence="Diagnosis: Acute Bronchitis with wheeze",
            importance="high",
            subtype="Diagnosis",
        ),
    ]

    # Scoped identity resolution only resolves Person and Organization
    resolved_entities, edges, disambig_log = run_scoped_identity_resolution(nodes=nodes, edges=[])
    categories = {e.category for e in resolved_entities}
    assert "Other" not in categories
    assert "Person" in categories
    # Ensure type="Other" nodes were completely bypassed by disambiguation and clustering
    assert len(disambig_log) == 1, "Only Person nodes evaluated against each other; Other nodes never evaluated"
    assert disambig_log[0].label_a == "Dr. Abhay Raut"
    assert disambig_log[0].label_b == "Dr. Arhay Raut"
    assert not any(e.source in ("n_003", "n_004") or e.target in ("n_003", "n_004") for e in edges)

    # Scoped identifier matching only resolves Identifier
    resolved_ids, edges2, match_log = run_identifier_matching(nodes=nodes, edges=edges)
    assert len(resolved_ids) == 0  # No Identifier nodes present
    assert len(match_log) == 0  # Zero comparisons attempted for non-Identifier nodes


def test_target_pages_bypass_phase_a_and_b():
    """
    Pre-flight unit test for the target_pages parameter:
    Asserts Phase A (segmentation) and Phase B (navigation) are completely bypassed (0 LLM calls)
    and Step 5 (expectation check) gracefully no-ops (passes open with 0 reruns).
    """
    mock_llm = MockPipelineLLM()
    # Provide 4 pages of markdown
    pages = [
        {"page_number": 1, "markdown": "P. D. HINDUJA HOSPITAL DISCHARGE SUMMARY Hs No: 192673 MRS CHANDER KOCHHAR"},
        {"page_number": 2, "markdown": "Clinical details..."},
        {"page_number": 3, "markdown": "Lab results..."},
        {"page_number": 4, "markdown": "HAYAAT CHEMIST Bill No: 456 AOTAL ANT: 1485.75"},
    ]

    result = run_navigation_extraction_pipeline(
        pages=pages,
        llm=mock_llm,
        target_pages=[1, 4],
    )

    # 1. Assert Phase A & Phase B LLM calls are zero
    assert result.cost_summary.segmentation_llm_calls == 0, "Phase A segmentation was not bypassed!"
    assert result.cost_summary.navigation_llm_calls == 0, "Phase B navigation was not bypassed!"
    assert mock_llm.summary_calls == 0, "Mock LLM received segmentation calls despite bypass!"
    assert mock_llm.nav_calls == 0, "Mock LLM received navigation calls despite bypass!"

    # 2. Assert navigated pages matches target_pages
    assert result.cost_summary.navigated_pages_N == 2
    assert result.cost_summary.extraction_llm_calls == 2

    # 3. Assert Step 5 (expectation check) gracefully no-ops without reruns or errors
    assert len(result.expectation_check_results) == 2
    assert all(r.passed for r in result.expectation_check_results), "Step 5 failed unexpectedly on synthetic stubs!"
    assert not any(r.rerun_triggered for r in result.expectation_check_results), "Step 5 triggered reruns unexpectedly!"
    assert result.cost_summary.expectation_check_rerun_calls == 0


