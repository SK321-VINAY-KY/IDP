import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adapters.llm.base import LLMClient
from src.ai.layer1_routing import pipeline
from src.ai.schemas.page import PageClassification, PageProfile, VLMAnalysis


class FakeLLM(LLMClient):
    def __init__(self, analysis):
        self.analysis = analysis
        self.analysis_calls = 0

    def analyze_page(self, image_bytes, page_profile_hint):
        self.analysis_calls += 1
        return self.analysis

    def classify_page(self, image_bytes, page_profile_hint):
        raise AssertionError("legacy classifier should not be called")

    def transcribe_handwriting(self, image_bytes):
        raise AssertionError("VLM transcription fallback should not be called")


def scanned_profile():
    return PageProfile(
        page_number=1, has_text=False, char_count=0, image_coverage=0.8,
        has_tables=False, is_scanned=True, has_vector_drawings=False,
        primary_script="latin", complexity_score=0, dpi_estimate=200,
    )


def run_page(monkeypatch, client, **kwargs):
    monkeypatch.setattr(pipeline, "inspect_page", lambda page, page_number: scanned_profile())
    monkeypatch.setattr(pipeline.settings, "routing_mode", "single_engine")
    return pipeline.process_page(
        page=object(), page_number=1,
        page_context={"pdf_path": "page.pdf", "image_array": object()},
        llm_client=client, page_image_bytes=b"image", **kwargs,
    )


def test_high_confidence_vlm_direct_is_terminal(monkeypatch):
    client = FakeLLM(VLMAnalysis(
        can_extract_directly=True, confidence=0.95,
        detected_capabilities={"ocr"}, required_capabilities={"ocr"},
        extracted_markdown="printed text",
    ))
    monkeypatch.setattr(pipeline, "_run_engine_task", lambda *args: (_ for _ in ()).throw(
        AssertionError("specialized engine must not run")))

    output, metadata = run_page(monkeypatch, client)

    assert output.markdown == "printed text"
    assert output.engines_used == ["vlm_direct"]
    assert metadata.engine_plan == ["vlm_direct"]
    assert client.analysis_calls == 1


def test_low_confidence_vlm_uses_only_required_ocr(monkeypatch):
    client = FakeLLM(VLMAnalysis(
        can_extract_directly=True, confidence=0.60,
        detected_capabilities={"ocr"}, required_capabilities={"ocr"},
        extracted_markdown="should not be accepted",
    ))
    calls = []

    def fake_engine(task, context, llm):
        calls.append(task.engine)
        return "ocr text", 0.90, 1.0

    monkeypatch.setattr(pipeline, "_run_engine_task", fake_engine)
    output, _ = run_page(monkeypatch, client)

    assert output.markdown == "ocr text"
    assert calls == ["paddleocr_printed"]


def test_exact_transcription_requirement_rejects_direct_vlm(monkeypatch):
    client = FakeLLM(VLMAnalysis(
        can_extract_directly=True, confidence=0.99,
        detected_capabilities={"ocr"}, required_capabilities={"ocr"},
        extracted_markdown="semantic policy number",
    ))
    calls = []
    monkeypatch.setattr(pipeline, "_run_engine_task", lambda task, context, llm: (
        calls.append(task.engine) or ("exact policy number", 0.95, 1.0)
    ))

    output, _ = run_page(
        monkeypatch, client, extraction_requirements={"exact_transcription": True}
    )

    assert output.markdown == "exact policy number"
    assert calls == ["paddleocr_printed"]


def test_vlm_exact_flag_rejects_direct_vlm(monkeypatch):
    client = FakeLLM(VLMAnalysis(
        can_extract_directly=True, confidence=0.99,
        detected_capabilities={"ocr"}, required_capabilities={"ocr"},
        extracted_markdown="semantic claim number",
        exact_transcription_required=True,
    ))
    calls = []
    monkeypatch.setattr(pipeline, "_run_engine_task", lambda task, context, llm: (
        calls.append(task.engine) or ("exact claim number", 0.95, 1.0)
    ))

    output, _ = run_page(monkeypatch, client)

    assert output.markdown == "exact claim number"
    assert calls == ["paddleocr_printed"]
