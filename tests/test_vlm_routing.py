import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adapters.llm.base import LLMClient
from src.ai.layer1_routing import pipeline
from src.ai.schemas.page import PageClassification, PageProfile, VLMAnalysis


class FakeLLM(LLMClient):
    def __init__(self, analysis=None, transcribe_result=("vlm fallback text", 0.92)):
        self.analysis = analysis
        self.analysis_calls = 0
        self.transcribe_result = transcribe_result
        self.transcribe_calls = 0

    def analyze_page(self, image_bytes, page_profile_hint):
        self.analysis_calls += 1
        return self.analysis

    def classify_page(self, image_bytes, page_profile_hint):
        raise AssertionError("legacy classifier should not be called")

    def transcribe_handwriting(self, image_bytes):
        self.transcribe_calls += 1
        return self.transcribe_result


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


def test_paddle_is_routed_first_and_vlm_not_called_when_confident(monkeypatch):
    class NoVlmLLM(FakeLLM):
        def transcribe_handwriting(self, image_bytes):
            raise AssertionError("VLM must not be called when PaddleOCR succeeds with confidence >= 0.75")

    client = NoVlmLLM()
    calls = []

    def fake_engine(task, context, llm):
        calls.append(task.engine)
        return "paddle text", 0.90, 1.0

    monkeypatch.setattr(pipeline, "_run_engine_task", fake_engine)
    output, metadata = run_page(monkeypatch, client)

    assert output.markdown == "paddle text"
    assert output.confidence == 0.90
    assert output.engines_used == ["paddleocr_printed"]
    assert calls == ["paddleocr_printed"]
    assert output.escalated is False
    assert client.transcribe_calls == 0


def test_escalation_to_vlm_when_paddle_has_low_confidence(monkeypatch):
    client = FakeLLM(transcribe_result=("vlm clean transcription", 0.95))
    calls = []

    def fake_engine(task, context, llm):
        calls.append(task.engine)
        if task.engine == "paddleocr_printed":
            return "low conf paddle text", 0.60, 1.0
        elif task.engine == "vlm_transcribe":
            text, conf = llm.transcribe_handwriting(context["image_bytes"])
            return text, conf, 1.0
        return "", 0.0, 1.0

    monkeypatch.setattr(pipeline, "_run_engine_task", fake_engine)
    output, metadata = run_page(monkeypatch, client)

    assert output.markdown == "vlm clean transcription"
    assert output.confidence == 0.95
    assert "paddleocr_printed" in calls
    assert "vlm_transcribe" in calls
    assert "vlm_transcribe" in output.engines_used
    assert output.escalated is True
    assert client.transcribe_calls == 1


def test_escalation_to_vlm_when_paddle_fails(monkeypatch):
    client = FakeLLM(transcribe_result=("vlm fallback after failure", 0.92))
    calls = []

    def fake_engine(task, context, llm):
        calls.append(task.engine)
        if task.engine == "paddleocr_printed":
            return "", 0.0, 1.0  # Paddle failed
        elif task.engine == "vlm_transcribe":
            text, conf = llm.transcribe_handwriting(context["image_bytes"])
            return text, conf, 1.0
        return "", 0.0, 1.0

    monkeypatch.setattr(pipeline, "_run_engine_task", fake_engine)
    output, metadata = run_page(monkeypatch, client)

    assert output.markdown == "vlm fallback after failure"
    assert output.confidence == 0.92
    assert "vlm_transcribe" in output.engines_used
    assert output.escalated is True
    assert client.transcribe_calls == 1


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
