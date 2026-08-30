"""
File: ollama_client.py
Purpose: Local Ollama implementation of LLMClient (Qwen2-VL-2B), OpenAI-compatible
         endpoint via Instructor. No API keys, fully offline.
Owner: engineer-a@idp-pilot
Created: 2026-08-19 | Deps: instructor, openai, tenacity
"""
import base64

import instructor
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CLASSIFY_PROMPT = """You are inspecting a scanned document page image.
Determine: the best processing route (digital, scanned, handwritten, or skip),
the percentage of the page that is handwritten (0.0-1.0), the visual noise level
(0.0-1.0), and any preprocessing needed (deskew, denoise, upscale).
Respond only with the structured fields requested."""


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        self._client = instructor.from_openai(
            OpenAI(base_url=settings.ollama_base_url, api_key="ollama"),  # Ollama ignores the key
            mode=instructor.Mode.JSON,
        )
        self._model = settings.vlm_model_name

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    def classify_page(self, image_bytes: bytes, page_profile_hint: dict) -> PageClassification:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        try:
            result = self._client.chat.completions.create(
                model=self._model,
                response_model=PageClassification,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _CLASSIFY_PROMPT + f"\nHints: {page_profile_hint}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                        ],
                    }
                ],
            )
            logger.info("vlm.classify_page.success", route=result.route, confidence=result.confidence)
            return result
        except Exception as exc:
            logger.error("vlm.classify_page.failed", error=str(exc), error_type=type(exc).__name__)
            raise

    def transcribe_handwriting(self, image_bytes: bytes) -> tuple[str, float]:
        # Not used when settings.handwriting_engine == "trocr" (the pilot default).
        # Kept implemented so classify-only ambiguous cases can still fall back
        # to VLM transcription without a code change if trocr quality is insufficient.
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self._model,
            response_model=None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe all handwritten and printed text on this page, preserving structure as markdown."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or ""
        # Ollama vision models don't return a native confidence score; approximate
        # from response length/structure as a conservative placeholder.
        confidence = 0.6 if len(text.strip()) > 20 else 0.2
        return text, confidence
