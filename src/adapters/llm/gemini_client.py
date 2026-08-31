"""
File: gemini_client.py
Purpose: Optional Gemini-compatible implementation of `LLMClient`.

This adapter tries to use the `google.generativeai` client when available
and falls back to an OpenAI-compatible `instructor` wrapper if a proxy
or alternative endpoint is supplied via `settings.gemini_base_url`.

Owner: engineer-a@idp-pilot
Created: 2026-08-24
"""
import base64
from typing import Any

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:
    # Provide no-op fallbacks when tenacity isn't installed (demo mode).
    def retry(*_a, **_k):
        def _decorator(f):
            return f

        return _decorator

    def stop_after_attempt(_n):
        return None

    def wait_exponential(*_a, **_k):
        return None

from src.adapters.llm.base import LLMClient
from src.ai.schemas.page import PageClassification, VLMAnalysis
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CLASSIFY_PROMPT = """You are inspecting a scanned document page image.
Determine: the best processing route (digital, scanned, handwritten, or skip),
the percentage of the page that is handwritten (0.0-1.0), the visual noise level
(0.0-1.0), and any preprocessing needed (deskew, denoise, upscale).
Respond only with the structured fields requested."""


class GeminiClient(LLMClient):
    def __init__(self) -> None:
        # Model name used by native or proxy clients
        

        # Prefer the official Google Generative AI client when available.
        if genai is not None and settings.gemini_api_key:
            # Configure the deprecated `google.generativeai` package if present
            try:
                genai.configure(api_key=settings.gemini_api_key)
                # Prefer the GenerativeModel / ChatSession API when available.
                # Create the GenerativeModel lazily to avoid constructor-time
                # network or validation errors; initialize on first use.
                if hasattr(genai, "GenerativeModel"):
                    self._gen_model = None
                    self._mode = "google_native"
                else:
                    # Older package surface without GenerativeModel
                    self._gen_model = None
                    self._mode = "google_native_basic"
            except Exception:
                self._mode = "unavailable"
        else:
            # Fall back to an OpenAI-compatible proxy via `instructor` if available.
            try:
                import instructor  # type: ignore
                from openai import OpenAI  # type: ignore

                self._client = instructor.from_openai(
                    OpenAI(base_url=settings.gemini_base_url or settings.ollama_base_url, api_key=settings.gemini_api_key or ""),
                    mode=instructor.Mode.JSON,
                )
                self._mode = "instructor_proxy"
            except Exception:
                # Dependencies missing; mark unavailable and surface on use.
                self._mode = "unavailable"

        self._model = settings.vlm_model_name

    def analyze_page(self, image_bytes: bytes, page_profile_hint: dict) -> VLMAnalysis:
        """Request extraction-capable analysis from OpenAI-compatible Gemini proxies."""
        if self._mode != "instructor_proxy":
            return super().analyze_page(image_bytes, page_profile_hint)
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        result = self._client.chat.completions.create(
            model=self._model,
            response_model=VLMAnalysis,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "Analyze and attempt to extract this page. Return required capabilities "
                    "(ocr, handwriting, table_structure, layout, text_extraction), whether "
                    "direct extraction is reliable, confidence, extracted_markdown, and "
                    "exact_transcription_required for IDs, dates, amounts, and codes."
                ) + f"\nHints: {page_profile_hint}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ]}],
        )
        logger.info("vlm.analyze_page.success",
                    can_extract_directly=result.can_extract_directly,
                    confidence=result.confidence)
        return result

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    def classify_page(self, image_bytes: bytes, page_profile_hint: dict) -> PageClassification:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        if self._mode == "unavailable":
            raise RuntimeError(
                "GeminiClient unavailable: missing optional dependencies.\n"
                "Install 'google-generativeai' or the 'instructor' + 'openai' packages, or set llm_provider to 'ollama'."
            )
        try:
            if self._mode == "google_native":
                # Use GenerativeModel -> ChatSession when available.
                if getattr(self, "_gen_model", None) is None:
                    try:
                        self._gen_model = genai.GenerativeModel(self._model)
                    except Exception:
                        # Fall back to a more basic path if model construction fails
                        self._mode = "google_native_basic"
                        self._gen_model = None
                if self._gen_model is None:
                    # downgrade to basic handling below
                    pass
                else:
                    chat = self._gen_model.start_chat()
                prompt = _CLASSIFY_PROMPT + f"\nHints: {page_profile_hint}\nImage: data:image/png;base64,{b64_image}"
                response = chat.send_message(prompt)
                text = getattr(response, "text", "")
                try:
                    result = PageClassification.model_validate_json(text)
                    logger.info("vlm.classify_page.success", route=result.route, confidence=result.confidence)
                    return result
                except Exception:
                    # If the model didn't return structured JSON, fall back
                    # to a conservative classification.
                    logger.warning("vlm.classify_page.parse_failed", sample=text[:200])
                    return PageClassification(route="scanned", handwritten_pct=0.0, confidence=0.5)

            if self._mode == "google_native_basic":
                # Best-effort for very old surface: call ChatSession via helper API
                try:
                    model = genai.get_model(self._model) if hasattr(genai, "get_model") else None
                except Exception:
                    model = None
                if model is not None and hasattr(model, "start_chat"):
                    chat = model.start_chat()
                    prompt = _CLASSIFY_PROMPT + f"\nHints: {page_profile_hint}\nImage: data:image/png;base64,{b64_image}"
                    response = chat.send_message(prompt)
                    text = getattr(response, "text", "")
                    try:
                        result = PageClassification.model_validate_json(text)
                        logger.info("vlm.classify_page.success", route=result.route, confidence=result.confidence)
                        return result
                    except Exception:
                        logger.warning("vlm.classify_page.parse_failed", sample=text[:200])
                        return PageClassification(route="scanned", handwritten_pct=0.0, confidence=0.5)

            if self._mode == "instructor_proxy":
                # instructor/OpenAI-compatible proxy path
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

            # If we reach here the client wasn't configured (missing deps)
            if self._mode == "unavailable":
                raise RuntimeError(
                    "GeminiClient unavailable: missing 'instructor' or 'openai' packages. Install them or set 'llm_provider' to 'ollama'."
                )

        except Exception as exc:
            logger.error("vlm.classify_page.failed", error=str(exc), error_type=type(exc).__name__)
            raise

    def transcribe_handwriting(self, image_bytes: bytes) -> tuple[str, float]:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        if self._mode == "unavailable":
            raise RuntimeError(
                "GeminiClient unavailable: missing optional dependencies.\n"
                "Install 'google-generativeai' or the 'instructor' + 'openai' packages, or set llm_provider to 'ollama'."
            )
        try:
            if self._mode == "google_native":
                if getattr(self, "_gen_model", None) is None:
                    try:
                        self._gen_model = genai.GenerativeModel(self._model)
                    except Exception:
                        self._mode = "google_native_basic"
                        self._gen_model = None
                if self._gen_model is None:
                    # Let the basic path handle this below
                    pass
                else:
                    chat = self._gen_model.start_chat()
                    prompt = "Transcribe all handwritten and printed text on this page, preserving structure as markdown.\nImage: data:image/png;base64,%s" % b64_image
                    response = chat.send_message(prompt)
                text = getattr(response, "text", "")
                confidence = 0.6 if len(text.strip()) > 20 else 0.2
                return text, confidence

            if self._mode == "google_native_basic":
                # Attempt via older model API surface
                try:
                    model = genai.get_model(self._model) if hasattr(genai, "get_model") else None
                except Exception:
                    model = None
                if model is not None and hasattr(model, "start_chat"):
                    chat = model.start_chat()
                    prompt = "Transcribe all handwritten and printed text on this page, preserving structure as markdown.\nImage: data:image/png;base64,%s" % b64_image
                    response = chat.send_message(prompt)
                    text = getattr(response, "text", "")
                    confidence = 0.6 if len(text.strip()) > 20 else 0.2
                    return text, confidence

            if self._mode == "instructor_proxy":
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
            confidence = 0.6 if len(text.strip()) > 20 else 0.2
            return text, confidence

            if self._mode == "unavailable":
                raise RuntimeError(
                    "GeminiClient unavailable: missing 'instructor' or 'openai' packages. Install them or set 'llm_provider' to 'ollama'."
                )

        except Exception as exc:
            logger.error("vlm.transcribe_handwriting.failed", error=str(exc), error_type=type(exc).__name__)
            raise
