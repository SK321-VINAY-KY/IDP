"""
File: vlm_probe.py
Purpose: VLM fallback for the probe — ONE small Bedrock Converse call per sampled
         page (not per page of the document). Returns script / handwriting /
         table signals as strict JSON.
Deps: boto3 (injected client for tests)
NOTE: standalone on purpose — it does not depend on bedrock_client.py (not in the
      repomix). If you prefer, swap `_call` to use your existing client.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from src.ai.layer1_probe.types import SCRIPTS, SampleObs
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT = (
    "You are a document-script identifier. Look at this scanned page and answer with ONE JSON "
    "object and nothing else, keys exactly:\n"
    '{"script": <dominant writing system, one of: latin, devanagari, tamil, bengali, gujarati, '
    "gurmukhi, kannada, malayalam, odia, telugu, arabic, cyrillic, greek, thai, hebrew, chinese, "
    'japanese, korean, other>, "script_confidence": <0-1>, '
    '"mixed_scripts": <true if a second writing system covers more than ~20% of the text>, '
    '"handwriting_fraction": <0-1, fraction of text that is handwritten>, '
    '"has_table": <true if a table/grid of rows and columns is present>}\n'
    "Ignore logos, stamps and signatures when judging script."
)

_ALIASES = {
    "english": "latin", "roman": "latin", "hindi": "devanagari", "marathi": "devanagari",
    "sanskrit": "devanagari", "nepali": "devanagari", "punjabi": "gurmukhi",
    "oriya": "odia", "han": "chinese", "hanzi": "chinese", "kana": "japanese", "hangul": "korean",
    "russian": "cyrillic",
}


def _norm_script(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = _ALIASES.get(s, s)
    return s if s in SCRIPTS else ("other" if s else "unknown")


def _clip01(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def parse_vlm_json(text: str) -> Optional[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class BedrockScriptProbe:
    def __init__(self, model_id: str, region: str = "ap-south-1", client: Any = None):
        self.model_id = model_id
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=60),
            )
        self.client = client

    @staticmethod
    def _image_format(data: bytes) -> str:
        """Detect the Converse image format from the leading magic bytes.

        PNG (\\x89PNG) -> 'png'; JPEG (\\xff\\xd8\\xff) -> 'jpeg'; anything else or
        ambiguous -> 'jpeg' (preserves the Lambda path, which feeds real JPEG).
        Raw bytes are passed through unchanged (no decode/re-encode).
        """
        if data[:4] == b"\x89PNG":
            return "png"
        if data[:3] == b"\xff\xd8\xff":
            return "jpeg"
        return "jpeg"

    def _call(self, jpeg: bytes) -> str:
        fmt = self._image_format(jpeg)
        resp = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [
                {"image": {"format": fmt, "source": {"bytes": jpeg}}},
                {"text": _PROMPT},
            ]}],
            inferenceConfig={"maxTokens": 300, "temperature": 0.0},
        )
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

    def observe(self, page_number: int, jpeg: bytes) -> Optional[SampleObs]:
        try:
            data = parse_vlm_json(self._call(jpeg))
        except Exception as exc:                       # network/throttle/model errors
            logger.warning("probe.vlm_failed", page_number=page_number, error=str(exc))
            return None
        if not data:
            logger.warning("probe.vlm_unparseable", page_number=page_number)
            return None
        return SampleObs(
            page_number=page_number,
            script=_norm_script(data.get("script")),
            script_confidence=_clip01(data.get("script_confidence"), 0.5),
            mixed_scripts=bool(data.get("mixed_scripts", False)),
            handwriting_prob=_clip01(data.get("handwriting_fraction"), 0.0),
            has_table=bool(data.get("has_table", False)),
            source="vlm",
        )
