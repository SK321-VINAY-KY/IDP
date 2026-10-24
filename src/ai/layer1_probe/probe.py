"""
File: probe.py
Purpose: Layer 1.5 probe orchestrator. Runs between Layer 1 (PyMuPDF profile) and the
         Step Functions page Map.

  Phase A  per candidate page : render ONCE -> CV signals (blank / ruled table) ->
           page JPEG uploaded to S3 for the engines.
  Phase B  per DOCUMENT       : sample 2-3 pages (more only if they disagree) ->
           ONNX script observer (optional) -> VLM fallback -> doc-level script prior.
           Result cached by S3 ETag.
  Phase C  per page           : script from own sample or nearest sample,
           table/blank from own CV -> routing table -> engine + ocr_lang.

Pages that don't need probing (digital text, L1-blank) pass through unchanged.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import cv2
import fitz
import numpy as np

from src.ai.layer1_probe import cv_features, imaging
from src.ai.layer1_probe.aggregate import (
    additional_sample_pages, aggregate_doc, nearest_sample, pick_sample_pages, samples_disagree,
)
from src.ai.layer1_probe.routing import decide_route
from src.ai.layer1_probe.types import (
    DocPrior, PageProbe, ProbeConfig, SampleObs,
    ROUTE_DOCLING, ROUTE_PRINTED, ROUTE_SKIP,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe(doc_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)


def needs_probe(profile: Optional[dict], given_route: Optional[str], cfg: ProbeConfig) -> bool:
    if cfg.force_probe_all:
        return given_route != ROUTE_SKIP
    if given_route in (ROUTE_SKIP, ROUTE_DOCLING):
        return False
    if given_route is not None:           # printed / handwritten / vlm from L1 -> verify script
        return True
    p = profile or {}
    return bool(
        p.get("is_scanned")
        or not p.get("has_text", False)
        or p.get("char_count", 0) <= cfg.digital_char_threshold
        or p.get("primary_script", "unknown") == "unknown"
    )


def _gray_mask_from_jpeg(jpeg: bytes, cfg: ProbeConfig) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = imaging.analysis_gray(rgb, cfg.page_image_dpi, cfg.analysis_dpi)
    return gray, imaging.ink_mask(gray)


def probe_document(
    pdf_bytes: bytes, *, document_id: str, s3_bucket: Optional[str], s3_key: Optional[str],
    pages_in: Optional[List[dict]], cfg: ProbeConfig, s3: Any = None, vlm: Any = None,
    onnx: Any = None, etag: Optional[str] = None,
    time_left: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)

    given: Dict[int, dict] = {}
    for p in (pages_in or []):
        if isinstance(p, int):
            p = {"page_number": p}
        given[int(p["page_number"])] = p
    page_numbers = sorted(given) if given else list(range(1, total + 1))
    page_numbers = [n for n in page_numbers if 1 <= n <= total]

    # ---- candidate selection ------------------------------------------------
    from src.ai.layer1_routing.inspect import inspect_page
    results: Dict[int, PageProbe] = {}
    candidates: List[int] = []
    for pn in page_numbers:
        g = given.get(pn, {})
        profile = g.get("profile")
        if profile is None and g.get("route") is None:
            profile = inspect_page(doc[pn - 1], pn).model_dump()
        if needs_probe(profile, g.get("route"), cfg):
            candidates.append(pn)
        else:
            results[pn] = PageProbe(page_number=pn, probed=False, status="not_needed",
                                    route=g.get("route") or ROUTE_DOCLING,
                                    route_reason="passthrough_digital_or_blank",
                                    script=(profile or {}).get("primary_script", "unknown"),
                                    script_source="embedded")

    # ---- Phase A: render once, CV signals, async upload ---------------------
    img_bucket = cfg.image_bucket or s3_bucket
    do_upload = bool(cfg.write_page_images and s3 is not None and img_bucket)
    jpegs: Dict[int, bytes] = {}
    feats: Dict[int, dict] = {}
    futures: Dict[int, Any] = {}
    pool = ThreadPoolExecutor(max_workers=cfg.upload_workers)

    def _upload(pn: int, data: bytes) -> str:
        key = f"{cfg.image_prefix}/{_safe(document_id)}/{pn}.jpg"
        s3.put_object(Bucket=img_bucket, Key=key, Body=data, ContentType="image/jpeg")
        return key

    timed_out: List[int] = []
    for pn in candidates:
        if time_left is not None and time_left() < cfg.reserve_s:
            timed_out.append(pn)
            continue
        try:
            rgb = imaging.render_rgb(doc[pn - 1], cfg.page_image_dpi)
            gray = imaging.analysis_gray(rgb, cfg.page_image_dpi, cfg.analysis_dpi)
            mask = imaging.ink_mask(gray)
            ink = cv_features.ink_ratio(mask)
            blank = ink < cfg.blank_ink_ratio
            table, tconf = (False, 0.0) if blank else cv_features.detect_table(
                mask, cfg.table_min_h_lines, cfg.table_min_v_lines, cfg.table_min_intersections)
            w, h = imaging.image_size(rgb)
            jpeg = imaging.encode_jpeg(rgb, cfg.jpeg_quality)
            jpegs[pn] = jpeg
            feats[pn] = dict(ink=ink, blank=blank, table=table, tconf=tconf, w=w, h=h)
            if do_upload and not blank:
                futures[pn] = pool.submit(_upload, pn, jpeg)
        except Exception as exc:                       # one bad page must not kill the document
            logger.error("probe.page_failed", page_number=pn, error=str(exc))
            results[pn] = PageProbe(page_number=pn, probed=False, status="error",
                                    route=given.get(pn, {}).get("route") or ROUTE_PRINTED,
                                    route_reason=f"probe_error:{type(exc).__name__}")
    doc.close()

    # ---- Phase B: document-level script prior --------------------------------
    stats = {"candidates": len(candidates), "vlm_calls": 0, "onnx_calls": 0, "cache_hit": False}
    nonblank = [pn for pn in candidates if pn in feats and not feats[pn]["blank"]]
    prior = DocPrior()
    cache_key = f"{cfg.cache_prefix}/{cfg.cache_version}/{_safe(etag)}.json" if etag else None

    if nonblank and cfg.cache_enabled and s3 is not None and img_bucket and cache_key:
        try:
            body = s3.get_object(Bucket=img_bucket, Key=cache_key)["Body"].read()
            prior = DocPrior.model_validate_json(body)
            stats["cache_hit"] = True
        except Exception:
            pass

    def _observe(pn: int) -> Optional[SampleObs]:
        obs = None
        jpeg = jpegs[pn]
        if onnx is not None:
            gray, mask = _gray_mask_from_jpeg(jpeg, cfg)
            obs = onnx.observe(pn, gray, mask)
            stats["onnx_calls"] += 1
        if (obs is None or obs.script_confidence < cfg.onnx_min_conf or obs.mixed_scripts) \
                and vlm is not None and cfg.vlm_enabled:
            v = vlm.observe(pn, jpeg)
            stats["vlm_calls"] += 1
            if v is not None:
                obs = v
        if obs is None:
            obs = SampleObs(page_number=pn, source="none")
        return obs

    if nonblank and not stats["cache_hit"]:
        wave = pick_sample_pages(nonblank, cfg.n_samples)
        with ThreadPoolExecutor(max_workers=max(1, len(wave))) as ex:
            obs = list(ex.map(_observe, wave))
        if samples_disagree(obs) and len(nonblank) > len(wave) and cfg.max_samples > len(wave):
            more = additional_sample_pages(nonblank, wave, cfg.max_samples - len(wave))
            if more:
                with ThreadPoolExecutor(max_workers=len(more)) as ex:
                    obs += list(ex.map(_observe, more))
        prior = aggregate_doc(obs)
        if (cfg.cache_enabled and s3 is not None and img_bucket and cache_key
                and prior.script != "unknown" and prior.script_confidence >= cfg.min_script_conf):
            try:
                s3.put_object(Bucket=img_bucket, Key=cache_key, ContentType="application/json",
                              Body=prior.model_dump_json().encode())
            except Exception as exc:
                logger.warning("probe.cache_write_failed", error=str(exc))

    # ---- uploads finished? --------------------------------------------------
    image_keys: Dict[int, Optional[str]] = {}
    for pn, fut in futures.items():
        try:
            image_keys[pn] = fut.result()
        except Exception as exc:
            logger.warning("probe.upload_failed", page_number=pn, error=str(exc))
            image_keys[pn] = None
    pool.shutdown(wait=True)

    # ---- Phase C: per-page decision -----------------------------------------
    sampled = set(prior.sampled_pages)

    def _page_values(pn: int):
        samp = nearest_sample(pn, prior.samples)
        if prior.heterogeneous and samp is not None:
            return (samp.script, samp.script_confidence,
                    samp.mixed_scripts, samp.source, samp)
        return (prior.script, prior.script_confidence,
                prior.mixed_scripts, prior.source, samp)

    for pn in nonblank:
        f = feats[pn]
        script, sconf, mixed, hsrc, samp = _page_values(pn)
        has_table = bool(f["table"])
        if pn in sampled and samp is not None and samp.page_number == pn and samp.has_table:
            has_table = True
        choice = decide_route(script=script, script_conf=sconf, mixed=mixed,
                              is_blank=False, has_table=has_table, cfg=cfg)
        results[pn] = PageProbe(
            page_number=pn, probed=True, status="ok", route=choice.route, ocr_lang=choice.ocr_lang,
            use_structure=choice.use_structure, route_reason=choice.reason,
            script=script, script_confidence=sconf,
            script_source="sample" if pn in sampled else "prior",
            mixed_scripts=mixed, has_table=has_table, table_confidence=f["tconf"],
            is_blank=False, ink_ratio=round(f["ink"], 5),
            image_s3_bucket=img_bucket if image_keys.get(pn) else None,
            image_s3_key=image_keys.get(pn), image_width=f["w"], image_height=f["h"],
        )

    for pn in candidates:
        if pn in feats and feats[pn]["blank"]:
            results[pn] = PageProbe(page_number=pn, probed=True, status="blank", route=ROUTE_SKIP,
                                    route_reason="blank_page", is_blank=True,
                                    ink_ratio=round(feats[pn]["ink"], 5))

    for pn in timed_out:                                  # budget ran out: prior-only routing
        script, sconf, mixed, hsrc, _ = _page_values(pn)
        choice = decide_route(script=script, script_conf=sconf, mixed=mixed,
                              is_blank=False, has_table=False, cfg=cfg)
        results[pn] = PageProbe(page_number=pn, probed=False, status="skipped_timeout",
                                route=choice.route, ocr_lang=choice.ocr_lang,
                                route_reason=f"timeout_prior:{choice.reason}",
                                script=script, script_confidence=sconf, script_source="prior")

    pages_out = [results[pn].model_dump() for pn in sorted(results)]
    stats.update(elapsed_s=round(time.monotonic() - t0, 2), probed=len(feats),
                 timed_out=len(timed_out), total_pages=total)
    logger.info("probe.complete", document_id=document_id, doc_script=prior.script,
                doc_script_conf=prior.script_confidence, heterogeneous=prior.heterogeneous, **stats)

    return {
        "document_id": document_id,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "doc_prior": prior.model_dump(exclude={"samples"}),
        "pages": pages_out,
        "stats": stats,
    }
