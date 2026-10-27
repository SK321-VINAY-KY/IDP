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


def _rgb_from_image(img: Any) -> np.ndarray:
    """Decode an in-memory page image to an RGB uint8 array.

    Accepts an RGB numpy array (returned as-is) or encoded bytes (PNG/JPEG),
    decoded with cv2.imdecode + BGR->RGB in the _gray_mask_from_jpeg style.
    """
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, (bytes, bytearray, memoryview)):
        bgr = cv2.imdecode(np.frombuffer(bytes(img), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode page image bytes")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise TypeError(f"unsupported page image type: {type(img).__name__}")


# ---------------------------------------------------------------------------
# Shared Phase-B/C core (used by BOTH probe_document and probe_document_images)
# ---------------------------------------------------------------------------

def _make_observe(cfg: ProbeConfig, vlm: Any, onnx: Any, jpegs: Dict[int, bytes],
                  stats: Dict[str, Any]) -> Callable[[int], Optional[SampleObs]]:
    """Build the per-sample observer closure. Holds the lipika-vs-vlm telemetry
    and the VLM-overrides-confident-latin seam so both entry points behave the
    same. `jpegs[pn]` must hold the encoded page image for sample page `pn`.
    """

    def _observe(pn: int) -> Optional[SampleObs]:
        obs = None
        jpeg = jpegs[pn]
        if onnx is not None:
            gray, mask = _gray_mask_from_jpeg(jpeg, cfg)
            obs = onnx.observe(pn, gray, mask)
            stats["onnx_calls"] += 1

        # Capture lipika's guess BEFORE any VLM call (kept only for logging).
        lip = obs
        lip_script = lip.script if lip is not None else None
        lip_conf = lip.script_confidence if lip is not None else None
        # n_windows / margin are non-routing telemetry exposed by the line-wise
        # observer; read defensively so stub/VLM-only tests still work.
        lip_margin = getattr(onnx, "last_margin", None) if onnx is not None else None
        n_windows = getattr(onnx, "last_n_windows", None) if onnx is not None else None
        if lip is not None:
            stats["lipika_resolved"] += 1

        vlm_called = False
        v = None
        if (obs is None or obs.script_confidence < cfg.onnx_min_conf or obs.mixed_scripts) \
                and vlm is not None and cfg.vlm_enabled:
            v = vlm.observe(pn, jpeg)
            vlm_called = True
            stats["vlm_calls"] += 1
            if v is not None:
                obs = v
        elif (cfg.vlm_for_handwriting and obs is not None and obs.script == "latin"
              and obs.script_confidence >= cfg.onnx_min_conf and not obs.mixed_scripts
              and vlm is not None and cfg.vlm_enabled):
            # Confident-latin pages normally SKIP the VLM; when the handwriting flag is
            # on, make ONE extra call and adopt ONLY its handwriting_prob (keep ONNX
            # script/confidence) so latin handwritten pages can route correctly.
            v = vlm.observe(pn, jpeg)
            vlm_called = True
            stats["vlm_calls"] += 1
            if v is not None:
                # CRITICAL SEAM: the VLM is already called for every confident-latin
                # page via the handwriting flag. If it CONTRADICTS lipika's "latin"
                # (returns a different script), the VLM wins for routing so a latin
                # guess the VLM disputes does NOT route to Paddle. An agreeing/latin
                # VLM keeps lipika's script and the handwriting-only behavior.
                if v.script and v.script != obs.script:
                    obs = v
                else:
                    obs.handwriting_prob = v.handwriting_prob

        # Telemetry (log only; no image bytes). `agree` = the VLM was called AND
        # returned the same script as lipika; None when the VLM was not called.
        vlm_script = v.script if (vlm_called and v is not None) else None
        agree: Optional[bool] = (vlm_script == lip_script) if vlm_called else None
        if vlm_called and lip_script is not None:
            if agree:
                stats["lipika_vlm_agree"] += 1
            else:
                stats["lipika_vlm_disagree"] += 1
        logger.info("probe.lipika_vs_vlm", page_number=pn, lipika_script=lip_script,
                    lipika_conf=lip_conf, margin=lip_margin, n_windows=n_windows,
                    vlm_script=vlm_script, agree=agree)

        if obs is None:
            obs = SampleObs(page_number=pn, source="none")
        return obs

    return _observe


def _sample_and_aggregate(nonblank: List[int], jpegs: Dict[int, bytes], cfg: ProbeConfig,
                          vlm: Any, onnx: Any, stats: Dict[str, Any]) -> DocPrior:
    """Phase B (minus any S3 cache): sample ~n_samples pages, widen if they
    disagree, run the observer, and aggregate into a document-level prior.
    """
    if not nonblank:
        return DocPrior()
    _observe = _make_observe(cfg, vlm, onnx, jpegs, stats)

    def _workers(n: int) -> int:
        # Each sample observer may issue one VLM (Bedrock) call; run them in
        # parallel. cfg.vlm_concurrency caps the pool; <=0 keeps the prior
        # "one thread per page in this wave" behaviour.
        if cfg.vlm_concurrency > 0:
            return max(1, min(n, cfg.vlm_concurrency))
        return max(1, n)

    wave = pick_sample_pages(nonblank, cfg.n_samples)
    with ThreadPoolExecutor(max_workers=_workers(len(wave))) as ex:
        obs = list(ex.map(_observe, wave))
    if samples_disagree(obs) and len(nonblank) > len(wave) and cfg.max_samples > len(wave):
        more = additional_sample_pages(nonblank, wave, cfg.max_samples - len(wave))
        if more:
            with ThreadPoolExecutor(max_workers=_workers(len(more))) as ex:
                obs += list(ex.map(_observe, more))
    return aggregate_doc(obs)


def _decide_pages(candidates: List[int], nonblank: List[int], timed_out: List[int],
                  feats: Dict[int, dict], prior: DocPrior, cfg: ProbeConfig,
                  given: Optional[Dict[int, dict]] = None,
                  image_keys: Optional[Dict[int, Optional[str]]] = None,
                  img_bucket: Optional[str] = None) -> Dict[int, PageProbe]:
    """Phase C: nearest-sample inheritance + decide_route for non-blank pages,
    blank pages -> skip, timed-out pages -> prior-only routing. Returns the
    per-page PageProbe map (image_s3_* populated only when image_keys given).
    """
    given = given or {}
    image_keys = image_keys or {}
    results: Dict[int, PageProbe] = {}
    sampled = set(prior.sampled_pages)

    def _page_values(pn: int):
        samp = nearest_sample(pn, prior.samples)
        if prior.heterogeneous and samp is not None:
            hw = samp.handwriting_prob if samp.handwriting_prob is not None else prior.handwriting_prob
            return (samp.script, samp.script_confidence,
                    samp.mixed_scripts, samp.source, samp, hw)
        hw = prior.handwriting_prob
        if hw is None and samp is not None:
            hw = samp.handwriting_prob
        return (prior.script, prior.script_confidence,
                prior.mixed_scripts, prior.source, samp, hw)

    for pn in nonblank:
        f = feats[pn]
        script, sconf, mixed, hsrc, samp, hw = _page_values(pn)
        has_table = bool(f["table"])
        if pn in sampled and samp is not None and samp.page_number == pn and samp.has_table:
            has_table = True
        choice = decide_route(script=script, script_conf=sconf, mixed=mixed,
                              is_blank=False, has_table=has_table, cfg=cfg,
                              handwriting_prob=hw)
        results[pn] = PageProbe(
            page_number=pn, probed=True, status="ok", route=choice.route, ocr_lang=choice.ocr_lang,
            use_structure=choice.use_structure, route_reason=choice.reason,
            script=script, script_confidence=sconf,
            script_source="sample" if pn in sampled else "prior",
            mixed_scripts=mixed, handwriting_prob=hw, has_table=has_table, table_confidence=f["tconf"],
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
        script, sconf, mixed, hsrc, _, hw = _page_values(pn)
        choice = decide_route(script=script, script_conf=sconf, mixed=mixed,
                              is_blank=False, has_table=False, cfg=cfg,
                              handwriting_prob=hw)
        results[pn] = PageProbe(page_number=pn, probed=False, status="skipped_timeout",
                                route=choice.route, ocr_lang=choice.ocr_lang,
                                route_reason=f"timeout_prior:{choice.reason}",
                                script=script, script_confidence=sconf, handwriting_prob=hw,
                                script_source="prior")
    return results


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
    stats = {"candidates": len(candidates), "vlm_calls": 0, "onnx_calls": 0, "cache_hit": False,
             "lipika_resolved": 0, "lipika_vlm_agree": 0, "lipika_vlm_disagree": 0}
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

    if nonblank and not stats["cache_hit"]:
        prior = _sample_and_aggregate(nonblank, jpegs, cfg, vlm, onnx, stats)
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
    results.update(_decide_pages(candidates, nonblank, timed_out, feats, prior, cfg,
                                 given=given, image_keys=image_keys, img_bucket=img_bucket))

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


def probe_document_images(
    images: Dict[int, Any], page_numbers: List[int], *,
    profiles: Optional[Dict[int, Any]] = None, cfg: ProbeConfig,
    vlm: Any = None, onnx: Any = None,
    time_left: Optional[Callable[[], float]] = None,
) -> Dict[int, PageProbe]:
    """In-memory probe entry point for the live pipeline. No fitz, no S3, no cache.

    images:       page_number -> RGB numpy array OR encoded bytes (PNG/JPEG).
    page_numbers: pages to probe (caller already filtered with needs_probe).
    profiles:     optional per-page metadata (unused for routing; passthrough).

    Computes Phase-A CV signals in memory, encodes a per-page JPEG only to feed
    the observer (image_s3_* stay absent), then runs the SAME Phase-B/C helpers
    as probe_document. Returns {page_number -> PageProbe}.
    """
    t0 = time.monotonic()
    _ = profiles  # optional metadata only; routing never reads it
    candidates = [pn for pn in page_numbers if pn in images]

    jpegs: Dict[int, bytes] = {}
    feats: Dict[int, dict] = {}
    results: Dict[int, PageProbe] = {}
    timed_out: List[int] = []
    for pn in candidates:
        if time_left is not None and time_left() < cfg.reserve_s:
            timed_out.append(pn)
            continue
        try:
            rgb = _rgb_from_image(images[pn])
            gray = imaging.analysis_gray(rgb, cfg.page_image_dpi, cfg.analysis_dpi)
            mask = imaging.ink_mask(gray)
            ink = cv_features.ink_ratio(mask)
            blank = ink < cfg.blank_ink_ratio
            table, tconf = (False, 0.0) if blank else cv_features.detect_table(
                mask, cfg.table_min_h_lines, cfg.table_min_v_lines, cfg.table_min_intersections)
            w, h = imaging.image_size(rgb)
            jpegs[pn] = imaging.encode_jpeg(rgb, cfg.jpeg_quality)
            feats[pn] = dict(ink=ink, blank=blank, table=table, tconf=tconf, w=w, h=h)
        except Exception as exc:                       # one bad page must not kill the document
            logger.error("probe.page_failed", page_number=pn, error=str(exc))
            results[pn] = PageProbe(page_number=pn, probed=False, status="error",
                                    route=ROUTE_PRINTED,
                                    route_reason=f"probe_error:{type(exc).__name__}")

    stats = {"candidates": len(candidates), "vlm_calls": 0, "onnx_calls": 0, "cache_hit": False,
             "lipika_resolved": 0, "lipika_vlm_agree": 0, "lipika_vlm_disagree": 0}
    nonblank = [pn for pn in candidates if pn in feats and not feats[pn]["blank"]]
    prior = _sample_and_aggregate(nonblank, jpegs, cfg, vlm, onnx, stats)

    results.update(_decide_pages(candidates, nonblank, timed_out, feats, prior, cfg))

    stats.update(elapsed_s=round(time.monotonic() - t0, 2), probed=len(feats),
                 timed_out=len(timed_out), total_pages=len(candidates))
    logger.info("probe.complete_images", doc_script=prior.script,
                doc_script_conf=prior.script_confidence, heterogeneous=prior.heterogeneous, **stats)
    return results
