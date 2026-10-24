"""
File: aggregate.py
Purpose: Combine sampled-page observations into a document-level prior, and
         pick sample pages (spread across the doc, widened when they disagree).
Deps: stdlib
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from src.ai.layer1_probe.types import DocPrior, SampleObs


def pick_sample_pages(candidates: List[int], n: int) -> List[int]:
    """Evenly spread picks (first, middle, last, ...) from sorted candidate page numbers."""
    candidates = sorted(candidates)
    if n <= 0 or not candidates:
        return []
    if len(candidates) <= n:
        return candidates
    idx = {round(i * (len(candidates) - 1) / (n - 1)) for i in range(n)} if n > 1 else {0}
    return [candidates[i] for i in sorted(idx)]


def additional_sample_pages(candidates: List[int], already: List[int], extra: int) -> List[int]:
    """More spread picks, avoiding pages already sampled."""
    left = [c for c in sorted(candidates) if c not in set(already)]
    return pick_sample_pages(left, extra)


def samples_disagree(obs: List[SampleObs], conf_floor: float = 0.6) -> bool:
    good = [o for o in obs if o.script not in ("unknown",) and o.script_confidence >= conf_floor]
    return len({o.script for o in good}) > 1


def aggregate_doc(obs: List[SampleObs]) -> DocPrior:
    obs = [o for o in obs if o is not None]
    if not obs:
        return DocPrior()
    weight: Dict[str, float] = defaultdict(float)
    for o in obs:
        if o.script != "unknown":
            weight[o.script] += max(o.script_confidence, 0.05)
    if not weight:
        return DocPrior(samples=obs, sampled_pages=[o.page_number for o in obs])
    script = max(weight, key=weight.__getitem__)
    agreeing = [o for o in obs if o.script == script]
    share = weight[script] / sum(weight.values())
    conf = round(share * (sum(o.script_confidence for o in agreeing) / len(agreeing)), 3)
    srcs = sorted({o.source for o in obs})
    hw_vals = [o.handwriting_prob for o in obs if o.handwriting_prob is not None]
    hw = round(sum(hw_vals) / len(hw_vals), 3) if hw_vals else None
    return DocPrior(
        script=script,
        script_confidence=conf,
        mixed_scripts=any(o.mixed_scripts for o in obs),
        handwriting_prob=hw,
        heterogeneous=samples_disagree(obs),
        source="+".join(srcs) or "none",
        sampled_pages=[o.page_number for o in obs],
        samples=obs,
    )


def nearest_sample(page_number: int, samples: List[SampleObs]) -> Optional[SampleObs]:
    """Observation from the closest sampled page (ties -> earlier page)."""
    usable = [s for s in samples if s.script != "unknown"]
    if not usable:
        return None
    return min(usable, key=lambda s: (abs(s.page_number - page_number), s.page_number))
