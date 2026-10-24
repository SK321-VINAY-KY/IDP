"""
File: scripts/export_lipika_script_onnx.py
Purpose: DEVELOPER-ONLY, OFFLINE export of the lipika (loopdesk-ai/lipika) script
         head to ONNX for the Layer 1.5 probe's whole-image script observer.

This script is NOT imported at runtime and NOT run by CI. torch / fontrecog /
transformers / onnx / onnxruntime.quantization are HEAVY, OFFLINE-ONLY deps and
are imported INSIDE main() so merely importing this module never imports torch.

It loads loopdesk-ai/lipika (a FontNet: ConvNeXtV2 backbone + family/script/weight/
italic heads), isolates the backbone + SCRIPT head only into a thin nn.Module whose
forward(pixel_values) -> script_logits, exports FP32 ONNX (opset 17), dynamic batch
axis, then dynamic-INT8-quantizes. It validates torch vs FP32 ONNX vs INT8 ONNX on
the four docs/ images and writes three artifacts to src/ai/layer1_probe/models/.

Preprocessing is Option A (single letterbox forward on the whole image) reproducing
lipika's own `letterbox`: pad to square with white (255), BICUBIC resize to 224x224,
scale to 0..1. No ImageNet mean/std is applied by lipika, so mean=[0,0,0], std=[1,1,1].

Run:  .venv\Scripts\python.exe scripts/export_lipika_script_onnx.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

# Module-relative output dir (src/ai/layer1_probe/models). Resolve WITHOUT
# importing any heavy dep at module scope.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_MODELS_DIR = os.path.join(_REPO, "src", "ai", "layer1_probe", "models")

# lipika script index -> probe vocabulary name (design section 4).
# Perso-Arabic -> arabic; Meetei Mayek / Ol Chiki / Syloti Nagri -> other
# (not in the probe SCRIPTS vocab); everything else is a direct lowercase match.
_LIPIKA_TO_PROBE = {
    "bengali": "bengali",
    "devanagari": "devanagari",
    "gujarati": "gujarati",
    "gurmukhi": "gurmukhi",
    "kannada": "kannada",
    "latin": "latin",
    "malayalam": "malayalam",
    "meetei_mayek": "other",
    "odia": "odia",
    "ol_chiki": "other",
    "perso_arabic": "arabic",
    "syloti_nagri": "other",
    "tamil": "tamil",
    "telugu": "telugu",
}

# (filename, expected probe-vocab label) -- validation set under docs/.
_VALIDATION = [
    ("Screenshot 2026-10-23 140614.png", "latin"),
    ("Screenshot 2026-10-23 140632.png", "latin"),
    ("Screenshot 2026-10-23 142509.png", "telugu"),
    ("Screenshot 2026-10-23 142632.png", "devanagari"),
]

_INPUT_SIZE = 224


def main() -> int:
    # --- heavy imports, OFFLINE ONLY, kept inside main() on purpose ---
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from fontrecog.infer.predictor import Predictor, letterbox

    print("[export] loading loopdesk-ai/lipika ...")
    try:
        predictor = Predictor.from_pretrained("loopdesk-ai/lipika")
    except Exception as e:  # install/import/download failure -> STOP-AND-ASK
        print(f"[export] FATAL: could not load lipika: {e}", file=sys.stderr)
        return 2

    model = predictor.model.eval()
    scripts = list(predictor.space.scripts)  # lipika's raw 14 script names, index order
    probe_labels = [_LIPIKA_TO_PROBE.get(s, "other") for s in scripts]
    print(f"[export] lipika scripts ({len(scripts)}): {scripts}")
    print(f"[export] probe labels           : {probe_labels}")

    # --- thin wrapper: backbone -> drop -> script head; returns script logits ---
    class ScriptHead(nn.Module):
        def __init__(self, fontnet: nn.Module):
            super().__init__()
            self.backbone = fontnet.backbone
            self.drop = fontnet.drop
            self.script = fontnet.script

        def forward(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
            feats = self.backbone(pixel_values)
            return self.script(self.drop(feats))

    head = ScriptHead(model).eval()

    def _letterbox_batch(img: "Image.Image") -> "torch.Tensor":
        # single-forward (Option A): one letterboxed whole image, batch of 1
        return letterbox(img, size=_INPUT_SIZE).unsqueeze(0)

    # sanity: wrapper logits must match the full model's script logits
    with torch.no_grad():
        probe_img = Image.new("RGB", (256, 96), (255, 255, 255))
        x = _letterbox_batch(probe_img)
        full = model(x)["script"]
        wrapped = head(x)
        if not torch.allclose(full, wrapped, atol=1e-4):
            print("[export] FATAL: wrapper logits diverge from FontNet.script",
                  file=sys.stderr)
            return 3

    os.makedirs(_MODELS_DIR, exist_ok=True)
    fp32_path = os.path.join(_MODELS_DIR, "lipika_script.onnx")
    int8_path = os.path.join(_MODELS_DIR, "lipika_script.int8.onnx")
    labels_path = os.path.join(_MODELS_DIR, "lipika_script.labels.json")

    # --- export FP32 ONNX (opset 17, dynamic batch) ---
    print(f"[export] exporting FP32 ONNX -> {fp32_path}")
    dummy = torch.zeros(1, 3, _INPUT_SIZE, _INPUT_SIZE, dtype=torch.float32)
    try:
        # dynamo=False uses the stable TorchScript exporter (no onnxscript, no
        # unicode-logging crash on a cp1252 console); opset 17 per design.
        torch.onnx.export(
            head, dummy, fp32_path,
            input_names=["pixel_values"], output_names=["script_logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "script_logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except Exception as e:
        print(f"[export] FATAL: torch.onnx.export failed: {e}", file=sys.stderr)
        return 4

    # --- dynamic INT8 quantization (saved regardless of validation) ---
    print(f"[export] quantizing INT8 ONNX -> {int8_path}")
    int8_ok = True
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    except Exception as e:
        int8_ok = False
        print(f"[export] WARNING: INT8 quantization failed: {e}", file=sys.stderr)

    # --- validation: torch vs FP32 ONNX vs INT8 ONNX on docs/ images ---
    import onnxruntime as ort

    def _softmax_np(a: "np.ndarray") -> "np.ndarray":
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _onnx_predict(sess: "ort.InferenceSession", img: "Image.Image"):
        iname = sess.get_inputs()[0].name
        x = _letterbox_batch(img).numpy().astype(np.float32)
        logits = np.asarray(sess.run(None, {iname: x})[0], dtype=np.float32)
        probs = _softmax_np(logits[0])
        idx = int(probs.argmax())
        return probe_labels[idx], float(probs[idx])

    def _torch_predict(img: "Image.Image"):
        with torch.no_grad():
            x = _letterbox_batch(img)
            probs = torch.softmax(head(x)[0], dim=0)
            idx = int(probs.argmax())
            return probe_labels[idx], float(probs[idx])

    fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    int8_sess = (ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
                 if int8_ok and os.path.exists(int8_path) else None)

    docs_dir = os.path.join(_REPO, "docs")
    print("\n[export] validation (image -> expect | torch | fp32 | int8):")
    validated = 0
    int8_preserves = int8_sess is not None
    for fname, expect in _VALIDATION:
        path = os.path.join(docs_dir, fname)
        if not os.path.exists(path):
            print(f"  SKIP (missing): {fname}")
            continue
        validated += 1
        img = Image.open(path).convert("RGB")
        t_lab, t_c = _torch_predict(img)
        f_lab, f_c = _onnx_predict(fp32_sess, img)
        if int8_sess is not None:
            i_lab, i_c = _onnx_predict(int8_sess, img)
            i_str = f"{i_lab}({i_c:.3f})"
            if i_lab != t_lab:
                int8_preserves = False
        else:
            i_str = "n/a"
        print(f"  {fname}: expect={expect} | torch={t_lab}({t_c:.3f}) | "
              f"fp32={f_lab}({f_c:.3f}) | int8={i_str}")

    if validated == 0:
        print("[export] FATAL: no validation image found under docs/", file=sys.stderr)
        return 5

    shipped_default = "int8" if (int8_ok and int8_sess is not None and int8_preserves) else "fp32"
    if int8_ok and int8_sess is not None and not int8_preserves:
        print("[export] INT8 flipped at least one prediction -> keeping FP32 as shipped default "
              "(int8 file still saved for the record).")

    meta = {
        "labels": probe_labels,
        "input_size": [_INPUT_SIZE, _INPUT_SIZE],
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "preprocessing": "single_forward",
        "shipped_default": shipped_default,
        "source_model": "loopdesk-ai/lipika",
        "exported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[export] wrote {labels_path} (shipped_default={shipped_default})")
    print("[export] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
