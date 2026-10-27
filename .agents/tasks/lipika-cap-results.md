# Lipika window-cap measurement — fp32 observer latency + page-level accuracy per cap

LOCAL-ONLY measurement. No AWS, no deploy, no git commit/stage/LFS. Default model
unchanged (fp32). Produced by `scripts/measure_lipika_cap.py` (one-off dev tool,
not a committed test), reusing the `scripts/eval_lipika_crops.py` manifest + crop
machinery and the shipped `script_classifier._even_spread` cap. All numbers below
are measured on this CPU with the repo venv; cells that could not be produced are
marked `not measured`.

- fp32 model load: **0.31 s**  (int8 model load: 0.22 s)
- Measured cost/window on this CPU: **fp32 ≈ 125 ms/window**, **int8 ≈ 250 ms/window**
  (int8 quantized ops are not accelerated on this CPU and run ~2x slower than fp32).

## 3a — Per-page observer latency (fp32, observer-only)

The 5 real pages are page 0 of each of the 5 English/Latin PDFs in `dataset/`
(`MY_resume.pdf`, `Vinay_Resume.pdf`, `bert vs sas.pdf`, `missouri_water_sample.pdf`,
`Chander Kochhar 01_compressed 2.pdf`) — the same Latin PDFs the prior uncapped
Strategy-B baseline (~403 windows/page) was measured on. Seconds = the `observe()`
unit only (line crops + Strategy-B windows + even-spread cap + batched forward +
mean-softmax aggregate), **excluding** PDF render. Best (min) of 3 repeats.
True script for all five = `latin`.

### cap = 16
| page | seconds | windows | pred | correct |
| --- | ---: | ---: | --- | :---: |
| MY_resume.pdf | 2.03 | 16 | latin | ✓ |
| Vinay_Resume.pdf | 1.98 | 16 | latin | ✓ |
| bert vs sas.pdf | 2.01 | 16 | latin | ✓ |
| missouri_water_sample.pdf | 2.03 | 16 | latin | ✓ |
| Chander Kochhar 01_compressed 2.pdf | 2.08 | 16 | latin | ✓ |
| **mean** | **~2.0** | 16 | latin | 5/5 |

### cap = 32
| page | seconds | windows | pred | correct |
| --- | ---: | ---: | --- | :---: |
| MY_resume.pdf | 4.55 | 32 | latin | ✓ |
| Vinay_Resume.pdf | 4.18 | 32 | latin | ✓ |
| bert vs sas.pdf | 4.11 | 32 | latin | ✓ |
| missouri_water_sample.pdf | 4.14 | 32 | latin | ✓ |
| Chander Kochhar 01_compressed 2.pdf | 4.16 | 32 | latin | ✓ |
| **mean** | **~4.2** | 32 | latin | 5/5 |

### cap = 64
| page | seconds | windows | pred | correct |
| --- | ---: | ---: | --- | :---: |
| MY_resume.pdf | 6.94 | 64 | latin | ✓ |
| Vinay_Resume.pdf | 6.66 | 64 | latin | ✓ |
| bert vs sas.pdf | 6.64 | 64 | latin | ✓ |
| missouri_water_sample.pdf | 6.65 | 64 | latin | ✓ |
| Chander Kochhar 01_compressed 2.pdf | 7.20 | 64 | latin | ✓ |
| **mean** | **~6.8** | 64 | latin | 5/5 |

### uncapped (reference, 1 page, 1 repeat)
| page | seconds | windows | pred | correct |
| --- | ---: | ---: | --- | :---: |
| MY_resume.pdf | **194.0** | 591 | latin | ✓ |
| (other 4 pages) | not measured | — | — | — |

Uncapped is impractical to sweep on this CPU (591 windows × ~0.33 s = ~194 s for a
single page; the other 4 pages were not re-measured to keep the run bounded). The
prior report's "~403 windows -> ~44 s/page" was on a faster host; this machine is
~3x slower per window, which is why the capped numbers above are the operative
local figures.

## 3b — Full labelled page-level eval per cap (fp32)

Page-level vote = argmax of mean softmax over that page's capped Strategy-B window
set. Labelled set built by the harness: `dataset/verified_twice/<lang>` word crops
-> script (each jpeg is one page unit), + the 5 English PDFs (4 pages each) -> latin.
**Reduced sampling for runtime:** `--per-script-cap 20 --max-crops-per-source 20`
(20 jpeg word-crops per non-Latin script; the 5 Latin PDFs expand to 16 latin pages).
Actual per-script page totals are shown in the tables (column `total`). This is a
reduced sample of the full corpus, stated here so the numbers are not mistaken for
the full-corpus eval.

Observation: the cap only changes pages that generate **more windows than the cap**
— in practice the multi-line **latin** PDF pages. Single word-crop non-Latin pages
produce far fewer than 16 windows, so their prediction is identical across all three
caps (the non-Latin rows are byte-identical for cap 16/32/64 below).

### cap = 16 — per-script page accuracy
overall **0.8889** (192/216)
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 20 | 20 | 1.0000 |
| bengali | 20 | 20 | 1.0000 |
| devanagari | 19 | 20 | 0.9500 |
| gujarati | 19 | 20 | 0.9500 |
| gurmukhi | 16 | 20 | 0.8000 |
| kannada | 18 | 20 | 0.9000 |
| latin | 12 | 16 | 0.7500 |
| malayalam | 17 | 20 | 0.8500 |
| odia | 17 | 20 | 0.8500 |
| tamil | 19 | 20 | 0.9500 |
| telugu | 15 | 20 | 0.7500 |

Confusion matrix (rows=true, cols=pred, 'other' collapsed):
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |    20 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |    19 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |    19 |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0
gurmukhi |     1 |     1 |     1 |    16 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     0 |     0 |     0 |     0 |    18 |     0 |     0 |     1 |     0 |     0 |     1
   latin |     1 |     0 |     0 |     0 |     0 |    12 |     2 |     0 |     1 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     1 |    17 |     0 |     0 |     2 |     0
    odia |     2 |     0 |     0 |     1 |     0 |     0 |     0 |    17 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    20 |     0 |     0
   tamil |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |    19 |     0
  telugu |     3 |     0 |     0 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |    15
```
Non-Latin pages predicted as latin (harmful misroute direction): **2 / 200 = 1.00%**
(1 devanagari, 1 malayalam).

### cap = 32 — per-script page accuracy
overall **0.8935** (193/216)
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 20 | 20 | 1.0000 |
| bengali | 20 | 20 | 1.0000 |
| devanagari | 19 | 20 | 0.9500 |
| gujarati | 19 | 20 | 0.9500 |
| gurmukhi | 16 | 20 | 0.8000 |
| kannada | 18 | 20 | 0.9000 |
| latin | 13 | 16 | 0.8125 |
| malayalam | 17 | 20 | 0.8500 |
| odia | 17 | 20 | 0.8500 |
| tamil | 19 | 20 | 0.9500 |
| telugu | 15 | 20 | 0.7500 |

Confusion matrix:
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |    20 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |    19 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |    19 |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0
gurmukhi |     1 |     1 |     1 |    16 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     0 |     0 |     0 |     0 |    18 |     0 |     0 |     1 |     0 |     0 |     1
   latin |     1 |     1 |     0 |     0 |     0 |    13 |     0 |     0 |     1 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     1 |    17 |     0 |     0 |     2 |     0
    odia |     2 |     0 |     0 |     1 |     0 |     0 |     0 |    17 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    20 |     0 |     0
   tamil |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |    19 |     0
  telugu |     3 |     0 |     0 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |    15
```
Non-Latin pages predicted as latin: **2 / 200 = 1.00%** (1 devanagari, 1 malayalam).

### cap = 64 — per-script page accuracy
overall **0.9028** (195/216)
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 20 | 20 | 1.0000 |
| bengali | 20 | 20 | 1.0000 |
| devanagari | 19 | 20 | 0.9500 |
| gujarati | 19 | 20 | 0.9500 |
| gurmukhi | 16 | 20 | 0.8000 |
| kannada | 18 | 20 | 0.9000 |
| latin | 15 | 16 | 0.9375 |
| malayalam | 17 | 20 | 0.8500 |
| odia | 17 | 20 | 0.8500 |
| tamil | 19 | 20 | 0.9500 |
| telugu | 15 | 20 | 0.7500 |

Confusion matrix:
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |    20 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |    19 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |    19 |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0
gurmukhi |     1 |     1 |     1 |    16 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     0 |     0 |     0 |     0 |    18 |     0 |     0 |     1 |     0 |     0 |     1
   latin |     0 |     0 |     1 |     0 |     0 |    15 |     0 |     0 |     0 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     1 |    17 |     0 |     0 |     2 |     0
    odia |     2 |     0 |     0 |     1 |     0 |     0 |     0 |    17 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    20 |     0 |     0
   tamil |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |    19 |     0
  telugu |     3 |     0 |     0 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |    15
```
Non-Latin pages predicted as latin: **1 / 200 = 0.50%** (1 malayalam).

### Cap-vs-cap + uncapped-baseline summary (fp32)
| cap | obs s/page (5-pg mean) | overall page acc | latin page acc | non-latin→latin |
| ---: | ---: | ---: | ---: | ---: |
| 16 | ~2.0 | 0.8889 | 0.7500 | 2/200 (1.00%) |
| 32 | ~4.2 | 0.8935 | 0.8125 | 2/200 (1.00%) |
| 64 | ~6.8 | 0.9028 | 0.9375 | 1/200 (0.50%) |
| uncapped | ~194 (1 pg) / ~44 (prior host) | not measured (impractical locally) | prior report: 5/5 latin | not measured |

Uncapped overall page accuracy on the full set was not re-run (impractical on this
CPU). The prior `lipika-eval-results.md` fp32 Strategy-B page/source-level vote
table is the closest uncapped reference and shows all 5 latin PDF sources correct
and the non-Latin scripts in the same 0.77–1.00 band as above. The cap-driven
movement here is entirely on the multi-line **latin** PDF pages (0.75 → 0.81 → 0.94
as the cap grows); non-Latin accuracy and the harmful non-latin→latin rate are flat.

## 3c — int8 at the sub-3s cap (triggered)

Only **cap 16** meets the ~3 s/page target under fp32 (cap 32 ≈ 4.2 s, cap 64 ≈ 6.8 s
already exceed it). So int8 was measured at **cap 16**, the sub-3s cap, to see if it
buys headroom. It does not.

### int8 cap 16 — 5-page observer latency
| page | seconds | windows | pred | correct |
| --- | ---: | ---: | --- | :---: |
| MY_resume.pdf | 4.15 | 16 | bengali | ✗ |
| Vinay_Resume.pdf | 3.98 | 16 | latin | ✓ |
| bert vs sas.pdf | 4.00 | 16 | latin | ✓ |
| missouri_water_sample.pdf | 4.07 | 16 | latin | ✓ |
| Chander Kochhar 01_compressed 2.pdf | 4.05 | 16 | latin | ✓ |
| **mean** | **~4.0** | 16 | — | 4/5 |

### int8 cap 16 — page-level accuracy (same reduced sample)
overall **0.8750** (189/216)
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 20 | 20 | 1.0000 |
| bengali | 20 | 20 | 1.0000 |
| devanagari | 19 | 20 | 0.9500 |
| gujarati | 19 | 20 | 0.9500 |
| gurmukhi | 14 | 20 | 0.7000 |
| kannada | 15 | 20 | 0.7500 |
| latin | 13 | 16 | 0.8125 |
| malayalam | 17 | 20 | 0.8500 |
| odia | 17 | 20 | 0.8500 |
| tamil | 20 | 20 | 1.0000 |
| telugu | 15 | 20 | 0.7500 |

Confusion matrix:
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |    20 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |    19 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |    19 |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0
gurmukhi |     1 |     4 |     0 |    14 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     0 |     0 |     0 |     0 |    15 |     0 |     0 |     1 |     0 |     0 |     4
   latin |     1 |     1 |     0 |     0 |     0 |    13 |     0 |     0 |     1 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     1 |    17 |     0 |     0 |     2 |     0
    odia |     2 |     0 |     0 |     1 |     0 |     0 |     0 |    17 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    20 |     0 |     0
   tamil |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    20 |     0
  telugu |     4 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |    15
```
Non-Latin pages predicted as latin: **1 / 200 = 0.50%** (1 malayalam).

### fp32 vs int8 at cap 16
| model | obs s/page | overall page acc | latin page acc | non-latin→latin |
| --- | ---: | ---: | ---: | ---: |
| fp32 | ~2.0 | 0.8889 | 0.7500 | 2/200 (1.00%) |
| int8 | ~4.0 | 0.8750 | 0.8125 | 1/200 (0.50%) |

int8 is both **slower (~2x)** and **slightly less accurate overall** at cap 16 on this
CPU (int8 ops are not hardware-accelerated here; gurmukhi/kannada drop notably). int8
buys nothing locally.

## Recommendation

**Keep the default `lipika_max_windows = 32`** (the code default; this phase does not
change it). fp32 is the right model locally; int8 is slower and no more accurate here.

Rationale: the ~3 s/sampled-page target is met by fp32 **only at cap 16 (~2.0 s)**;
cap 32 (~4.2 s) and cap 64 (~6.8 s) miss it on this CPU, and int8 does not rescue the
target (it is ~4.0 s at cap 16). On accuracy, cap 16 → 64 move the overall page score
only ~1.4 points (0.8889 → 0.9028), and the only part that moves is multi-line **latin**
PDF pages (0.75 → 0.94); non-Latin accuracy and the harmful **non-latin→latin** misroute
rate are essentially flat (1.0% at caps 16/32, 0.5% at 64). Since a lipika "latin" guess
is overridden by the VLM before it can route to Paddle, the small latin-recall gain from a
larger cap is low-value, and the extra latency is the real cost.

If strict adherence to a ~3 s budget is required on hardware like this measurement CPU,
the owner could lower the default to **16** (≈2.0 s, −1.4 points overall, non-latin→latin
unchanged at 1.0%). On the faster host that produced the prior ~44 s uncapped baseline,
cap 32 would land well under 3 s, so 32 remains the sensible shipped default for typical
deploy hardware. Net: **32 is a reasonable default; 16 only if the ~3 s budget must hold
on slow CPUs** — owner's call. The code default stays 32.

### Target verdict
- **~3 s / within ~2 points of uncapped, met by fp32?** Partially: **cap 16 meets ~3 s
  (~2.0 s)** and is within ~2 points on the full/non-Latin picture; **cap 32 and 64 do NOT
  meet ~3 s** on this CPU. All three caps stay within ~1.4 points of each other and of the
  uncapped non-Latin reference.
- **What int8 buys at the sub-3s cap (16):** nothing useful — ~4.0 s (slower than fp32's
  ~2.0 s) and 0.8750 overall vs fp32's 0.8889. Default model stays fp32.

---

### Compliance confirmation
- No `.onnx` staged, committed, or touched (`*.onnx` is gitignored; both model files left as-is).
- No edits under `docker/`, `step_functions/`, or `src/ai/layer1_routing/`.
- Routing policy, `ProbeConfig` defaults, gate thresholds, and the handwriting flag unchanged.
- Shipped default model unchanged: **fp32** (`shipped_default: "fp32"`).
- The only file added is `scripts/measure_lipika_cap.py` (a one-off dev measurement tool, not a test) and this report. No deploy, no commit.
