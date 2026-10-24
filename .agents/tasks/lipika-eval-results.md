# Lipika script ONNX eval — fp32 vs int8 under strategies A/B/C

> SMOKE RUN (reduced sampling) — not the full eval. Re-run without --min-per-script/--per-script-cap limits for final numbers.

## Strategies
- **A** — full-crop letterbox = current observer path (`script_classifier._letterbox`: white pad to square, cv2.resize INTER_CUBIC, /255, NCHW).
- **B** — square-ish windows (aspect 1:1..2:1) cut along the longer axis, each letterboxed as A; softmax averaged over windows.
- **C** — faithful numpy/PIL REPLICA of `fontrecog.infer.predictor.letterbox` (PIL RGB, pad to square fill=255, Image.BICUBIC resize to 224, /255, CHW). **fontrecog is NOT imported** (its letterbox returns a torch.Tensor and pulls torch).

## Sources & crops
Per-script SOURCE counts (crops=jpeg files, pdfs=PDF files):

| script | sources | crops produced |
| --- | ---: | ---: |
| arabic | 3 | 3 |
| bengali | 3 | 3 |
| devanagari | 3 | 3 |
| gujarati | 3 | 3 |
| gurmukhi | 3 | 3 |
| kannada | 3 | 3 |
| latin | 5 | 20 |
| malayalam | 3 | 3 |
| odia | 3 | 3 |
| tamil | 3 | 3 |
| telugu | 3 | 3 |

Models run: fp32, int8

## fp32 — strategy A

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 12 | 20 | 0.6000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gurmukhi |     1 |     0 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |     0 |     0
 kannada |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0
   latin |     0 |     0 |     0 |     1 |     2 |    12 |     1 |     0 |     4 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0
    odia |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0
   tamil |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0
  telugu |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     2
```

### Top1-top2 softmax margin
- n=50 min=0.1524 median=0.9988 mean=0.8609 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 4 | 5 | 0.8000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

## fp32 — strategy B

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 14 | 20 | 0.7000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gurmukhi |     1 |     0 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |     0 |     0
 kannada |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0
   latin |     1 |     0 |     0 |     1 |     0 |    14 |     0 |     0 |     3 |     1 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0
    odia |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0
   tamil |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0
  telugu |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     2
```

### Top1-top2 softmax margin
- n=50 min=0.1161 median=0.9974 mean=0.8419 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 4 | 5 | 0.8000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

## fp32 — strategy C

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 14 | 20 | 0.7000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gurmukhi |     0 |     1 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |     0 |     0
 kannada |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0
   latin |     0 |     0 |     0 |     1 |     1 |    14 |     1 |     0 |     3 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0
    odia |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0
   tamil |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0
  telugu |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     2
```

### Top1-top2 softmax margin
- n=50 min=0.2192 median=0.9981 mean=0.8451 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 4 | 5 | 0.8000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

## int8 — strategy A

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 13 | 20 | 0.6500 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 3 | 3 | 1.0000 |
| telugu | 2 | 3 | 0.6667 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gurmukhi |     1 |     0 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |     0 |     0
 kannada |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0
   latin |     1 |     0 |     0 |     1 |     1 |    13 |     1 |     0 |     3 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0
    odia |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0
   tamil |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0
  telugu |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     2
```

### Top1-top2 softmax margin
- n=50 min=0.0150 median=0.9990 mean=0.8293 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 4 | 5 | 0.8000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 3 | 3 | 1.0000 |
| telugu | 2 | 3 | 0.6667 |

## int8 — strategy B

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 16 | 20 | 0.8000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 3 | 3 | 1.0000 |
| telugu | 3 | 3 | 1.0000 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gurmukhi |     1 |     0 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |     0 |     0
 kannada |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0
   latin |     1 |     0 |     0 |     0 |     0 |    16 |     0 |     0 |     2 |     1 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0
    odia |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0
   tamil |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0
  telugu |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3
```

### Top1-top2 softmax margin
- n=50 min=0.0150 median=0.9954 mean=0.8123 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 4 | 5 | 0.8000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 3 | 3 | 1.0000 |
| telugu | 3 | 3 | 1.0000 |

## int8 — strategy C

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 15 | 20 | 0.7500 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gujarati |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
gurmukhi |     0 |     1 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |     0 |     0
 kannada |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0 |     0 |     0
   latin |     0 |     0 |     0 |     0 |     1 |    15 |     1 |     0 |     3 |     0 |     0
malayala |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0 |     0 |     0
    odia |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
  arabic |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     3 |     0 |     0
   tamil |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     0 |     2 |     0
  telugu |     0 |     0 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |     2
```

### Top1-top2 softmax margin
- n=50 min=0.1701 median=0.9991 mean=0.8559 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 3 | 3 | 1.0000 |
| bengali | 3 | 3 | 1.0000 |
| devanagari | 3 | 3 | 1.0000 |
| gujarati | 3 | 3 | 1.0000 |
| gurmukhi | 2 | 3 | 0.6667 |
| kannada | 3 | 3 | 1.0000 |
| latin | 4 | 5 | 0.8000 |
| malayalam | 3 | 3 | 1.0000 |
| odia | 2 | 3 | 0.6667 |
| tamil | 2 | 3 | 0.6667 |
| telugu | 2 | 3 | 0.6667 |

