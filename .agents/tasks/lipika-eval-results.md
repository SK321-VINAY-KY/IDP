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
| arabic | 60 | 60 |
| bengali | 60 | 60 |
| devanagari | 60 | 60 |
| gujarati | 60 | 60 |
| gurmukhi | 60 | 60 |
| kannada | 60 | 60 |
| latin | 5 | 150 |
| malayalam | 60 | 60 |
| odia | 60 | 60 |
| tamil | 60 | 60 |
| telugu | 60 | 60 |

**Caveat:** scripts below --min-per-script sources: latin

Models run: fp32, int8

## fp32 — strategy A

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 57 | 60 | 0.9500 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 52 | 60 | 0.8667 |
| gujarati | 56 | 60 | 0.9333 |
| gurmukhi | 51 | 60 | 0.8500 |
| kannada | 51 | 60 | 0.8500 |
| latin | 101 | 150 | 0.6733 |
| malayalam | 56 | 60 | 0.9333 |
| odia | 54 | 60 | 0.9000 |
| tamil | 55 | 60 | 0.9167 |
| telugu | 50 | 60 | 0.8333 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |    60 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     2 |    52 |     0 |     2 |     0 |     1 |     0 |     1 |     0 |     1 |     1
gujarati |     1 |     0 |    56 |     2 |     0 |     0 |     0 |     1 |     0 |     0 |     0
gurmukhi |     4 |     3 |     1 |    51 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     1 |     0 |     0 |     1 |    51 |     0 |     0 |     1 |     1 |     0 |     5
   latin |     0 |     4 |     1 |     7 |     3 |   101 |     2 |     2 |    17 |     5 |     8
malayala |     0 |     0 |     0 |     0 |     0 |     1 |    56 |     1 |     0 |     1 |     1
    odia |     3 |     0 |     3 |     0 |     0 |     0 |     0 |    54 |     0 |     0 |     0
  arabic |     1 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |    57 |     0 |     0
   tamil |     1 |     0 |     2 |     0 |     0 |     0 |     0 |     1 |     0 |    55 |     1
  telugu |     2 |     0 |     0 |     1 |     7 |     0 |     0 |     0 |     0 |     0 |    50
```

### Top1-top2 softmax margin
- n=750 min=0.0048 median=0.9992 mean=0.8819 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 57 | 60 | 0.9500 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 52 | 60 | 0.8667 |
| gujarati | 56 | 60 | 0.9333 |
| gurmukhi | 51 | 60 | 0.8500 |
| kannada | 51 | 60 | 0.8500 |
| latin | 5 | 5 | 1.0000 |
| malayalam | 56 | 60 | 0.9333 |
| odia | 54 | 60 | 0.9000 |
| tamil | 55 | 60 | 0.9167 |
| telugu | 50 | 60 | 0.8333 |

## fp32 — strategy B

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 56 | 60 | 0.9333 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 52 | 60 | 0.8667 |
| gujarati | 55 | 60 | 0.9167 |
| gurmukhi | 51 | 60 | 0.8500 |
| kannada | 53 | 60 | 0.8833 |
| latin | 125 | 150 | 0.8333 |
| malayalam | 51 | 60 | 0.8500 |
| odia | 52 | 60 | 0.8667 |
| tamil | 55 | 60 | 0.9167 |
| telugu | 46 | 60 | 0.7667 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  othe |  odia |  arab |  tami |  telu
---------------------------------------------------------------------------------------------------------
 bengali |    60 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     2 |    52 |     0 |     3 |     1 |     1 |     0 |     0 |     0 |     0 |     1 |     0
gujarati |     0 |     0 |    55 |     3 |     0 |     0 |     1 |     0 |     1 |     0 |     0 |     0
gurmukhi |     4 |     2 |     2 |    51 |     0 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     2 |     0 |     0 |     0 |    53 |     0 |     0 |     0 |     2 |     1 |     0 |     2
   latin |     2 |     3 |     4 |     3 |     2 |   125 |     2 |     0 |     3 |     3 |     3 |     0
malayala |     1 |     0 |     2 |     0 |     1 |     1 |    51 |     0 |     2 |     0 |     2 |     0
   other |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
    odia |     5 |     0 |     2 |     1 |     0 |     0 |     0 |     0 |    52 |     0 |     0 |     0
  arabic |     2 |     0 |     1 |     0 |     0 |     0 |     0 |     1 |     0 |    56 |     0 |     0
   tamil |     1 |     0 |     2 |     1 |     0 |     0 |     0 |     0 |     1 |     0 |    55 |     0
  telugu |     5 |     0 |     0 |     1 |     7 |     0 |     0 |     0 |     0 |     1 |     0 |    46
```

### Top1-top2 softmax margin
- n=750 min=0.0021 median=0.9986 mean=0.8542 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 56 | 60 | 0.9333 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 52 | 60 | 0.8667 |
| gujarati | 55 | 60 | 0.9167 |
| gurmukhi | 51 | 60 | 0.8500 |
| kannada | 53 | 60 | 0.8833 |
| latin | 5 | 5 | 1.0000 |
| malayalam | 51 | 60 | 0.8500 |
| odia | 52 | 60 | 0.8667 |
| tamil | 55 | 60 | 0.9167 |
| telugu | 46 | 60 | 0.7667 |

## fp32 — strategy C

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 56 | 60 | 0.9333 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 49 | 60 | 0.8167 |
| gujarati | 56 | 60 | 0.9333 |
| gurmukhi | 50 | 60 | 0.8333 |
| kannada | 52 | 60 | 0.8667 |
| latin | 107 | 150 | 0.7133 |
| malayalam | 56 | 60 | 0.9333 |
| odia | 56 | 60 | 0.9333 |
| tamil | 54 | 60 | 0.9000 |
| telugu | 53 | 60 | 0.8833 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |    60 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     3 |    49 |     2 |     3 |     0 |     0 |     0 |     1 |     0 |     1 |     1
gujarati |     1 |     0 |    56 |     2 |     0 |     0 |     0 |     1 |     0 |     0 |     0
gurmukhi |     2 |     6 |     1 |    50 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     1 |     0 |     0 |     1 |    52 |     0 |     0 |     1 |     1 |     0 |     4
   latin |     1 |     3 |     1 |     2 |     2 |   107 |     2 |     1 |    26 |     4 |     1
malayala |     0 |     0 |     0 |     0 |     0 |     1 |    56 |     1 |     0 |     1 |     1
    odia |     3 |     0 |     0 |     0 |     0 |     1 |     0 |    56 |     0 |     0 |     0
  arabic |     2 |     0 |     2 |     0 |     0 |     0 |     0 |     0 |    56 |     0 |     0
   tamil |     2 |     1 |     2 |     0 |     0 |     0 |     0 |     0 |     0 |    54 |     1
  telugu |     1 |     0 |     0 |     1 |     5 |     0 |     0 |     0 |     0 |     0 |    53
```

### Top1-top2 softmax margin
- n=750 min=0.0046 median=0.9994 mean=0.9069 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 56 | 60 | 0.9333 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 49 | 60 | 0.8167 |
| gujarati | 56 | 60 | 0.9333 |
| gurmukhi | 50 | 60 | 0.8333 |
| kannada | 52 | 60 | 0.8667 |
| latin | 5 | 5 | 1.0000 |
| malayalam | 56 | 60 | 0.9333 |
| odia | 56 | 60 | 0.9333 |
| tamil | 54 | 60 | 0.9000 |
| telugu | 53 | 60 | 0.8833 |

## int8 — strategy A

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 54 | 60 | 0.9000 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 51 | 60 | 0.8500 |
| gujarati | 53 | 60 | 0.8833 |
| gurmukhi | 47 | 60 | 0.7833 |
| kannada | 48 | 60 | 0.8000 |
| latin | 100 | 150 | 0.6667 |
| malayalam | 55 | 60 | 0.9167 |
| odia | 55 | 60 | 0.9167 |
| tamil | 55 | 60 | 0.9167 |
| telugu | 50 | 60 | 0.8333 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  othe |  odia |  arab |  tami |  telu
---------------------------------------------------------------------------------------------------------
 bengali |    60 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     4 |    51 |     1 |     1 |     0 |     0 |     0 |     0 |     1 |     0 |     1 |     1
gujarati |     2 |     1 |    53 |     2 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
gurmukhi |     4 |     7 |     1 |    47 |     0 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     1 |     0 |     0 |     1 |    48 |     0 |     0 |     0 |     1 |     1 |     0 |     8
   latin |     3 |     3 |     1 |     6 |     2 |   100 |     1 |     0 |     2 |    14 |    15 |     3
malayala |     1 |     0 |     0 |     0 |     0 |     1 |    55 |     0 |     1 |     0 |     1 |     1
   other |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
    odia |     4 |     0 |     1 |     0 |     0 |     0 |     0 |     0 |    55 |     0 |     0 |     0
  arabic |     4 |     0 |     0 |     1 |     0 |     0 |     0 |     1 |     0 |    54 |     0 |     0
   tamil |     3 |     0 |     0 |     0 |     0 |     0 |     1 |     0 |     1 |     0 |    55 |     0
  telugu |     3 |     1 |     0 |     2 |     4 |     0 |     0 |     0 |     0 |     0 |     0 |    50
```

### Top1-top2 softmax margin
- n=750 min=0.0018 median=0.9991 mean=0.8740 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 54 | 60 | 0.9000 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 51 | 60 | 0.8500 |
| gujarati | 53 | 60 | 0.8833 |
| gurmukhi | 47 | 60 | 0.7833 |
| kannada | 48 | 60 | 0.8000 |
| latin | 5 | 5 | 1.0000 |
| malayalam | 55 | 60 | 0.9167 |
| odia | 55 | 60 | 0.9167 |
| tamil | 55 | 60 | 0.9167 |
| telugu | 50 | 60 | 0.8333 |

## int8 — strategy B

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 54 | 60 | 0.9000 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 51 | 60 | 0.8500 |
| gujarati | 52 | 60 | 0.8667 |
| gurmukhi | 49 | 60 | 0.8167 |
| kannada | 50 | 60 | 0.8333 |
| latin | 128 | 150 | 0.8533 |
| malayalam | 50 | 60 | 0.8333 |
| odia | 51 | 60 | 0.8500 |
| tamil | 54 | 60 | 0.9000 |
| telugu | 47 | 60 | 0.7833 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  othe |  odia |  arab |  tami |  telu
---------------------------------------------------------------------------------------------------------
 bengali |    60 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     5 |    51 |     1 |     1 |     1 |     0 |     0 |     0 |     0 |     0 |     1 |     0
gujarati |     2 |     1 |    52 |     3 |     0 |     0 |     0 |     0 |     2 |     0 |     0 |     0
gurmukhi |     4 |     6 |     0 |    49 |     0 |     0 |     0 |     0 |     0 |     0 |     1 |     0
 kannada |     2 |     0 |     0 |     0 |    50 |     0 |     0 |     0 |     2 |     1 |     0 |     5
   latin |     3 |     3 |     5 |     1 |     1 |   128 |     1 |     0 |     2 |     2 |     3 |     1
malayala |     2 |     0 |     3 |     0 |     0 |     1 |    50 |     0 |     2 |     0 |     2 |     0
   other |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
    odia |     7 |     0 |     1 |     1 |     0 |     0 |     0 |     0 |    51 |     0 |     0 |     0
  arabic |     4 |     0 |     1 |     0 |     0 |     0 |     0 |     1 |     0 |    54 |     0 |     0
   tamil |     4 |     0 |     0 |     1 |     0 |     0 |     0 |     0 |     1 |     0 |    54 |     0
  telugu |     7 |     1 |     0 |     0 |     4 |     0 |     0 |     0 |     0 |     1 |     0 |    47
```

### Top1-top2 softmax margin
- n=750 min=0.0018 median=0.9979 mean=0.8446 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 54 | 60 | 0.9000 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 51 | 60 | 0.8500 |
| gujarati | 52 | 60 | 0.8667 |
| gurmukhi | 49 | 60 | 0.8167 |
| kannada | 50 | 60 | 0.8333 |
| latin | 5 | 5 | 1.0000 |
| malayalam | 50 | 60 | 0.8333 |
| odia | 51 | 60 | 0.8500 |
| tamil | 54 | 60 | 0.9000 |
| telugu | 47 | 60 | 0.7833 |

## int8 — strategy C

### Per-script top-1 accuracy
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 57 | 60 | 0.9500 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 50 | 60 | 0.8333 |
| gujarati | 55 | 60 | 0.9167 |
| gurmukhi | 50 | 60 | 0.8333 |
| kannada | 48 | 60 | 0.8000 |
| latin | 114 | 150 | 0.7600 |
| malayalam | 56 | 60 | 0.9333 |
| odia | 55 | 60 | 0.9167 |
| tamil | 54 | 60 | 0.9000 |
| telugu | 51 | 60 | 0.8500 |

### Confusion matrix (rows=true, cols=pred, 'other' collapsed)
```
true\pred |  beng |  deva |  guja |  gurm |  kann |  lati |  mala |  odia |  arab |  tami |  telu
-------------------------------------------------------------------------------------------------
 bengali |    60 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |     0
devanaga |     5 |    50 |     1 |     1 |     0 |     0 |     0 |     1 |     0 |     1 |     1
gujarati |     1 |     1 |    55 |     1 |     0 |     0 |     0 |     2 |     0 |     0 |     0
gurmukhi |     2 |     7 |     1 |    50 |     0 |     0 |     0 |     0 |     0 |     0 |     0
 kannada |     1 |     1 |     0 |     0 |    48 |     0 |     0 |     1 |     1 |     0 |     8
   latin |     1 |     3 |     1 |     1 |     1 |   114 |     2 |     2 |    23 |     2 |     0
malayala |     1 |     0 |     0 |     0 |     0 |     1 |    56 |     0 |     0 |     1 |     1
    odia |     4 |     0 |     0 |     0 |     0 |     1 |     0 |    55 |     0 |     0 |     0
  arabic |     3 |     0 |     0 |     0 |     0 |     0 |     0 |     0 |    57 |     0 |     0
   tamil |     2 |     1 |     2 |     0 |     0 |     0 |     1 |     0 |     0 |    54 |     0
  telugu |     2 |     0 |     0 |     2 |     4 |     0 |     0 |     1 |     0 |     0 |    51
```

### Top1-top2 softmax margin
- n=750 min=0.0022 median=0.9994 mean=0.9043 max=1.0000

### Page/source-level vote accuracy (argmax of mean softmax per source)
- minimum crop count per source N = 1
| script | correct | total | acc |
| --- | ---: | ---: | ---: |
| arabic | 57 | 60 | 0.9500 |
| bengali | 60 | 60 | 1.0000 |
| devanagari | 50 | 60 | 0.8333 |
| gujarati | 55 | 60 | 0.9167 |
| gurmukhi | 50 | 60 | 0.8333 |
| kannada | 48 | 60 | 0.8000 |
| latin | 5 | 5 | 1.0000 |
| malayalam | 56 | 60 | 0.9333 |
| odia | 55 | 60 | 0.9167 |
| tamil | 54 | 60 | 0.9000 |
| telugu | 51 | 60 | 0.8500 |

