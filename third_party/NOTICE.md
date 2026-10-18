# Third-party code and models

| What | Where | Licence |
|---|---|---|
| ESPnet-derived model + beam search code (Auto-AVSR, Imperial College London) | `src/espnet/` | Apache-2.0 (`LICENSE.auto_avsr-Apache-2.0`) |
| Mouth alignment logic (ported into `src/silent_hermes/roi.py`), mean face, token list | Auto-AVSR via Chaplin | Apache-2.0 / MIT (`LICENSE.chaplin-MIT`) |
| `unigram5000.model` sentencepiece model | `src/silent_hermes/assets/` | Apache-2.0 (Auto-AVSR) |
| Auto-AVSR weights `LRS3_V_WER19.1`, subword LM | `models/` (downloaded, not committed) | Weights carry the training data's terms (LRS3 etc.): personal/research use |
| BlazeFace short-range | `models/mediapipe/` | Apache-2.0 (Google MediaPipe) |
| GRID corpus (self-test only) | outside repo | CC-BY-4.0, Cooke et al. 2006, doi:10.5281/zenodo.3625687 |

Vendored espnet code is unmodified from Chaplin's copy.
