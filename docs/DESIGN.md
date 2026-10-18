# silent-hermes: design

Goal: mouth a command silently at the laptop camera and have Hermes run it. Local only.

## Why this shape (research, Oct 2026)

| Finding | Source | Consequence |
|---|---|---|
| Best open-sentence lip reading is still ~19% WER on LRS3 (VALLR 18.7, Auto-AVSR 19.1). Best private-data result is 12.5%. | [VALLR, ICCV 2025](https://arxiv.org/abs/2503.21408), [Auto-AVSR](https://github.com/mpc001/auto_avsr) | Free dictation will be wrong often. Commands must be a closed set. |
| Auto-AVSR is mature, ~250M params, open code (Apache-2.0); Chaplin runs it live on a webcam. | [Chaplin](https://github.com/amanvirparhar/chaplin) | Use it as the encoder/decoder, vendored (`src/espnet`). |
| Speaker adaptation helps most where error is highest; >half the gain within ~7 min of data. | [Yeo et al., AAAI 2025](https://arxiv.org/abs/2409.00986), [OLKAVS 2026](https://arxiv.org/pdf/2609.28988) | Enrol on Bosco's own lips; LoRA later. |
| All benchmarks are voiced speech. Silent mouthing is smaller and not covered. | (inference, no paper found) | Enrolment clips must be silent, in the real setup. |

## Pipeline

```
camera (640x480, ~30fps) -> resample 25fps -> BlazeFace keypoints (eyes, nose, mouth)
  -> similarity-align to mean face -> 96x96 grey mouth crop      [roi.py]
  -> Auto-AVSR conformer encoder -> (T, 768) frames              [vsr.py]
       |-> phrase score: log P(phrase | video), decoder + CTC, per command  (zero-shot)
       |-> template score: DTW to Bosco's enrolled clips per command        (personal)
  -> fused probabilities -> accept only if p >= p_min and margin >= m       [matcher.py]
  -> `_none` class (enrolled non-command clips) absorbs idle mouth movement
  -> confirm gate for side effects -> `hermes -z "<prompt>"`                 [dispatch.py]
```

Dictation (`type: dictate`): beam search + subword LM, show 3-best, Bosco picks or types a fix,
confirms, then it goes to Hermes. Never auto-sent.

## Measured on this laptop (Ryzen 7 5700U, CPU only)

- Model load 5–7 s, ~2.2 GB RSS.
- Command path: encode 2.5 s clip 0.5 s + score 5 phrases 0.13 s → **~0.6 s**.
- Dictation beam search (beam 10 + LM): ~6 s.
- `hermes -z` round trip for a trivial prompt: 17 s (Hermes, not lip reading, dominates).

## Self-test: GRID speaker s1 (`scripts/grid_selftest.py`, report in `reports/grid_s1.json`)

16 two-word commands (`bin|lay|place|set` x `blue|green|red|white`), 10 clips each, 30 `_none`
clips (sentence tails), leave-one-out. Spoken studio video, so it proves the pipeline, not silent
laptop accuracy.

| Metric | Value |
|---|---|
| Top-1, model phrase scores only (zero enrolment) | 83.8% |
| Top-1, your-clips templates (DTW) | **91.9%** |
| Accepted correctly at <=1% false accepts | 86.3% (gate wants 95%: FAIL) |
| False accepts | 0.5% (1 of 190) |
| Main confusions | red <-> white, every verb (near-identical lip shapes) |

Read: the method works with 10 clips per command, and the failures are the look-alike pairs.
Real commands should be longer (3+ syllables) and pairwise distinct; GRID's are 0.6 s and
deliberately confusable. Fusion picked templates only here (phrase scores added nothing).

## Phases and gates

| Phase | What | Gate |
|---|---|---|
| 0 Spike | `doctor`: camera, face rate, latency | face in ≥95% of frames, encode < 1 s ✔ (latency) |
| 1 Enrol + match | `enrol` 10 silent reps per command + 10 `_none`; `eval` LOO | ≥95% of command clips accepted correctly, ≤1% false accepts |
| 2 Live | `run --dry-run` for a day, then live | no wrong command executed |
| 3 LoRA (if gate 1 fails) | adapter fine-tune on ~20 min of labelled silent clips, on the LAN GPU box (RTX 2070, 8 GB) | beats Phase-1 eval on the same held-out clips |
| 1b Phone portal | record on iPhone, STT checks the words, lip-read feedback per word, keep good clips | built ✔ |
| 4 Hands-free (optional) | mouth-motion endpointing instead of Enter; global hotkey | false trigger rate measured over an idle hour |

Phase 3 and 4 are not built yet. Phase 3 needs Phase-1 data first (the mouth crops `enrol`
stores are exactly the training input).

## Privacy

- No preview window; full frames live only in memory during a clip.
- Stored: 96x96 grey mouth crops + encoder features, in `data/` (mode 0700, git-ignored).
- Phone portal: plain HTTP bound to the Tailscale IP only (tailnet devices; not LAN, not public).
  Uploaded videos are deleted after processing (`portal.keep_video: false`).
- `data/` never leaves this machine unless Phase 3 copies it to the LAN GPU box, by explicit step.

## Known limits

- Similar mouth shapes collide (p/b/m, f/v, t/d/n/s). Eval prints the confusion pairs; rename phrases that collide.
- Lighting/angle shift hurts. Enrol in the setup you will use; re-enrol if accuracy drops.
- GRID self-test is spoken studio video: it proves the pipeline, not silent laptop accuracy.
