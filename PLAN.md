# Handoff: state and plan (2026-10-18)

Development stopped mid-change. This file says what works, what doesn't, and what to do next.
Background: `README.md` (usage), `docs/DESIGN.md` (research, measurements, gates).

## Goal

Mouth a command silently at a camera. A lip-reading model picks it from a fixed list
(`config/commands.yaml`) and runs it via `hermes -z "<prompt>"`. Everything runs locally on a
CPU laptop (Ryzen 7 5700U, 7 GB RAM).

## Verified

| Piece | Evidence |
|---|---|
| Auto-AVSR wrapper (`vsr.py`): encode, phrase scoring, beam search | CPU: ~0.6 s per 2.5 s command clip; ~6 s dictation |
| Mouth crop pipeline (`roi.py`, MediaPipe Tasks BlazeFace) | GRID clips decode to partly right text ("bin blue at f two now" -> "pimp you have two now") |
| iPhone rotation handling (ffmpeg autorotate) | correctly tagged portrait HEVC .mov reads the same as the original |
| Matcher + LOO eval + calibration (`matcher.py`, `evaluate.py`) | GRID s1, 16 two-word commands x 10 clips: 91.9% top-1, 86% accepted at <=1% false accepts (`reports/grid_s1.json`) |
| `hermes -z` dispatch | "pong" round trip, 17 s |
| Unit tests | `pytest`: 11 pass (model-free) |
| Portal v1 (per-clip UI) | worked end to end on test video. **Superseded, see below.** |

## In progress (uncommitted before this push, NOT verified end to end)

Bosco's intended flow, replacing the per-clip UI. There is **one recording mode**:

1. The portal shows a reading script (`script.py: make_script`): each command phrase x 10, shuffled,
   plus 12 everyday filler sentences (these become the `_none` "not a command" class).
2. Bosco reads it **out loud** on iPhone video, in one or several videos, then uploads.
3. `portal/jobs.py`: ffmpeg normalise -> faster-whisper word timestamps -> `script.find_phrases`
   locates every command/filler line -> cuts the mouth crops for each hit (+-4 frames) ->
   adds them to `data/enrol/<id>/` automatically. Deleting an upload removes the clips it added.
4. The card shows per command: found N, and "lips read k/N" (zero-shot readability check).

Status:
- `script.py`, the new `jobs.py`/`app.py`/`index.html`: written; unit tests for `find_phrases` and `make_script` pass.
- **Not yet run end to end.** A test was mid-flight: 45 script lines synthesised with Piper TTS
  over GRID face video as an iPhone-style HEVC .mov, uploaded to a portal with a scratch data dir.
  It was stopped before it finished. Repeat it (recipe below).

## Next steps (in order)

1. **E2E test of the read-aloud flow.** Synthesise script lines (Piper `en_US-lessac-medium`; edge-tts
   returned no audio from this host), mux over talking-face video, upload to a portal started with
   `--settings <scratch settings with data_dir elsewhere>`. Check: lines heard ~= lines read, clips per
   command, delete undoes the enrolment. The lips won't match TTS audio, so ignore the readability numbers.
2. **Fix the known gap: voiced vs silent.** The read-aloud clips are *voiced* lips; live use is
   *silent* mouthing (smaller movements). Options: read in a whisper (STT still works, lips closer to silent);
   or add a short silent pass validated by the voiced templates. Measure with `silent-hermes eval`
   on a held-out set of real silent clips before trusting it.
3. Update `docs/hermes-skill.md` (and the live Hermes skill it was copied from) to the one-mode flow.
   It still describes the v1 per-clip buttons.
4. Bosco's real command list into `config/commands.yaml` (current list is a placeholder).
   Phrase rules: 3+ syllables, distinct lip shapes (p/b/m, f/v, t/d/n/s collide).
5. Phase 1 gate: `silent-hermes eval` >= 95% accepted, <= 1% false accepts. Then `run --dry-run`.
6. If the gate fails: Phase 3 LoRA adapter fine-tune on the LAN GPU box (RTX 2070, 8 GB) using the enrolled crops.

## Constraints (keep)

- Portal is **on demand**, never a service: `scripts/portal_ctl.sh start|stop`; it exits after 30 idle
  minutes; runs under `systemd-run --scope` MemoryMax=3G / MemorySwapMax=512M and refuses to start
  below 2.5 GB free RAM. The laptop froze 3x on 2026-10-18 from parallel heavy jobs: one heavy job at a time.
- Tailnet only (binds the Tailscale IP, plain HTTP). Never public.
- Privacy: nobody (human or agent) opens Bosco's videos. Uploads are deleted after processing; only
  96x96 mouth crops + text stay in `data/` (0700, git-ignored). Use GRID (CC-BY-4.0) for tests.
- Uncalibrated matching false-fires (p=0.88 on a non-command). `run` asks before every command until `eval` has run.
- `pkill -f "silent-hermes portal"` from a shell whose own command line contains that text kills the shell.
  Use `portal_ctl.sh stop`, or kill by the port's PID.

## Setup from a fresh clone

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
VIRTUAL_ENV=.venv uv pip install -e '.[dev]' httpx
scripts/fetch_models.sh          # ~1.2 GB weights, sha256-checked; not in git
.venv/bin/python -m pytest -q
```
GRID test data (for `scripts/grid_selftest.py`): Zenodo record 3625687, `s1.zip` + `alignments.zip`.
