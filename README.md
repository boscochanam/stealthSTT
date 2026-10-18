# silent-hermes

Silently mouth a command at the laptop camera → Hermes runs it. Fully local, CPU-only.

Auto-AVSR lip-reading encoder + your own enrolled clips + a reject option. Design, research
and phase gates: [docs/DESIGN.md](docs/DESIGN.md).

## Setup (once)

```bash
cd ~/silent-hermes
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
VIRTUAL_ENV=.venv uv pip install -e '.[dev]'
scripts/fetch_models.sh            # ~1.2 GB, checked against models.sha256
```

## Use

1. **Edit `config/commands.yaml`**: your phrases and the Hermes prompt each one sends.
2. **Check the camera**: `.venv/bin/silent-hermes doctor` (no image is shown or saved).
3. **Enrol**: `.venv/bin/silent-hermes enrol --reps 10`. Mouth each phrase silently on GO. It also asks for `_none` clips (rest, smile, talk): these teach it what is *not* a command.
4. **Evaluate**: `.venv/bin/silent-hermes eval`. Prints accuracy, false accepts, confused pairs, PASS/FAIL against the gate, and writes `data/calibration.json`.
5. **Run**: `.venv/bin/silent-hermes run --dry-run` first (prints the hermes command), then without `--dry-run`. Enter → mouth → Enter.

Until `eval` has run, every command asks for `y` before it runs.

## Phone portal (record on iPhone)

```bash
scripts/portal_ctl.sh start      # http://<tailscale-ip>:8810/  (tailnet only, not public)
scripts/portal_ctl.sh stop       # or let it exit by itself after 30 idle minutes
```

On demand only, never a service. Runs under a 3 GB memory cap; refuses to start below ~2.5 GB free RAM.
Hermes skill: `~/.hermes/skills/productivity/silent-hermes/SKILL.md`.

On the iPhone (Tailscale on): open `http://<tailscale-ip>:8810/`, pick a phrase, pick
**Out loud / Whisper / Silent**, tap **Record with front camera**. Each upload is:
normalised (rotation, 25 fps) → speech-to-text (faster-whisper `small.en`, if there is sound)
→ lip-read → shown as *you said* vs *lips read*, word by word, plus which command it matched.

- **Keep for training** copies the mouth crops into `data/enrol/<command>/` (the same store `enrol` uses).
  It is refused when speech-to-text heard something other than the prompt.
- **Free talk** (out loud, 1–3 min): speech-to-text labels every sentence; **Save pieces** writes
  them to `data/finetune/` for the Phase-3 LoRA fine-tune.
- The uploaded video is deleted after processing; only mouth crops + text stay.

Suggested loop per command: 1 clip out loud (checks the words and your framing), then 10 silent.
Then `silent-hermes eval`.

## Layout

```
config/          settings.yaml, commands.yaml (edit these)
src/silent_hermes/
  vsr.py         Auto-AVSR wrapper: embed, phrase scoring, beam-search dictation
  roi.py         face keypoints -> aligned 96x96 mouth crops
  capture.py     headless webcam capture
  matcher.py     DTW templates + phrase scores -> fused decision with reject
  engine.py      glue: clip -> decision
  store.py       config + enrolment store (data/, mouth crops only)
  evaluate.py    leave-one-out eval + threshold calibration
  dispatch.py    hermes -z / hermes send, confirm gate
  cli.py         doctor | enrol | eval | run | decode | portal
  portal/        FastAPI upload portal: jobs.py (ffmpeg, STT, lip-read), static/index.html
src/espnet/      vendored Auto-AVSR model code (Apache-2.0)
scripts/         fetch_models.sh, grid_selftest.py, portal_ctl.sh
tests/           model-free unit tests (pytest)
reports/         self-test results
third_party/     licences + NOTICE.md
```

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/grid_selftest.py ~/silent-hermes-testdata/grid   # needs GRID s1 + alignments
```
