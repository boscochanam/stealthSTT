---
name: silent-hermes
description: "Silent lip-read commands for Hermes: start the iPhone recording portal, enrol, eval, run."
---

# silent-hermes: silent lip-read commands

Bosco mouths a command silently at a camera; a lip-reading model picks it from a fixed list and
runs it through `hermes -z`. Project: `~/silent-hermes` (git, local). Read its
`README.md` (how to use) and `docs/DESIGN.md` (research, measured numbers, phase gates) before changing anything.

## When to use
- "start the lip portal", "I want to record lip clips", "train the lip reader", "silent commands"
- Changing the command list, re-enrolling, checking accuracy, running live

## On demand only (never always-on)
The portal is NOT a service. Start it when Bosco asks; it exits by itself after 30 idle minutes
(`portal.idle_shutdown_minutes` in `config/settings.yaml`).

```bash
free -m                                                  # need >= ~2.5 GB "available"
~/silent-hermes/scripts/portal_ctl.sh start     # refuses if RAM is short
~/silent-hermes/scripts/portal_ctl.sh status
~/silent-hermes/scripts/portal_ctl.sh stop
curl -s http://$(tailscale ip -4):8810/api/status        # verify
```
URL for Bosco: `http://<tailscale-ip>:8810/`. Tailnet only, plain HTTP (it binds the Tailscale
IP; not on localhost, not in `tailscale serve`, not public). Don't add a tailscale mapping unasked.

Memory: the lip model is ~2.2 GB + whisper ~0.5 GB per job. `portal_ctl.sh` runs it in a
`systemd-run --scope` with MemoryMax=3G / MemorySwapMax=512M, so an overload kills only the portal.
Don't run it next to other heavy jobs (the laptop froze 3x on 2026-10-18 from parallel heavy jobs).

## What Bosco does on the iPhone (tell him this)
1. Tailscale on → open the URL in Safari.
2. Pick a phrase. For each command: 1 clip **Out loud** (speech-to-text confirms the words and the framing),
   then ~10 **Silent**. Tap **Record with front camera**, say it, stop, Use Video.
3. Read the card: *you said* vs *lips read* (green/red words), *command ✓/✗*, any *fix* tip (light, distance).
4. **Keep for training** on good silent clips (refused if speech-to-text heard a different phrase).
5. Also ~10 clips of **Not a command** (rest, smile, chew, chat): this is what stops false fires.
6. Optional: **Free talk** out loud 1–3 min → **Save pieces** = labelled data for a later LoRA fine-tune.

Phone at arm's length, face lit from the front, whole face in frame, mouth closed before and after.

## After clips are kept (on the laptop)
```bash
cd ~/silent-hermes
.venv/bin/silent-hermes eval                 # LOO accuracy; PASS needs >=95% accepted, <=1% false accepts
.venv/bin/silent-hermes run --dry-run        # Enter, mouth, Enter -> prints the hermes command
```
Until `eval` writes `data/calibration.json`, `run` asks `y` before every command. Commands with
`confirm: true` and all dictation always ask.

## Rules
- Commands live in `config/commands.yaml`. Phrases need 3+ syllables and distinct lip shapes
  (p/b/m, f/v, t/d/n/s look alike). Re-enrol a command after changing its phrase.
- Privacy: never open or view Bosco's clips. Uploads are deleted after processing; only 96x96 mouth
  crops + text stay in `data/` (0700, git-ignored). Inspect by counts/hash only.
- Keys/labels come from speech-to-text or the prompt, never guessed.
- Test data for self-tests: GRID corpus (CC-BY-4.0) in `~/silent-hermes-testdata/`, not Bosco's clips.

## Pitfalls
- `pkill -f "silent-hermes portal"` from a shell whose command line contains that text kills the shell itself. Use `portal_ctl.sh stop`.
- faster-whisper's own audio loader breaks with newer PyAV; the portal feeds raw 16 kHz samples instead.
- Uncalibrated matching false-fires (p=0.88 on a non-command in testing). Never skip `eval` before live use.
