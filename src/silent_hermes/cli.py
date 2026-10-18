"""silent-hermes CLI.

  doctor              camera, face detection, model speed (no images shown or saved)
  enrol [ids...]      record N silent repeats per command (and `_none` clips)
  eval                leave-one-out accuracy + write data/calibration.json
  run                 Enter to start mouthing, Enter to stop -> recognise -> Hermes
  decode FILE         lip-read a video file (free text n-best + command scores)
  portal              phone upload portal (tailnet only): STT + lip-read feedback, keep clips
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import dispatch
from .store import NEGATIVE, Store, load_calibration, load_commands, load_settings


def _load(args):
    from .roi import FaceKeypoints
    from .vsr import VSRModel

    s = load_settings(Path(args.settings) if args.settings else None)
    cmds = load_commands(Path(args.commands) if args.commands else None)
    t = time.monotonic()
    model = VSRModel(s.model_dir, s.lm_dir, threads=s.threads)
    det = FaceKeypoints(s.face_model)
    print(f"model loaded in {time.monotonic() - t:.1f}s")
    return s, cmds, model, det


def _clip_ok(clip, min_face=0.8, min_frames=15) -> str | None:
    if clip.crops is None:
        return "no face found"
    if clip.face_rate < min_face:
        return f"face only in {clip.face_rate:.0%} of frames (need {min_face:.0%}); check light/framing"
    if len(clip.crops) < min_frames:
        return "clip too short"
    return None


def cmd_doctor(args):
    from .capture import Camera, record_timed

    s, cmds, model, det = _load(args)
    cam = Camera(s.camera)
    print("Look at the camera, mouth closed, for 2 s...")
    frames, fps = cam.record(seconds=2.0)
    h, w = frames[0].shape[:2] if frames else (0, 0)
    bright = float(np.mean([f.mean() for f in frames])) if frames else 0
    lms = [det(f) for f in frames]
    rate = sum(l is not None for l in lms) / max(1, len(lms))
    eye = [np.linalg.norm(l[0] - l[1]) for l in lms if l is not None]
    line = f"camera {w}x{h} @ {fps:.1f} fps measured | brightness {bright:.0f}/255 | face in {rate:.0%} of frames"
    if eye:
        line += f" | eye distance {np.median(eye):.0f}px"
    print(line)
    if eye and np.median(eye) < 50:
        print("  ! face is small: sit closer (eye distance under ~50px loses lip detail)")
    if bright < 60:
        print("  ! image is dark: add front light")
    clip = record_timed(cam, det, s.clip_seconds)
    cam.close()
    bad = _clip_ok(clip)
    if bad:
        print("  !", bad)
        return 1
    t = time.monotonic()
    emb = model.embed(clip.crops)
    t_emb = time.monotonic() - t
    t = time.monotonic()
    model.score_phrases(emb, [c.phrase for c in cmds])
    print(f"clip {len(clip.crops)} frames | encode {t_emb:.2f}s | score {len(cmds)} phrases {time.monotonic() - t:.2f}s")
    print("doctor: OK")
    return 0


def cmd_enrol(args):
    from .capture import Camera, record_timed

    s, cmds, model, det = _load(args)
    store = Store(s.data_dir)
    by_id = {c.id: c for c in cmds}
    targets = args.ids or [c.id for c in cmds] + ([NEGATIVE] if args.negatives else [])
    for t in targets:
        if t != NEGATIVE and t not in by_id:
            sys.exit(f"unknown command id {t!r}")
    cam = Camera(s.camera)
    print(f"{args.reps} reps each, {s.clip_seconds}s window. Mouth the phrase silently when you see GO.\n"
          "Keys after each take: Enter = keep, r = redo, s = skip command, q = quit.")
    try:
        for rnd in range(args.reps):  # interleave commands so lighting/fatigue spread evenly
            for cid in targets:
                label = "(anything that is NOT a command: rest, smile, talk, yawn)" if cid == NEGATIVE else f'"{by_id[cid].phrase}"'
                while True:
                    have = len(store.items(cid))
                    input(f"\n[{cid} {have}] {label}  - press Enter, then GO after 0.5s ")
                    time.sleep(0.5)
                    print("  GO", flush=True)
                    clip = record_timed(cam, det, s.clip_seconds)
                    bad = _clip_ok(clip)
                    if bad:
                        print("  x", bad, "- redo")
                        continue
                    k = input("  keep? [Enter/r/s/q] ").strip().lower()
                    if k == "q":
                        return 0
                    if k == "r":
                        continue
                    if k == "s":
                        break
                    emb = model.embed(clip.crops).cpu().numpy()
                    store.add(cid, clip.crops, emb)
                    break
    finally:
        cam.close()
        print("\ncounts:", store.counts())
    return 0


def _stored_features(model, store, cmds):
    import torch

    ids = [c.id for c in cmds]
    phrases = [c.phrase for c in cmds]
    labels, embs, scores = [], [], []
    neg = bool(store.items(NEGATIVE))
    for cid in ids + ([NEGATIVE] if neg else []):
        for p in store.items(cid):
            e = np.load(p)["emb"].astype(np.float32)
            sc = model.score_phrases(torch.from_numpy(e), phrases)
            if neg:
                sc = np.append(sc, np.median(sc))
            labels.append(cid)
            embs.append(e)
            scores.append(sc)
    return labels, embs, np.array(scores), ids + ([NEGATIVE] if neg else [])


def cmd_eval(args):
    from .evaluate import evaluate
    from .vsr import VSRModel

    s = load_settings(Path(args.settings) if args.settings else None)
    cmds = load_commands(Path(args.commands) if args.commands else None)
    store = Store(s.data_dir)
    model = VSRModel(s.model_dir, s.lm_dir, threads=s.threads)
    labels, embs, scores, ids = _stored_features(model, store, cmds)
    counts = {i: labels.count(i) for i in ids}
    thin = [i for i, n in counts.items() if n < 3]
    if thin:
        print("need >= 3 clips each; short:", thin)
        return 1
    rep = evaluate(labels, embs, scores, ids, max_false_accept=args.max_fa)
    print(json.dumps(asdict(rep), indent=2))
    gate = rep.accept_rate >= 0.95 and rep.false_accept_rate <= args.max_fa
    print(f"\nGATE (>=95% accepted correctly, <= {args.max_fa:.0%} false accepts): {'PASS' if gate else 'FAIL'}")
    if not args.no_write:
        (s.data_dir / "calibration.json").write_text(json.dumps(rep.calibration, indent=2))
        (s.data_dir / "eval_report.json").write_text(json.dumps(asdict(rep), indent=2))
        print(f"wrote {s.data_dir / 'calibration.json'}")
    return 0 if gate else 2


def _dictate(cam, det, model, s):
    from .capture import record_until_enter

    input("  dictation: press Enter, mouth your text, press Enter to stop ")
    clip = record_until_enter(cam, det, max_seconds=15)
    bad = _clip_ok(clip)
    if bad:
        print("  x", bad)
        return None
    hyps = model.transcribe(model.embed(clip.crops), nbest=3)
    for i, h in enumerate(hyps, 1):
        print(f"   {i}. {h.text.lower()}")
    pick = input("  pick 1-3, type a correction, or Enter to cancel: ").strip()
    if not pick:
        return None
    return hyps[int(pick) - 1].text.lower() if pick.isdigit() and 1 <= int(pick) <= len(hyps) else pick


def cmd_run(args):
    from .capture import Camera, record_until_enter
    from .engine import Recogniser

    s, cmds, model, det = _load(args)
    store = Store(s.data_dir)
    calib = load_calibration(s.data_dir)
    rec = Recogniser(model, cmds, store, calib)
    by_id = {c.id: c for c in cmds}
    if not any(store.items(c.id) for c in cmds):
        print("note: nothing enrolled; using the model's phrase scores only (expect weak accuracy)")
    # Uncalibrated thresholds false-fire on non-commands (seen on GRID: p=0.88 for a non-command
    # sentence), so every command asks first until `eval` has written calibration.json.
    confirm_all = not calib
    if confirm_all:
        print("note: not calibrated yet (run `silent-hermes eval`); every command will ask before running")
    cam = Camera(s.camera)
    print("Enter = start, mouth a command, Enter = stop. Ctrl-C quits.")
    try:
        while True:
            input("\n> ready ")
            clip = record_until_enter(cam, det, max_seconds=8)
            bad = _clip_ok(clip)
            if bad:
                print("  x", bad)
                continue
            t = time.monotonic()
            res, _ = rec.recognise(clip.crops)
            ms = (time.monotonic() - t) * 1000
            top = ", ".join(f"{k} {v:.2f}" for k, v in list(res.probs.items())[:3])
            if res.command is None:
                print(f"  ? not sure ({top}) [{ms:.0f} ms] - nothing run")
                continue
            cmd = by_id[res.command]
            print(f'  heard: "{cmd.phrase}" p={res.prob:.2f} [{ms:.0f} ms]')
            text = None
            if cmd.action.get("type") == "dictate":
                text = _dictate(cam, det, model, s)
                if text is None:
                    print("  cancelled")
                    continue
            argv = dispatch.build_argv(cmd, s, text)
            if (cmd.confirm or confirm_all or text is not None) and not dispatch.ask_yes(f"  run {argv[-1][:80]!r}? [y/N] "):
                print("  cancelled")
                continue
            dispatch.run(argv, s, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()
    return 0


def cmd_decode(args):
    from .engine import Recogniser
    from .roi import crops_from_video

    s, cmds, model, det = _load(args)
    crops, rate = crops_from_video(Path(args.file), det)
    if crops is None:
        sys.exit("no face found")
    print(f"{len(crops)} frames, face in {rate:.0%}")
    emb = model.embed(crops)
    for h in model.transcribe(emb, nbest=args.nbest):
        print(f"  {h.score:8.2f}  {h.text}")
    rec = Recogniser(model, cmds, Store(s.data_dir), load_calibration(s.data_dir))
    res, _ = rec.recognise(crops)
    print("command:", res.command, f"(best {res.best} p={res.prob:.2f} margin={res.margin:.2f})")
    return 0


def cmd_portal(args):
    import subprocess

    from .portal.app import serve

    s = load_settings(Path(args.settings) if args.settings else None)
    cfg = s.extra.get("portal", {}) or {}
    host = args.host or cfg.get("host", "127.0.0.1")
    if host == "tailscale":
        host = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, check=True).stdout.split()[0]
    port = args.port or int(cfg.get("port", 8810))
    idle = float(cfg.get("idle_shutdown_minutes", 30))
    print(f"portal on http://{host}:{port}/ (stops after {idle:g} idle min)", flush=True)
    serve(host, port, idle, Path(args.settings) if args.settings else None)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="silent-hermes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings")
    ap.add_argument("--commands")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("enrol")
    p.add_argument("ids", nargs="*")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--no-negatives", dest="negatives", action="store_false")
    p.set_defaults(fn=cmd_enrol)
    p = sub.add_parser("eval")
    p.add_argument("--max-fa", type=float, default=0.01)
    p.add_argument("--no-write", action="store_true")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("run")
    p.add_argument("--dry-run", action="store_true", help="print the hermes command instead of running it")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("decode")
    p.add_argument("file")
    p.add_argument("--nbest", type=int, default=3)
    p.set_defaults(fn=cmd_decode)
    p = sub.add_parser("portal")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(fn=cmd_portal)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
