"""End-to-end self-test on the GRID corpus (CC-BY-4.0), one speaker standing in for "you".

GRID sentences start with <verb> <colour> (bin|lay|place|set x blue|green|red|white): 16
two-word "commands". Each clip is trimmed to those two words using the word alignments;
the sentence tail (letter digit adverb) supplies `_none` clips. Then the same leave-one-out
evaluation as `silent-hermes eval` runs on it.

Caveat: GRID is spoken aloud in a studio, so this checks the pipeline and the method,
not how well silent mouthing on a laptop camera will work.

usage: python scripts/grid_selftest.py ~/silent-hermes-testdata/grid --reps 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from silent_hermes.evaluate import evaluate
from silent_hermes.roi import FaceKeypoints, crops_from_video
from silent_hermes.store import NEGATIVE, load_settings
from silent_hermes.vsr import VSRModel

PAD = 2  # frames of context kept either side of the trimmed span


def words(align: Path) -> list[tuple[int, int, str]]:
    out = []
    for line in align.read_text().split("\n"):
        p = line.split()
        if len(p) == 3 and p[2] not in ("sil", "sp"):
            out.append((int(p[0]) // 1000, int(p[1]) // 1000, p[2]))  # GRID units: 1000 per 25 fps frame
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_root", type=Path)
    ap.add_argument("--speaker", default="s1")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--negatives", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("grid_selftest_report.json"))
    a = ap.parse_args()

    vids = sorted((a.grid_root / a.speaker / a.speaker).glob("*.mpg"))
    aligns = a.grid_root / "align" / "alignments" / a.speaker
    s = load_settings()
    model = VSRModel(s.model_dir, s.lm_dir, threads=s.threads)
    det = FaceKeypoints(s.face_model)

    per_cls, negs = defaultdict(list), []
    t0, skipped = time.monotonic(), 0
    for v in vids:
        w = words(aligns / f"{v.stem}.align")
        if len(w) != 6:
            continue
        cls = f"{w[0][2]}_{w[1][2]}"
        need_cls = len(per_cls[cls]) < a.reps
        need_neg = len(negs) < a.negatives and not need_cls
        if not (need_cls or need_neg):
            if all(len(x) >= a.reps for x in per_cls.values()) and len(per_cls) == 16 and len(negs) >= a.negatives:
                break
            continue
        crops, rate = crops_from_video(v, det)
        if crops is None or rate < 0.9:
            skipped += 1
            continue
        if need_cls:
            lo, hi = max(0, w[0][0] - PAD), min(len(crops), w[1][1] + PAD)
            per_cls[cls].append(model.embed(crops[lo:hi]).numpy())
        else:
            lo, hi = max(0, w[3][0] - PAD), min(len(crops), w[5][1] + PAD)
            negs.append(model.embed(crops[lo:hi]).numpy())
    print(f"featurised in {time.monotonic() - t0:.0f}s, skipped {skipped} (face < 90%)", file=sys.stderr)

    ids = sorted(per_cls)
    phrases = [c.replace("_", " ") for c in ids]
    labels, embs = [], []
    for c in ids:
        labels += [c] * len(per_cls[c])
        embs += per_cls[c]
    labels += [NEGATIVE] * len(negs)
    embs += negs
    import torch

    scores = []
    for e in embs:
        sc = model.score_phrases(torch.from_numpy(e), phrases)
        scores.append(np.append(sc, np.median(sc)))
    rep = evaluate(labels, embs, np.array(scores), ids + [NEGATIVE], max_false_accept=0.01)
    out = asdict(rep) | {"speaker": a.speaker, "clips_per_command": {c: len(per_cls[c]) for c in ids},
                         "negatives": len(negs)}
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
