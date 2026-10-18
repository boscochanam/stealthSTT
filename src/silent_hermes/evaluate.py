"""Leave-one-out evaluation and threshold calibration over enrolled clips.

For each clip: templates = every other clip; phrase scores = model's log P(phrase | clip).
Searches fusion weights/temperatures, then picks the accept thresholds that give the most
correct accepts while keeping false accepts (wrong command, or a `_none` clip accepted as a
command) at or under `max_false_accept`.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np

from .matcher import Calibration, _log_softmax, dtw_batch
from .store import NEGATIVE


@dataclass
class Report:
    n_clips: int
    n_commands: int
    top1_phrase_only: float      # zero-enrolment baseline
    top1_template_only: float
    top1_fused: float
    accept_rate: float           # command clips accepted correctly at chosen thresholds
    false_accept_rate: float     # over all clips
    calibration: dict
    confusions: list


def pairwise_dtw(embs: list[np.ndarray]) -> np.ndarray:
    n = len(embs)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        D[i] = dtw_batch(embs[i], embs)
    np.fill_diagonal(D, np.inf)
    return D


def template_dist(D: np.ndarray, labels: np.ndarray, ids: list[str], k: int) -> np.ndarray:
    out = np.zeros((len(labels), len(ids)))
    for c, cid in enumerate(ids):
        cols = D[:, labels == cid]
        if cols.shape[1] == 0:
            out[:, c] = np.inf
            continue
        s = np.sort(cols, axis=1)[:, :k]
        s = np.where(np.isinf(s), np.nan, s)
        out[:, c] = np.nanmean(s, axis=1)
    finite = out[np.isfinite(out)]
    return np.where(np.isfinite(out), out, (finite.max() + 1) if finite.size else 1.0)


def _probs(phrase, dist, cal: Calibration) -> np.ndarray:
    lp = np.zeros_like(phrase)
    if cal.w_phrase:
        lp += cal.w_phrase * np.apply_along_axis(_log_softmax, 1, phrase / cal.t_phrase)
    if cal.w_template:
        lp += cal.w_template * np.apply_along_axis(_log_softmax, 1, -dist / cal.t_template)
    return np.exp(np.apply_along_axis(_log_softmax, 1, lp))


def _thresholds(probs, y, neg_idx, max_fa):
    order = np.sort(probs, 1)
    top, margin = order[:, -1], order[:, -1] - order[:, -2]
    pred = probs.argmax(1)
    is_cmd = y != neg_idx if neg_idx is not None else np.ones_like(y, bool)
    pred_cmd = pred != neg_idx if neg_idx is not None else np.ones_like(y, bool)
    best = (-1.0, 1.0, 0.95, 0.5)
    for p_min in np.arange(0.2, 0.99, 0.025):
        for m in np.arange(0.0, 0.8, 0.025):
            acc = (top >= p_min) & (margin >= m) & pred_cmd
            fa = (acc & (pred != y)).mean()
            ca = (acc & (pred == y) & is_cmd).sum() / max(1, is_cmd.sum())
            if fa <= max_fa and ca > best[0]:
                best = (ca, fa, float(p_min), float(m))
    return best


def evaluate(labels: list[str], embs: list[np.ndarray], phrase_scores: np.ndarray, ids: list[str],
             max_false_accept: float = 0.01) -> Report:
    """phrase_scores: (N, len(ids)) model scores, with a neutral column for NEGATIVE if present."""
    y_lab = np.array(labels)
    y = np.array([ids.index(l) for l in labels])
    neg_idx = ids.index(NEGATIVE) if NEGATIVE in ids else None
    is_cmd = y != neg_idx if neg_idx is not None else np.ones_like(y, bool)
    D = pairwise_dtw(embs)

    def top1(p):
        return float((p.argmax(1)[is_cmd] == y[is_cmd]).mean())

    # NEGATIVE is always the last id, so dropping its column keeps command indices intact
    ph_cmd = phrase_scores if neg_idx is None else np.delete(phrase_scores, neg_idx, 1)
    base_phrase = float((ph_cmd.argmax(1)[is_cmd] == y[is_cmd]).mean())
    dists = {k: template_dist(D, y_lab, ids, k) for k in (1, 3)}
    base_tmpl = top1(-dists[3])

    best = None
    for wp, wt, tp, tt, k in itertools.product([0, 0.5, 1], [0, 0.5, 1, 2], [0.1, 0.3, 1.0],
                                               [0.005, 0.01, 0.02, 0.05], [1, 3]):
        if wp == 0 and wt == 0:
            continue
        cal = Calibration(w_phrase=wp, w_template=wt, t_phrase=tp, t_template=tt, k=k)
        p = _probs(phrase_scores, dists[k], cal)
        ca, fa, pm, mg = _thresholds(p, y, neg_idx, max_false_accept)
        key = (ca, top1(p))
        if best is None or key > best[0]:
            cal.p_min, cal.margin = pm, mg
            best = (key, cal, p, fa)
    (ca, t1), cal, p, fa = best
    pred = p.argmax(1)
    conf = {}
    for t, q in zip(y[is_cmd], pred[is_cmd]):
        if t != q:
            conf[(ids[t], ids[q])] = conf.get((ids[t], ids[q]), 0) + 1
    confusions = sorted(([a, b, n] for (a, b), n in conf.items()), key=lambda r: -r[2])[:10]
    return Report(len(labels), len([i for i in ids if i != NEGATIVE]), base_phrase, base_tmpl, t1,
                  ca, fa, asdict(cal), confusions)
