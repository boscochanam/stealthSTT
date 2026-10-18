"""Closed-set command recognition with a reject option.

Two evidence sources, fused in log-probability space:
  phrase   - the model's own log P(phrase | video) for each command phrase (works with zero enrolment)
  template - DTW distance between the clip's encoder frames and your enrolled clips (personal)
A guess is accepted only if its fused probability and its margin over the runner-up clear
calibrated thresholds; otherwise the result is "reject" and nothing runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def dtw_batch(q: np.ndarray, temps: list[np.ndarray]) -> np.ndarray:
    """Length-normalised DTW (cosine cost) from query (n, d) to each template (m_k, d)."""
    if not temps:
        return np.zeros(0)
    q = _norm(q.astype(np.float32))
    K, M = len(temps), max(len(t) for t in temps)
    T = np.zeros((K, M, q.shape[1]), np.float32)
    lens = np.array([len(t) for t in temps])
    for k, t in enumerate(temps):
        T[k, : len(t)] = _norm(t.astype(np.float32))
    cost = 1.0 - np.einsum("nd,kmd->knm", q, T)  # (K, n, M)
    cost = np.where(np.arange(M)[None, None, :] >= lens[:, None, None], np.inf, cost)
    n = q.shape[0]
    D = np.full((K, n + 1, M + 1), np.inf, np.float32)
    D[:, 0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, M + 1):
            D[:, i, j] = cost[:, i - 1, j - 1] + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[np.arange(K), n, lens] / (n + lens)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


@dataclass
class Calibration:
    w_phrase: float = 1.0      # weight on phrase log-probs
    w_template: float = 1.0    # weight on template log-probs
    t_phrase: float = 0.3      # softmax temperature for phrase scores
    t_template: float = 0.02   # softmax temperature for DTW distances
    k: int = 3                 # templates averaged per command
    p_min: float = 0.6         # accept threshold on fused probability
    margin: float = 0.2        # accept threshold on (top - second) probability

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Result:
    command: str | None              # None = rejected
    best: str
    prob: float
    margin: float
    probs: dict[str, float] = field(default_factory=dict)


class Matcher:
    def __init__(self, ids: list[str], calib: Calibration | None = None):
        self.ids = ids
        self.cal = calib or Calibration()

    def template_scores(self, emb: np.ndarray, templates: dict[str, list[np.ndarray]]) -> np.ndarray | None:
        """Mean of the k smallest DTW distances per command (lower = closer). None if no templates."""
        if not any(templates.get(i) for i in self.ids):
            return None
        out = np.full(len(self.ids), np.nan)
        for c, cid in enumerate(self.ids):
            ts = templates.get(cid) or []
            if ts:
                d = np.sort(dtw_batch(emb, ts))
                out[c] = d[: self.cal.k].mean()
        out[np.isnan(out)] = np.nanmax(out) + 1.0  # unenrolled commands rank last
        return out

    def fuse(self, phrase: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
        c, lp = self.cal, np.zeros(len(self.ids))
        if phrase is not None and c.w_phrase:
            lp += c.w_phrase * _log_softmax(phrase / c.t_phrase)
        if dist is not None and c.w_template:
            lp += c.w_template * _log_softmax(-dist / c.t_template)
        return np.exp(_log_softmax(lp))

    def decide(self, probs: np.ndarray) -> Result:
        order = np.argsort(-probs)
        top = float(probs[order[0]])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
        best = self.ids[order[0]]
        ok = top >= self.cal.p_min and (top - second) >= self.cal.margin
        return Result(best if ok else None, best, top, top - second,
                      {self.ids[i]: float(probs[i]) for i in order[:5]})
