"""Reading script + finding its phrases in a speech-to-text transcript.

Bosco reads the script out loud on video. Speech-to-text (word timestamps) is searched for each
command phrase and each filler sentence; every hit becomes a labelled mouth clip. Order does
not matter and the script can be split over several videos: matching is by content.
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

from .store import NEGATIVE, Command

# Everyday sentences with varied mouth shapes; read as "not a command" examples.
FILLERS = [
    "the tea is getting cold on the table",
    "i will call you back after lunch",
    "my phone battery is almost empty",
    "that movie was longer than i expected",
    "we should leave before the traffic starts",
    "can you pass me the water bottle",
    "the meeting moved to thursday morning",
    "i forgot to buy milk again",
    "this chair is really comfortable",
    "it looks like it might rain later",
    "the train was fifteen minutes late",
    "let me think about it for a while",
]

_TOK = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return _TOK.findall(t)


@dataclass
class Line:
    n: int
    id: str
    text: str


def make_script(commands: list[Command], reps: int = 10, fillers: list[str] = FILLERS, seed: int = 7) -> list[Line]:
    items = [(c.id, c.phrase) for c in commands for _ in range(reps)] + [(NEGATIVE, f) for f in fillers]
    random.Random(seed).shuffle(items)
    return [Line(i + 1, cid, text) for i, (cid, text) in enumerate(items)]


def _edits(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


@dataclass
class Hit:
    id: str
    text: str      # what was actually heard
    start: float   # seconds
    end: float
    errors: int


def find_phrases(words: list[dict], targets: list[tuple[str, str]]) -> list[Hit]:
    """words: STT words [{"w", "s", "e"}]; targets: (id, phrase). Non-overlapping best matches.

    A phrase of L words may be heard with up to L//4 word errors (so 4+ word phrases tolerate one
    STT slip; short phrases must match exactly).
    """
    flat = []  # (token, start, end)
    for w in words:
        for t in tokens(w["w"]):
            flat.append((t, w["s"], w["e"]))
    toks = [f[0] for f in flat]
    cands = []
    for cid, phrase in targets:
        p = tokens(phrase)
        L, tol = len(p), len(p) // 4
        for i in range(len(toks)):
            for n in range(max(1, L - tol), L + tol + 1):
                if i + n > len(toks):
                    break
                e = _edits(p, toks[i:i + n])
                if e <= tol:
                    cands.append((e, -n, i, i + n, cid))
    cands.sort()
    used = [False] * len(toks)
    hits = []
    for e, _, i, j, cid in cands:
        if any(used[i:j]):
            continue
        used[i:j] = [True] * (j - i)
        hits.append(Hit(cid, " ".join(toks[i:j]), flat[i][1], flat[j - 1][2], e))
    return sorted(hits, key=lambda h: h.start)
