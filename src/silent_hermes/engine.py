"""Glue: clip -> embedding -> fused command decision."""
from __future__ import annotations

import numpy as np

from .matcher import Calibration, Matcher, Result
from .store import NEGATIVE, Command, Store
from .vsr import VSRModel


class Recogniser:
    def __init__(self, model: VSRModel, commands: list[Command], store: Store, calib: dict | None = None):
        self.model, self.commands, self.store = model, commands, store
        self.phrases = [c.phrase for c in commands]
        self.reload(calib)

    def reload(self, calib: dict | None = None):
        ids = [c.id for c in self.commands]
        self.templates = self.store.templates(ids + [NEGATIVE])
        self.use_negative = bool(self.templates.get(NEGATIVE))
        self.ids = ids + ([NEGATIVE] if self.use_negative else [])
        self.matcher = Matcher(self.ids, Calibration.from_dict(calib or {}))

    def features(self, emb) -> tuple[np.ndarray, np.ndarray | None]:
        phrase = self.model.score_phrases(emb, self.phrases)
        if self.use_negative:  # "not a command" has no phrase; give it a neutral phrase score
            phrase = np.append(phrase, np.median(phrase))
        dist = self.matcher.template_scores(emb.cpu().numpy(), self.templates)
        return phrase, dist

    def recognise(self, crops: np.ndarray) -> tuple[Result, object]:
        emb = self.model.embed(crops)
        phrase, dist = self.features(emb)
        res = self.matcher.decide(self.matcher.fuse(phrase, dist))
        if res.command == NEGATIVE:
            res.command = None
        return res, emb
