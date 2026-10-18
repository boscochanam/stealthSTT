"""Enrolment store and config loading.

data/enrol/<command_id>/<timestamp>.npz holds `crops` (T,96,96 uint8 mouth crops) and
`emb` (T,768 float16 encoder frames). Mouth crops only, never full frames. The data dir
is created mode 0700 and is git-ignored.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
NEGATIVE = "_none"  # enrolment class for "not a command" (idle, chatting, yawning)


@dataclass
class Command:
    id: str
    phrase: str
    action: dict
    confirm: bool = False


@dataclass
class Settings:
    model_dir: Path
    lm_dir: Path
    face_model: Path
    data_dir: Path
    camera: int | str = 0
    clip_seconds: float = 2.5
    threads: int | None = None
    hermes_bin: str = "hermes"
    hermes_timeout: int = 300
    extra: dict = field(default_factory=dict)


def _abs(p: str) -> Path:
    p = Path(os.path.expanduser(p))
    return p if p.is_absolute() else ROOT / p


def load_settings(path: Path | None = None) -> Settings:
    d = yaml.safe_load((path or ROOT / "config/settings.yaml").read_text())
    return Settings(
        model_dir=_abs(d["model_dir"]), lm_dir=_abs(d["lm_dir"]), face_model=_abs(d["face_model"]),
        data_dir=_abs(d.get("data_dir", "data")), camera=d.get("camera", 0),
        clip_seconds=float(d.get("clip_seconds", 2.5)), threads=d.get("threads"),
        hermes_bin=d.get("hermes_bin", "hermes"), hermes_timeout=int(d.get("hermes_timeout", 300)),
        extra=d,
    )


def load_commands(path: Path | None = None) -> list[Command]:
    d = yaml.safe_load((path or ROOT / "config/commands.yaml").read_text())
    cmds = [Command(c["id"], c["phrase"], c.get("action", {}), bool(c.get("confirm", False))) for c in d["commands"]]
    ids = [c.id for c in cmds]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate command ids in commands.yaml")
    if NEGATIVE in ids:
        raise ValueError(f"{NEGATIVE} is reserved")
    return cmds


class Store:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "enrol"
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(data_dir, 0o700)

    def add(self, cid: str, crops: np.ndarray, emb: np.ndarray) -> Path:
        d = self.root / cid
        d.mkdir(exist_ok=True)
        p = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**6:06d}.npz"
        np.savez_compressed(p, crops=crops, emb=emb.astype(np.float16))
        return p

    def items(self, cid: str) -> list[Path]:
        d = self.root / cid
        return sorted(d.glob("*.npz")) if d.exists() else []

    def counts(self) -> dict[str, int]:
        return {d.name: len(list(d.glob("*.npz"))) for d in sorted(self.root.iterdir()) if d.is_dir()}

    def templates(self, ids: list[str]) -> dict[str, list[np.ndarray]]:
        return {cid: [np.load(p)["emb"].astype(np.float32) for p in self.items(cid)] for cid in ids}

    def drop_last(self, cid: str) -> Path | None:
        items = self.items(cid)
        if items:
            items[-1].unlink()
            return items[-1]
        return None


def load_calibration(data_dir: Path) -> dict:
    p = data_dir / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}
