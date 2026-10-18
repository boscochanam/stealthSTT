"""Run a recognised command through Hermes.

Action types (config/commands.yaml):
  hermes   {prompt}            -> `hermes -z "<prompt>"`, prints the reply
  send     {target, text}      -> `hermes send -t <target> "<text>"` (no LLM)
  dictate  {prompt_template}   -> lip-read free text, you pick/correct it, then `hermes -z` the filled template
Commands marked `confirm: true`, and every dictation, wait for an explicit "y" before anything runs.
"""
from __future__ import annotations

import shlex
import subprocess

from .store import Command, Settings


def build_argv(cmd: Command, settings: Settings, text: str | None = None) -> list[str]:
    a, kind = cmd.action, cmd.action.get("type", "hermes")
    if kind == "hermes":
        return [settings.hermes_bin, "-z", a["prompt"]]
    if kind == "send":
        return [settings.hermes_bin, "send", "-t", a["target"], a["text"]]
    if kind == "dictate":
        if not text:
            raise ValueError("dictate needs text")
        return [settings.hermes_bin, "-z", a.get("prompt_template", "{text}").format(text=text)]
    raise ValueError(f"unknown action type {kind!r} for {cmd.id}")


def run(argv: list[str], settings: Settings, dry_run: bool = False) -> int:
    print("  $ " + shlex.join(argv))
    if dry_run:
        return 0
    p = subprocess.run(argv, text=True, capture_output=True, timeout=settings.hermes_timeout)
    out = (p.stdout or "").strip()
    if out:
        print(out)
    if p.returncode:
        print(f"  [hermes exited {p.returncode}] {(p.stderr or '').strip()[-500:]}")
    return p.returncode


def ask_yes(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False
