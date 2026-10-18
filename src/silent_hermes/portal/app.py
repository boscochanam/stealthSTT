"""Recording portal: show the reading script, take read-aloud videos, report what was kept.

Tailnet only: binds to the host's Tailscale IP (or 127.0.0.1), no public exposure.
  silent-hermes portal [--host 100.x.y.z] [--port 8810]
"""
from __future__ import annotations

import re
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .jobs import Portal

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 800 * 2**20
_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_EXT = {".mov", ".mp4", ".m4v", ".webm", ".mpg", ".avi", ".mkv", ".3gp"}


def create_app(portal: Portal | None = None) -> FastAPI:
    app = FastAPI(title="silent-hermes portal", docs_url=None, redoc_url=None)
    P = portal or Portal()

    def _uid(uid: str) -> str:
        if not _ID.match(uid) or not (P.root / uid).is_dir():
            raise HTTPException(404, "no such upload")
        return uid

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/script")
    def script():
        return [{"n": l.n, "id": l.id, "text": l.text} for l in P.script()]

    @app.get("/api/status")
    def status():
        return {"queue": P.q.qsize() + int(P.busy), "enrolled": P.store.counts(), "target": P.reps,
                "commands": [{"id": c.id, "phrase": c.phrase} for c in P.commands()],
                "calibrated": (P.s.data_dir / "calibration.json").exists()}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in _EXT:
            ext = ".mov"
        uid = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        dest = P.new_upload(uid, ext)
        size = 0
        with dest.open("wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    f.close()
                    P.delete(uid)
                    raise HTTPException(413, "video over 800 MB: record at 1080p or split the script")
                f.write(chunk)
        P.update(uid, bytes=size)
        P.enqueue(uid)
        return {"id": uid}

    @app.get("/api/uploads")
    def uploads():
        keep = ("id", "created", "status", "stage", "error", "duration", "lines_heard", "clips_kept",
                "too_short", "summary", "quality", "face_rate")
        return [{k: m.get(k) for k in keep} for m in P.list()[:50]]

    @app.delete("/api/uploads/{uid}")
    def delete(uid: str):
        uid = _uid(uid)
        if (P._read(uid) or {}).get("status") in ("queued", "processing"):
            raise HTTPException(409, "still processing")
        return {"clips_removed": P.delete(uid)}

    return app


def serve(host: str, port: int, idle_minutes: float = 30, settings_path: Path | None = None):
    """Run until stopped, or until `idle_minutes` pass with no requests and no job running."""
    import os
    import threading

    import uvicorn
    from fastapi import Request

    P = Portal(settings_path)
    app = create_app(P)
    last = [time.monotonic()]

    @app.middleware("http")
    async def touch(request: Request, call_next):
        last[0] = time.monotonic()
        return await call_next(request)

    def watchdog():
        while True:
            time.sleep(30)
            if idle_minutes and not P.busy and P.q.empty() and time.monotonic() - last[0] > idle_minutes * 60:
                print(f"idle {idle_minutes:g} min: stopping", flush=True)
                os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
