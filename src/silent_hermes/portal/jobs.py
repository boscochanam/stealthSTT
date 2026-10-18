"""Upload processing: read-aloud script video -> STT -> phrase clips -> enrolment store.

One mode. Bosco reads the script (script.py) out loud and uploads the video. Per upload:
  normalise (rotation, 25 fps) -> speech-to-text with word timestamps -> find every command
  phrase and filler sentence -> cut the mouth clip for each -> add it to data/enrol/<id>/
  (fillers go to `_none`). Deleting an upload removes the clips it added.
One worker, one job at a time (RAM: VSR ~2.2 GB; whisper is loaded per job and freed).
Videos, full-length crops and audio are deleted after processing; only the cut mouth clips stay.
"""
from __future__ import annotations

import gc
import json
import queue
import shutil
import subprocess
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..script import FILLERS, find_phrases, make_script
from ..store import NEGATIVE, Store, load_commands, load_settings

PAD_FRAMES = 4  # ~0.16 s of context either side of the spoken phrase (STT timing is +-0.1-0.2 s)
FPS = 25


class Portal:
    def __init__(self, settings_path: Path | None = None, commands_path: Path | None = None):
        self.s = load_settings(settings_path)
        self.commands_path = commands_path
        self.root = self.s.data_dir / "portal"
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.s.data_dir)
        cfg = self.s.extra.get("portal", {}) or {}
        self.whisper_model = cfg.get("whisper_model", "small.en")
        self.keep_video = bool(cfg.get("keep_video", False))
        self.idle_unload = int(cfg.get("idle_unload_seconds", 300))
        self.reps = int(cfg.get("script_reps", 10))
        self.q: queue.Queue[str] = queue.Queue()
        self.busy = False
        self._vsr = self._det = None
        self._last_used = 0.0
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._reaper, daemon=True).start()
        for d in sorted(self.root.iterdir()):  # resume anything left mid-way by a restart
            m = self._read(d.name)
            if m and m.get("status") in ("queued", "processing") and any(d.glob("upload.*")):
                self.q.put(d.name)

    # ---------- meta ----------
    def _read(self, uid: str) -> dict | None:
        p = self.root / uid / "meta.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _write(self, uid: str, meta: dict):
        p = self.root / uid / "meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(p)

    def update(self, uid: str, **kw) -> dict:
        with self._lock:
            m = self._read(uid) or {}
            m.update(kw)
            self._write(uid, m)
            return m

    def list(self) -> list[dict]:
        return [m for d in sorted(self.root.iterdir(), reverse=True) if (m := self._read(d.name))]

    def commands(self):
        return load_commands(self.commands_path)

    def script(self):
        return make_script(self.commands(), self.reps)

    # ---------- intake ----------
    def new_upload(self, uid: str, ext: str) -> Path:
        d = self.root / uid
        d.mkdir()
        self._write(uid, {"id": uid, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "uploading"})
        return d / f"upload{ext}"

    def enqueue(self, uid: str):
        self.update(uid, status="queued")
        self.q.put(uid)

    def delete(self, uid: str) -> int:
        """Remove the upload and every clip it added to the enrolment store."""
        d = self.root / uid
        if not (d.is_dir() and d.parent == self.root):
            return 0
        removed = 0
        for cid, name in (self._read(uid) or {}).get("enrolled_files", []):
            p = self.store.root / cid / name
            if p.parent.parent == self.store.root and p.exists():
                p.unlink()
                removed += 1
        shutil.rmtree(d)
        return removed

    # ---------- models ----------
    def _models(self):
        from ..roi import FaceKeypoints
        from ..vsr import VSRModel

        if self._vsr is None:
            self._vsr = VSRModel(self.s.model_dir, self.s.lm_dir, threads=self.s.threads)
            self._det = FaceKeypoints(self.s.face_model)
        self._last_used = time.monotonic()
        return self._vsr, self._det

    def _reaper(self):
        while True:
            time.sleep(30)
            if self._vsr is not None and not self.busy and self.q.empty() \
                    and time.monotonic() - self._last_used > self.idle_unload:
                self._vsr = self._det = None
                gc.collect()

    def _stt(self, pcm: Path) -> list[dict]:
        from faster_whisper import WhisperModel

        m = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
        try:
            # feed samples, not a path: faster-whisper's PyAV loader breaks on newer av releases
            audio = np.fromfile(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            segs, _ = m.transcribe(audio, language="en", word_timestamps=True, vad_filter=True,
                                   condition_on_previous_text=False)
            return [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
                    for sg in segs for w in (sg.words or [])]
        finally:
            del m
            gc.collect()

    # ---------- processing ----------
    def _worker(self):
        while True:
            uid = self.q.get()
            self.busy = True
            try:
                self._process(uid)
            except Exception as e:  # keep the worker alive; show the error on the upload
                self.update(uid, status="error", stage=None, error=f"{type(e).__name__}: {e}",
                            trace=traceback.format_exc()[-1500:])
            finally:
                self.busy = False
                for f in (self.root / uid).glob("*"):
                    if f.name.startswith(("norm", "audio")) or (f.name.startswith("upload") and not self.keep_video):
                        f.unlink()

    def _process(self, uid: str):
        d = self.root / uid
        src = next(d.glob("upload.*"))
        self.update(uid, status="processing", stage="preparing video")
        vid, pcm = d / "norm.mp4", d / "audio.s16"
        # autorotate (iPhone portrait), 25 fps, 480 px tall; audio 16 kHz mono PCM
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-an",
                        "-vf", f"fps={FPS},scale=-2:480", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        str(vid)], check=True, timeout=1800)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn",
                        "-ac", "1", "-ar", "16000", "-f", "s16le", str(pcm)], timeout=1800)
        if not pcm.exists() or pcm.stat().st_size < 16000 * 2:
            raise RuntimeError("no sound in the video: read the script out loud")
        dur = pcm.stat().st_size / 32000
        self.update(uid, duration=round(dur, 1), stage=f"speech-to-text ({dur / 60:.1f} min of audio)")

        words = self._stt(pcm)
        cmds = self.commands()
        targets = [(c.id, c.phrase) for c in cmds] + [(NEGATIVE, f) for f in FILLERS]
        hits = find_phrases(words, targets)
        self.update(uid, transcript=" ".join(w["w"] for w in words), stage=f"found {len(hits)} lines; finding your face")
        if not hits:
            raise RuntimeError("none of the script lines were heard")

        vsr, det = self._models()
        from ..roi import crops_from_video

        crops, face_rate = crops_from_video(vid, det)
        if crops is None:
            raise RuntimeError("no face found in the video")
        quality = self._quality(vid, det)
        self.update(uid, stage="cutting clips")

        phrases = [c.phrase for c in cmds]
        idx = {c.id: i for i, c in enumerate(cmds)}
        found, readable, enrolled, skipped = defaultdict(int), defaultdict(int), [], 0
        for h in hits:
            a = max(0, int(h.start * FPS) - PAD_FRAMES)
            b = min(len(crops), int(np.ceil(h.end * FPS)) + PAD_FRAMES)
            if b - a < 10:
                skipped += 1
                continue
            clip = crops[a:b]
            emb = vsr.embed(clip)
            if h.id != NEGATIVE:  # zero-shot check: would the model alone pick this phrase?
                readable[h.id] += int(np.argmax(vsr.score_phrases(emb, phrases)) == idx[h.id])
            p = self.store.add(h.id, clip, emb.cpu().numpy())
            enrolled.append([h.id, p.name])
            found[h.id] += 1
            self.update(uid, enrolled_files=enrolled)

        summary = [{"id": c.id, "phrase": c.phrase, "found": found[c.id], "readable": readable[c.id]} for c in cmds]
        summary.append({"id": NEGATIVE, "phrase": "not a command (filler lines)", "found": found[NEGATIVE], "readable": None})
        self.update(uid, status="done", stage=None, face_rate=round(face_rate, 3), quality=quality,
                    lines_heard=len(hits), clips_kept=len(enrolled), too_short=skipped, summary=summary)

    def _quality(self, vid: Path, det) -> dict:
        import cv2

        cap = cv2.VideoCapture(str(vid))
        bright, eyes, n, i = [], [], 0, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % 25 == 0:  # one frame per second
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                bright.append(float(rgb.mean()))
                k = det(rgb)
                if k is not None:
                    eyes.append(float(np.linalg.norm(k[0] - k[1])))
                n += 1
            i += 1
        cap.release()
        tips = []
        b = float(np.mean(bright)) if bright else 0
        e = float(np.median(eyes)) if eyes else 0
        if b < 60:
            tips.append("too dark: face a window or lamp")
        if eyes and e < 60:
            tips.append("face too small: hold the phone closer")
        if n and len(eyes) / n < 0.9:
            tips.append("face lost in some frames: keep your whole face in view")
        return {"brightness": round(b), "eye_px": round(e), "tips": tips}
