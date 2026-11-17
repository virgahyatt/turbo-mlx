"""Finding and downloading the catalog's checkpoints in the Hugging Face cache (HF_HOME, by default
~/.cache/huggingface, the same place the app kept them)."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi, constants
from huggingface_hub.utils import filter_repo_objects

from .catalog import Model

log = logging.getLogger("turbo-mlx-mcp.models")


def _snapshot_dir(repo: str, revision: str) -> Path:
    folder = "models--" + repo.replace("/", "--")
    return Path(constants.HF_HUB_CACHE) / folder / "snapshots" / revision


def _has_files(folder: Path, patterns: list[str] | tuple[str, ...]) -> bool:
    """Every literal file present, and every weights glob matching at least one file."""
    if not folder.is_dir():
        return False
    files: list[str] | None = None
    for pattern in patterns:
        if not any(ch in pattern for ch in "*?["):
            if not (folder / pattern).exists() and pattern.endswith((".safetensors", "config.json", ".json")):
                # Optional extras (README, LICENSE) may be missing upstream; weights and configs may not.
                if pattern not in ("README.md", "LICENSE", "LICENSE.md", "NOTICE", "NOTICE.md", "quantize_config.json",
                                   "split_model.json", "tokenizer_config.json", "added_tokens.json"):
                    return False
        elif pattern.endswith(".safetensors"):
            if files is None:
                files = [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() or p.is_symlink()]
            if not any(fnmatch.fnmatch(f, pattern) for f in files):
                return False
    return True


def model_path(model: Model) -> Path | None:
    folder = _snapshot_dir(model.repo, model.revision)
    return folder if _has_files(folder, model.download_patterns) else None


def companion_path(model: Model) -> Path | None:
    c = model.companion
    if c is None:
        return None
    folder = _snapshot_dir(c.repo, c.revision)
    return folder if _has_files(folder, c.patterns) else None


def is_installed(model: Model, quality: bool = False) -> bool:
    path = model_path(model)
    if path is None or (model.companion and companion_path(model) is None):
        return False
    return not quality or all((path / f).exists() for f in model.quality_files)


def _repo_dir(repo: str) -> Path:
    return Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _finished_bytes(repo: str, revision: str, patterns: list[str]) -> int:
    """Bytes of the wanted files already in the snapshot. Snapshot entries are links into the blob store (the
    repo's blobs/ or, in newer huggingface_hub, a shared hub/blobs/xx/ store), so they are followed."""
    folder = _snapshot_dir(repo, revision)
    if not folder.is_dir():
        return 0
    names = [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_symlink() or p.is_file()]
    total = 0
    for name in filter_repo_objects(names, allow_patterns=patterns):
        with contextlib.suppress(OSError):
            total += (folder / name).stat().st_size
    return total


def _partial_bytes(since: float, repos: list[str]) -> int:
    """Bytes of files being written (*.incomplete) since `since`, in the repos' blobs and the shared store."""
    hub = Path(constants.HF_HUB_CACHE)
    dirs = [_repo_dir(r) / "blobs" for r in repos]
    with contextlib.suppress(FileNotFoundError):
        dirs += [Path(e.path) for e in os.scandir(hub / "blobs") if e.is_dir()]
    total = 0
    for d in dirs:
        with contextlib.suppress(FileNotFoundError):
            for e in os.scandir(d):
                if e.name.endswith(".incomplete"):
                    with contextlib.suppress(OSError):
                        st = e.stat()
                        if st.st_mtime >= since - 60:
                            total += st.st_size
    return total


def _remote_bytes(repo: str, revision: str, patterns: list[str]) -> int | None:
    """Exact size of the files a download fetches, from the Hub's metadata."""
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True,
                                  token=os.environ.get("HF_TOKEN") or None)
    except Exception:  # noqa: BLE001 - offline or gated: fall back to the catalog's size
        return None
    files = {s.rfilename: s.size or 0 for s in info.siblings or []}
    return sum(files[name] for name in filter_repo_objects(list(files), allow_patterns=patterns))


@dataclass
class Download:
    model_id: str
    status: str = "running"  # running, done, failed, cancelled
    step: str = "starting"
    error: str | None = None
    started: float = 0.0
    finished: float | None = None
    total_bytes: int = 0
    repos: list[str] = field(default_factory=list)
    jobs: list[tuple[str, str, list[str]]] = field(default_factory=list)
    done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _samples: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)

    def downloaded_bytes(self) -> int:
        finished = sum(_finished_bytes(*j) for j in self.jobs)
        return finished + (_partial_bytes(self.started, self.repos) if self.status == "running" else 0)

    def summary(self) -> dict:
        have = self.downloaded_bytes()
        now = time.time()
        out: dict = {"model": self.model_id, "status": self.status}
        if self.status == "running":
            out["step"] = self.step
        if self.total_bytes:
            have = min(have, self.total_bytes)
            out["progress"] = f"{have / 1e9:.1f} of {self.total_bytes / 1e9:.1f} GB ({100 * have / self.total_bytes:.0f}%)"
        if self.status == "running":
            self._samples.append((now, have))
            t0, b0 = self._samples[0]
            if now - t0 >= 3 and have > b0:
                speed = (have - b0) / (now - t0)
                out["speed"] = f"{speed / 1e6:.0f} MB/s"
                if self.total_bytes > have:
                    out["eta_minutes"] = round((self.total_bytes - have) / speed / 60, 1)
        out["elapsed_minutes"] = round(((self.finished or now) - self.started) / 60, 1)
        if self.error:
            out["error"] = self.error
        return out


class Downloader:
    def __init__(self) -> None:
        self.downloads: dict[str, Download] = {}

    def start(self, model: Model, include_quality: bool = False) -> Download:
        current = self.downloads.get(model.id)
        if current and current.status == "running":
            return current
        d = Download(model_id=model.id, started=time.time(),
                     repos=[model.repo] + ([model.companion.repo] if model.companion else []))
        self.downloads[model.id] = d
        asyncio.create_task(self._run(d, model, include_quality))
        return d

    async def cancel(self, model_id: str) -> Download | None:
        d = self.downloads.get(model_id)
        if d is None or d.status != "running":
            return d
        d.status = "cancelled"  # set first, so _run does not report the killed process as a failure
        if d.proc and d.proc.returncode is None:
            d.proc.terminate()
            try:
                await asyncio.wait_for(d.proc.wait(), 5)
            except asyncio.TimeoutError:
                d.proc.kill()
        # A killed download leaves its file in progress as <hash>.<random>.incomplete, which no later download
        # reuses (the suffix is new each time): remove it rather than leak gigabytes.
        for repo in d.repos:
            with contextlib.suppress(FileNotFoundError):
                for e in os.scandir(_repo_dir(repo) / "blobs"):
                    if e.name.endswith(".incomplete"):
                        with contextlib.suppress(OSError):
                            if e.stat().st_mtime >= d.started - 60:
                                os.remove(e.path)
        d.finished = time.time()
        d.done_event.set()
        return d

    async def wait(self, d: Download, timeout: float, on_tick=None) -> None:
        deadline = time.monotonic() + timeout
        while d.status == "running":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(d.done_event.wait(), min(remaining, 5))
            if on_tick and d.status == "running":
                await on_tick(d)

    async def _run(self, d: Download, model: Model, include_quality: bool) -> None:
        patterns = model.download_patterns + (model.quality_files if include_quality else [])
        jobs = [(model.repo, model.revision, patterns)]
        if c := model.companion:
            jobs.append((c.repo, c.revision, list(c.patterns)))
        d.jobs = jobs
        try:
            d.step = "checking file sizes"
            sizes = await asyncio.gather(*(asyncio.to_thread(_remote_bytes, *j) for j in jobs))
            d.total_bytes = (sum(sizes) if all(s is not None for s in sizes)
                             else int(model.size_gb * 1e9))
            for repo, revision, pats in jobs:
                if d.status != "running":
                    return
                d.step = f"downloading {repo}"
                d.proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "turbo_mlx_mcp.download_worker", repo, revision, json.dumps(pats),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
                )
                _, err = await d.proc.communicate()
                if d.status != "running":
                    return
                if d.proc.returncode != 0:
                    tail = err.decode(errors="replace").strip().splitlines()[-1:] or ["unknown error"]
                    raise RuntimeError(tail[0])
            if is_installed(model, include_quality):
                d.status = "done"
            else:
                d.status, d.error = "failed", "download finished but some expected files are missing"
        except Exception as e:  # noqa: BLE001
            if d.status != "running":
                return
            log.exception("download %s", model.id)
            d.status, d.error = "failed", str(e)
            if any(x in str(e) for x in ("401", "403", "Gated", "gated")):
                d.error += " (gated repo: accept its licence on huggingface.co and set HF_TOKEN for the server)"
        finally:
            if d.finished is None:
                d.finished = time.time()
            d.done_event.set()
