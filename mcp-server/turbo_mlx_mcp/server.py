"""MCP server for turbo-engine: image, video, song, sound-effect, upscale and image-to-3D generation on
Apple silicon, served over Streamable HTTP so other machines on the local network can use it."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import hmac
import json
import logging
import os
import random
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent

from . import catalog, models
from .engine import TERMINAL, EngineManager, Job

log = logging.getLogger("turbo-mlx-mcp")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXT = {catalog.IMAGE: "png", catalog.VIDEO: "mp4", catalog.AUDIO: "wav", catalog.MODEL3D: "glb"}
MAX_INPUT_BYTES = 64 * 1024 * 1024


class Config:
    engine = os.environ.get("TURBO_ENGINE", str(REPO_ROOT / "build/bin/turbo-engine"))
    output_dir = Path(os.environ.get("TURBO_OUTPUT_DIR", str(Path.home() / "TurboMLX" / "outputs"))).expanduser()
    token = os.environ.get("TURBO_MCP_TOKEN") or None
    public_url = os.environ.get("TURBO_PUBLIC_URL") or None
    default_wait = float(os.environ.get("TURBO_WAIT_SECONDS", "600"))
    allow_local_paths = os.environ.get("TURBO_ALLOW_LOCAL_PATHS") == "1"
    # The app's "Save memory" default: on below 64 GB.
    low_memory = os.environ.get("TURBO_LOW_MEMORY", "auto")


cfg = Config()
engine = EngineManager(cfg.engine)  # one engine process for every session, started on first use
downloader = models.Downloader()


mcp = MCPServer(
    name="turbo-mlx",
    title="Turbo MLX",
    version="0.1.0",
    instructions=(
        "Local generative media on an Apple-silicon Mac (MLX). Call list_models first: a model must be downloaded "
        "(ask the user, then download_model with confirm=true) before it can generate. Jobs run one at a time on the GPU. Long jobs "
        "(video, songs, big models) may outlast the call: the tool then returns a job_id; poll get_job(job_id). "
        "A finished generation returns just its link (get_job with details=true adds the seed, size and timings). To change an existing picture use "
        "edit_image. Input pictures may be URLs of earlier results, http(s) URLs, data: URIs or base64; a picture "
        "pasted into the chat must first be uploaded from its file: POST the bytes to /upload on this server with the "
        "same Authorization header (curl --data-binary @file.png -H 'Authorization: Bearer ...' <server>/upload), "
        "which returns {\"url\": ...}."
    ),
)


# ---------------------------------------------------------------- helpers

def _lan_ip() -> str:
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outward interface
        return s.getsockname()[0]
    return socket.gethostname()


def _file_url(path: str | Path) -> str:
    return f"{cfg.public_url.rstrip('/')}/files/{Path(path).name}"


def _upload_command(file: str = "/path/to/picture.png") -> str:
    """The exact command a client runs to upload a local picture to this server (its own address, so clients
    never need it hard-coded). The token is the client's copy of TURBO_MCP_TOKEN."""
    auth = ' -H "Authorization: Bearer $(cat ~/.config/turbo-mlx-mcp/token)"' if cfg.token else ""
    return f'curl -s --data-binary @"{file}"{auth} {cfg.public_url.rstrip("/")}/upload'


def _not_uploaded(ref: str) -> ValueError:
    file = ref[1:] if ref.startswith("@") else ref  # OpenCode's file-mention marker is not part of the name
    return ValueError(f"{ref!r} is a file on your machine, which this server cannot read. Upload it first with: "
                      f"{_upload_command(file)}  -- then pass the \"url\" it returns.")


def _low_memory() -> bool:
    if cfg.low_memory in ("1", "true", "yes"):
        return True
    if cfg.low_memory in ("0", "false", "no"):
        return False
    return (engine.info or {}).get("memory", os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) < 64 << 30


def _resolve_model(model_id: str | None, media: str, *, family: str | None = None,
                   edits: bool = False) -> catalog.Model:
    if model_id:
        m = catalog.find(model_id)
        if m.media != media:
            raise ValueError(f"{m.id} makes {m.media}, not {media}")
        if edits and not m.takes_reference_image:
            raise ValueError(f"{m.id} cannot edit pictures")
        return m
    candidates = [m for m in catalog.CATALOG if m.media == media and (family is None or m.family == family)
                  and not m.is_upscaler and (not edits or m.takes_reference_image)]
    installed = [m for m in candidates if models.is_installed(m)]
    if not installed:
        names = ", ".join(m.id for m in candidates)
        what = "image-editing" if edits else media
        raise ValueError(f"no {what} model is downloaded yet. Options: {names}. Ask the user which one to download "
                         "(see list_models for sizes) before calling download_model.")
    return installed[0]


def _require_installed(m: catalog.Model, quality: bool = False) -> None:
    if not models.is_installed(m, quality):
        extra = " with include_quality=true" if quality and models.is_installed(m) else ""
        raise ValueError(f"{m.id} is not downloaded{extra}. It is about {m.size_gb:.0f} GB: ask the user before "
                         f"calling download_model(model_id={m.id!r}, confirm=true"
                         f"{', include_quality=true' if quality else ''}).")


def _snap(value: int, multiple: int) -> int:
    return max(256, min(4096, round(value / multiple) * multiple))


def _check_steps(m: catalog.Model, steps: int | None, quality: bool = False) -> int:
    lo, hi = m.step_range(quality)
    steps = steps or m.default_steps(quality)
    if not lo <= steps <= hi:
        raise ValueError(f"{m.id} takes {lo}-{hi} steps")
    return steps


def _input_image(ref: str) -> str:
    """Turns an http(s) URL, a data: URI, raw base64 or a URL/path of this server's outputs into a local path."""
    ref = ref.strip()
    inputs = cfg.output_dir / "inputs"
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(6).hex()}"
    # An earlier result or upload, by its URL or bare file name: used in place, nothing is fetched.
    if m := re.fullmatch(r"(?:.*/files/)?([A-Za-z0-9][A-Za-z0-9._-]*)", ref):
        path = cfg.output_dir / m.group(1)
        if path.is_file():
            return str(path)
    if ref.startswith(("http://", "https://")):
        req = urllib.request.Request(ref, headers={"User-Agent": "turbo-mlx-mcp"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            data = r.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("input image is larger than 64 MB")
        path = inputs / f"{name}{_guess_ext(data)}"
        path.write_bytes(data)
        return str(path)
    if ref.startswith("data:"):
        ref = ref.split(",", 1)[1]
    p = Path(ref).expanduser()
    if len(ref) < 1024 and (p.is_absolute() or ref.startswith(("~", "./", "../", "@")) or re.fullmatch(r"[^/\s]+\.(png|jpe?g|webp|heic)", ref, re.I)):
        if p.is_absolute() and p.is_file() and (cfg.allow_local_paths or p.resolve().is_relative_to(cfg.output_dir.resolve())):
            return str(p)
        raise _not_uploaded(ref)
    try:
        data = base64.b64decode(ref, validate=True)
    except binascii.Error as e:
        raise ValueError("image must be the URL of an earlier result or of an upload, an http(s) URL, a data: URI or "
                         f"base64. To use a local file, upload it first: {_upload_command()}") from e
    path = inputs / f"{name}{_guess_ext(data)}"
    path.write_bytes(data)
    return str(path)


def _guess_ext(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return ".heic"
    return ".png"


def _model_spec(m: catalog.Model) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "family": m.family, "path": str(models.model_path(m)), "name": m.repo,
        "variant": m.variant or "", "low_ram": _low_memory(),
    }
    if m.companion:
        spec["text_encoder_path"] = str(models.companion_path(m))
    if m.quantize:
        spec["quantize"] = m.quantize
    return spec


def _submit(m: catalog.Model, kind: str, params: dict[str, Any]) -> Job:
    job_id = os.urandom(16).hex()
    params["output"] = str(cfg.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id}.{EXT[m.media]}")
    params.setdefault("prompt", "")
    params.setdefault("width", 0)
    params.setdefault("height", 0)
    params.setdefault("guidance", 1.0)
    if params.get("seed") is None:
        params["seed"] = random.randint(0, 2**31 - 1)
    return engine.submit(model_id=m.id, kind=kind, model=_model_spec(m), params=params, job_id=job_id)


PREVIEW_SIDE = 1024  # longest side of an inline preview: a JPEG of 100-250 KB, well under clients' 1 MB limit


def _preview_jpeg(src: Path) -> bytes | None:
    """A small JPEG of a result (made once with sips, kept in outputs/previews) for include_image."""
    out = cfg.output_dir / "previews" / f"{src.stem}.jpg"
    if not out.is_file():
        out.parent.mkdir(exist_ok=True)
        done = subprocess.run(["/usr/bin/sips", "-Z", str(PREVIEW_SIDE), "-s", "format", "jpeg", "-s", "formatOptions", "82",
                               str(src), "--out", str(out)], capture_output=True)
        if done.returncode != 0 or not out.is_file():
            return None
    return out.read_bytes()


def _job_result(job: Job, details: bool = False, include_image: bool = False) -> CallToolResult:
    """A finished job answers with its link alone; `details` adds the seed, size, timings and peak memory, as
    text (no inline picture: a full-size PNG exceeds the 1 MB message limit of standard MCP clients). Running,
    failed and cancelled jobs answer with their short status."""
    if job.status == "done" and job.result and not details:
        content: list[Any] = [TextContent(type="text", text=_file_url(job.result["path"]))]
        content += _preview_content(job) if include_image else []
        return CallToolResult(content=content)
    info = job.summary()
    content: list[Any] = []
    if job.status == "done" and job.result:
        r = job.result
        info["seed"] = r.get("seed", job.params.get("seed"))
        info["url"] = _file_url(r["path"])
        info["file"] = Path(r["path"]).name
        for key in ("width", "height", "size", "frames", "duration", "sample_rate", "gaussians", "seconds", "peak_memory"):
            if key in r:
                info[key] = r[key]
        if r.get("poster"):
            info["poster_url"] = _file_url(r["poster"])
    elif job.status not in TERMINAL:
        info["queue_position"] = engine.queue_position(job)
        info["next"] = (f"Still running. Call get_job(job_id={job.id!r}); it waits up to 10 minutes by itself, "
                        "so do not sleep or poll quickly.")
    content.insert(0, TextContent(type="text", text=json.dumps(info, indent=1)))
    if include_image and job.status == "done":
        content += _preview_content(job)
    return CallToolResult(content=content, is_error=job.status == "failed")


def _preview_content(job: Job) -> list[Any]:
    """The inline preview of a finished image job (a video's first frame); nothing for sound or 3D."""
    r = job.result or {}
    image = r.get("path") if str(r.get("path", "")).endswith(".png") else r.get("poster")
    data = _preview_jpeg(Path(image)) if image and Path(image).is_file() else None
    if data is None:
        return []
    return [ImageContent(type="image", mime_type="image/jpeg", data=base64.b64encode(data).decode())]


async def _run(job: Job, ctx: Context | None, wait_seconds: float | None, details: bool = False,
               include_image: bool = False) -> CallToolResult:
    last = [0.0]

    async def report(j: Job) -> None:
        if ctx is None or time.monotonic() - last[0] < 2:
            return
        last[0] = time.monotonic()
        msg = j.phase or j.status
        with contextlib.suppress(Exception):
            if j.step is not None and j.total:
                await ctx.report_progress(j.step, j.total, msg)
            else:
                await ctx.report_progress(0, None, msg)

    await engine.wait(job, cfg.default_wait if wait_seconds is None else wait_seconds, report)
    return await asyncio.to_thread(_job_result, job, details, include_image)


def _error(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


WaitArg = Annotated[float | None, Field(description="Seconds to wait for the result before returning a job_id to poll "
                                                    "(default 600; 0 returns at once).", ge=0, le=3600)]
IncludeImageArg = Annotated[bool, Field(description="Also return a small preview of the result inline (a JPEG, at most "
                                                   f"{PREVIEW_SIDE} px; a video's first frame). Default: the link only.")]
SeedArg = Annotated[int | None, Field(description="Random seed; omit for a random one.", ge=0)]
ImageRef = Annotated[str, Field(description="Usually the URL (or file name) of an earlier result or upload from this server, used in place: never download and re-upload it. Also any http(s) URL, a data: URI or base64.")]


# ---------------------------------------------------------------- tools: models

@mcp.tool()
async def list_models(media: Annotated[Literal["image", "video", "audio", "3d"] | None,
                                       Field(description="Only models making this.")] = None) -> CallToolResult:
    """Lists the models this server can run, whether each is downloaded, and its default settings."""
    with contextlib.suppress(Exception):
        await engine.ensure_ready()
    rows = []
    for m in catalog.CATALOG:
        if media and m.media != media:
            continue
        d = downloader.downloads.get(m.id)
        rows.append({
            "id": m.id, "name": m.name, "media": m.media, "family": m.family, "detail": m.detail,
            "downloaded": models.is_installed(m),
            "download": d.summary() if d and d.status == "running" else None,
            "download_size_gb": m.size_gb, "recommended_ram_gb": m.memory_gb, "license": m.license,
            "default_steps": m.default_steps(), "step_range": m.step_range(),
            "default_guidance": m.default_guidance() if m.supports_guidance else None,
            "negative_prompt": m.supports_negative_prompt, "reference_images": m.takes_reference_image,
            "loras": m.takes_loras, "transparent_background": m.produces_alpha,
        })
    host = engine.info or {}
    text = json.dumps({"device": host.get("device"), "memory_gb": round(host.get("memory", 0) / 2**30) or None,
                       "upload_local_picture": _upload_command() + "   (returns {\"url\": ...} to pass as image)",
                       "models": rows}, indent=1)
    return CallToolResult(content=[TextContent(type="text", text=text)])


@mcp.tool()
async def download_model(
    model_id: Annotated[str, Field(description="A model id from list_models.")],
    ctx: Context,
    confirm: Annotated[bool, Field(description="Set true only after the user has explicitly agreed to this download "
                                               "and its size. Without it, nothing is downloaded.")] = False,
    include_quality: Annotated[bool, Field(description="LTX video only: also fetch the dev transformer needed by "
                                                       "quality=true (about 11-20 GB more).")] = False,
    wait_seconds: WaitArg = None,
) -> CallToolResult:
    """Downloads a model from Hugging Face (3-38 GB). Ask the user first, then call with confirm=true. The call waits
    up to wait_seconds (default 600) while the download runs, reporting progress; if it is not finished by then,
    call get_download, which waits again. Do not poll in a loop or sleep between calls."""
    try:
        m = catalog.find(model_id)
    except KeyError as e:
        return _error(str(e))
    if models.is_installed(m, include_quality):
        return CallToolResult(content=[TextContent(type="text", text=f"{m.id} is already downloaded.")])
    running = downloader.downloads.get(m.id)
    if not confirm and not (running and running.status == "running"):
        size = m.size_gb + (15 if include_quality else 0)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({
            "model": m.id, "status": "not started", "download_size_gb": round(size, 1),
            "recommended_ram_gb": m.memory_gb,
            "next": f"Ask the user whether to download {m.name} (about {size:.0f} GB). "
                    "Only if they agree, call download_model again with confirm=true.",
        }))])
    d = downloader.start(m, include_quality)
    return await _wait_download(d, ctx, wait_seconds)


async def _wait_download(d: models.Download, ctx: Context | None, wait_seconds: float | None) -> CallToolResult:
    async def tick(dl: models.Download) -> None:
        info = dl.summary()  # also records a speed sample
        if ctx is None:
            return
        have, total = dl.downloaded_bytes(), dl.total_bytes
        detail = ", ".join(str(info[k]) for k in ("progress", "speed") if k in info)
        if "eta_minutes" in info:
            detail += f", about {info['eta_minutes']:.0f} min left"
        with contextlib.suppress(Exception):
            await ctx.report_progress(min(have, total) / 1e9 if total else have / 1e9, total / 1e9 if total else None,
                                      f"Downloading {dl.model_id}: {detail or dl.step}")

    await downloader.wait(d, cfg.default_wait if wait_seconds is None else wait_seconds, tick)
    info = d.summary()
    if d.status == "running":
        info["next"] = (f"Still downloading. Tell the user the progress, then call get_download(model_id={d.model_id!r}); "
                        "it waits up to 10 minutes by itself, so do not sleep or poll quickly.")
    elif d.status == "done":
        info["next"] = "Ready: the model can be used now."
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(info, indent=1))],
                          is_error=d.status == "failed")


@mcp.tool()
async def get_download(
    ctx: Context,
    model_id: Annotated[str | None, Field(description="The model being downloaded; omit to list every download.")] = None,
    wait_seconds: WaitArg = None,
) -> CallToolResult:
    """Waits for a model download to finish (up to wait_seconds, default 600) while reporting progress, then returns
    its status: progress in GB and percent, speed and ETA. Without model_id, lists all downloads at once."""
    if model_id is None:
        running = [d for d in downloader.downloads.values() if d.status == "running"]
        if len(running) != 1:
            rows = [d.summary() for d in downloader.downloads.values()]
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(rows or "no downloads", indent=1))])
        d = running[0]
    else:
        try:
            m = catalog.find(model_id)
        except KeyError as e:
            return _error(str(e))
        d = downloader.downloads.get(m.id)
        if d is None:
            state = "already downloaded" if models.is_installed(m) else "not downloading"
            return CallToolResult(content=[TextContent(type="text", text=f"{m.id}: {state}")])
    return await _wait_download(d, ctx, wait_seconds)


@mcp.tool()
async def cancel_download(model_id: Annotated[str, Field(description="The model being downloaded.")]) -> CallToolResult:
    """Stops a model download. Files already finished are kept; the file in progress is discarded, so a later
    download_model starts that file over."""
    try:
        m = catalog.find(model_id)
    except KeyError as e:
        return _error(str(e))
    d = await downloader.cancel(m.id)
    if d is None:
        return _error(f"{m.id} is not being downloaded")
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(d.summary(), indent=1))])


# ---------------------------------------------------------------- tools: generation

@mcp.tool()
async def generate_image(
    prompt: Annotated[str, Field(description="What to draw. For edit models, what to change in the reference image(s).")],
    ctx: Context,
    model: Annotated[str | None, Field(description="Model id from list_models; default: the first downloaded image model.")] = None,
    width: Annotated[int, Field(ge=256, le=4096)] = 1024,
    height: Annotated[int, Field(ge=256, le=4096)] = 1024,
    steps: Annotated[int | None, Field(description="Default: the model's own.")] = None,
    guidance: Annotated[float | None, Field(description="CFG scale, for models that take it.", ge=1, le=10)] = None,
    negative_prompt: Annotated[str | None, Field(description="For models that take one.")] = None,
    reference_images: Annotated[list[str] | None, Field(description="Pictures to edit / take from (FLUX.2 Klein, FLUX 3, "
                                                                    "Qwen-Image Edit, SenseNova): the URL (or file name) of an earlier result or upload from this server, used in place: never download and re-upload it; or other URLs.")] = None,
    transparent_background: Annotated[bool, Field(description="Ming-Image only: keep the alpha channel.")] = False,
    loras: Annotated[list[dict] | None, Field(description='[{"path": "/abs/path/on/server.safetensors", "scale": 1.0}]')] = None,
    seed: SeedArg = None,
    wait_seconds: WaitArg = None,
    include_image: IncludeImageArg = False,
) -> CallToolResult:
    """Generates an image from a prompt (or edits reference images); returns the PNG's link."""
    try:
        m = _resolve_model(model, catalog.IMAGE, edits=bool(reference_images))
        if m.is_upscaler:
            raise ValueError("use upscale_image for SeedVR2")
        _require_installed(m)
        params: dict[str, Any] = {
            "prompt": prompt, "seed": seed, "steps": _check_steps(m, steps),
            "width": _snap(width, m.size_multiple), "height": _snap(height, m.size_multiple),
            "guidance": guidance if guidance is not None and m.supports_guidance else m.default_guidance(),
            "flatten_alpha": not (transparent_background and m.produces_alpha),
        }
        if negative_prompt and m.supports_negative_prompt:
            params["negative_prompt"] = negative_prompt
        if reference_images:
            if not m.takes_reference_image:
                raise ValueError(f"{m.id} does not take reference images")
            params["images"] = [_input_image(r) for r in reference_images]
        elif m.requires_reference_image:
            raise ValueError(f"{m.id} needs reference_images")
        if loras:
            if not m.takes_loras:
                raise ValueError(f"{m.id} does not take LoRAs")
            params["loras"] = [{"path": str(Path(lo["path"]).expanduser()), "scale": float(lo.get("scale", 1))} for lo in loras]
    except (ValueError, KeyError, OSError) as e:
        return _error(str(e))
    return await _run(_submit(m, "image", params), ctx, wait_seconds, include_image=include_image)


@mcp.tool()
async def edit_image(
    image: Annotated[str, Field(description="The picture to edit: the URL (or file name) of an earlier result or upload from this server, used in place: never download and re-upload it. Also any http(s) URL.")],
    instruction: Annotated[str, Field(description="What to change, e.g. 'make both people blond'. The rest is kept.")],
    ctx: Context,
    model: Annotated[str | None, Field(description="Default: the first downloaded model that edits (flux2-klein-4b, "
                                                   "flux3-klein, qwen-image-edit-2511, sensenova-u1.5).")] = None,
    width: Annotated[int | None, Field(ge=256, le=4096, description="Default 1024.")] = None,
    height: Annotated[int | None, Field(ge=256, le=4096, description="Default 1024.")] = None,
    extra_images: Annotated[list[str] | None, Field(description="More reference pictures, for models that read several.")] = None,
    steps: Annotated[int | None, Field(description="Default: the model's own.")] = None,
    seed: SeedArg = None,
    wait_seconds: WaitArg = None,
    include_image: IncludeImageArg = False,
) -> CallToolResult:
    """Edits a picture as the instruction says (change colours, hair, clothes, background, add or remove things)
    and returns the new PNG's link. A local file or a picture pasted into the chat cannot be sent directly: upload the file
    with the `upload_local_picture` command list_models gives, then pass the URL it returns. Don't send base64."""
    return await generate_image(
        prompt=instruction, ctx=ctx, model=model, width=width or 1024, height=height or 1024, steps=steps,
        reference_images=[image] + (extra_images or []), seed=seed, wait_seconds=wait_seconds,
        include_image=include_image,
    )


@mcp.tool()
async def generate_video(
    prompt: Annotated[str, Field(description="The clip, its motion and its sound.")],
    ctx: Context,
    model: Annotated[str | None, Field(description="ltx-2.3-q4, ltx-2.3-q8 or ltx-2.5-q4; default: the first downloaded.")] = None,
    width: Annotated[int, Field(ge=256, le=2048, description="Snapped to a multiple of 64.")] = 768,
    height: Annotated[int, Field(ge=256, le=2048, description="Snapped to a multiple of 64.")] = 512,
    seconds: Annotated[float, Field(ge=1, le=10)] = 5.0,
    fps: Literal[24, 25, 30] = 24,
    first_frame: Annotated[str | None, Field(description="Image the clip starts from: the URL (or file name) of an earlier result or upload from this server, used in place: never download and re-upload it; or another URL.")] = None,
    last_frame: Annotated[str | None, Field(description="Image the clip ends on (same forms as first_frame).")] = None,
    quality: Annotated[bool, Field(description="Full guided pipeline: slower, better (needs include_quality download).")] = False,
    auto_duration: Annotated[bool, Field(description="LTX-2.5 only: let the model pick the length, up to `seconds`.")] = False,
    steps: Annotated[int | None, Field(description="First-stage steps; default 8 (30 with quality).")] = None,
    guidance: Annotated[float | None, Field(description="With quality only: video CFG (default 3).")] = None,
    negative_prompt: Annotated[str | None, Field(description="With quality only.")] = None,
    seed: SeedArg = None,
    wait_seconds: WaitArg = None,
    include_image: IncludeImageArg = False,
) -> CallToolResult:
    """Generates a video clip with sound (MP4, H.264) with LTX-2; returns its link."""
    try:
        m = _resolve_model(model, catalog.VIDEO)
        _require_installed(m, quality)
        params: dict[str, Any] = {
            "prompt": prompt, "seed": seed, "steps": _check_steps(m, steps, quality),
            "width": _snap(width, 64), "height": _snap(height, 64),
            "frames": int(round(seconds * fps / 8)) * 8 + 1, "fps": float(fps),
            "guidance": guidance if quality and guidance is not None else m.default_guidance(quality),
        }
        if quality:
            params["quality"] = True
            if negative_prompt:
                params["negative_prompt"] = negative_prompt
        if auto_duration and m.family == "ltx-2.5":
            params["auto_duration"] = True
        if first_frame:
            params["image"] = _input_image(first_frame)
        if last_frame:
            params["last_image"] = _input_image(last_frame)
    except (ValueError, KeyError, OSError) as e:
        return _error(str(e))
    return await _run(_submit(m, "video", params), ctx, wait_seconds, include_image=include_image)


@mcp.tool()
async def generate_song(
    prompt: Annotated[str, Field(description="The music: genre, mood, instruments, voice.")],
    ctx: Context,
    lyrics: Annotated[str | None, Field(description="Lyrics with section tags on their own lines ([verse], [chorus], ...). "
                                                    "Omit for an instrumental.")] = None,
    duration: Annotated[float, Field(ge=10, le=360, description="Longest the song may last, in seconds.")] = 60,
    model: Annotated[str | None, Field(description="minimax-music3-4bit or minimax-music3-8bit.")] = None,
    steps: Annotated[int | None, Field(description="Flow steps per window, 8-30 (default 30).")] = None,
    seed: SeedArg = None,
    wait_seconds: WaitArg = None,
) -> CallToolResult:
    """Composes a song with vocals (or an instrumental) with MiniMax-Music3; returns the WAV's link."""
    try:
        m = _resolve_model(model, catalog.AUDIO, family="minimax-music3")
        if m.family != "minimax-music3":
            raise ValueError("generate_song takes a MiniMax-Music3 model")
        _require_installed(m)
        params = {"prompt": prompt, "lyrics": lyrics.strip() if lyrics and lyrics.strip() else "[instrumental]",
                  "duration": float(duration), "steps": _check_steps(m, steps), "seed": seed}
    except (ValueError, KeyError) as e:
        return _error(str(e))
    return await _run(_submit(m, "song", params), ctx, wait_seconds)


@mcp.tool()
async def generate_sound_effect(
    prompt: Annotated[str, Field(description="The sound: footsteps on gravel, rain on a tin roof, ... (no music or speech).")],
    ctx: Context,
    duration: Annotated[float, Field(ge=0.5, le=30)] = 5.0,
    steps: Annotated[int | None, Field(description="10-200 (default 100).")] = None,
    guidance: Annotated[float | None, Field(ge=1, le=10, description="Default 4.")] = None,
    negative_prompt: str | None = None,
    seed: SeedArg = None,
    wait_seconds: WaitArg = None,
) -> CallToolResult:
    """Makes a sound effect or ambience (48 kHz mono WAV) with MOSS-SoundEffect; returns its link."""
    try:
        m = catalog.find("moss-soundeffect")
        _require_installed(m)
        params = {"prompt": prompt, "duration": round(duration, 1), "steps": _check_steps(m, steps),
                  "guidance": guidance or m.default_guidance(), "seed": seed}
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
    except (ValueError, KeyError) as e:
        return _error(str(e))
    return await _run(_submit(m, "sound", params), ctx, wait_seconds)


@mcp.tool()
async def upscale_image(
    image: ImageRef,
    ctx: Context,
    factor: Annotated[float, Field(ge=1.5, le=4, description="How much the sides grow (2-4 recommended).")] = 2.0,
    softness: Annotated[float, Field(ge=0, le=1, description="Softening before enlarging, for noisy pictures.")] = 0.0,
    seed: SeedArg = None,
    wait_seconds: WaitArg = None,
    include_image: IncludeImageArg = False,
) -> CallToolResult:
    """Enlarges a picture with SeedVR2 3B (one-step diffusion upscaler); returns the PNG's link."""
    try:
        m = catalog.find("seedvr2-3b")
        _require_installed(m)
        params = {"images": [_input_image(image)], "upscale": factor, "softness": softness, "steps": 1, "seed": seed}
    except (ValueError, KeyError, OSError) as e:
        return _error(str(e))
    return await _run(_submit(m, "upscale", params), ctx, wait_seconds, include_image=include_image)


@mcp.tool()
async def image_to_3d(
    image: ImageRef,
    ctx: Context,
    kind: Annotated[Literal["synthetic", "real"], Field(description="synthetic: a render or generated picture cropped "
                                                                    "tight; real: a photo.")] = "synthetic",
    gaussians: Annotated[int, Field(ge=32768, le=1048576, description="Detail; a multiple of 32.")] = 262144,
    steps: Annotated[int, Field(ge=10, le=100)] = 50,
    guidance: Annotated[float, Field(ge=1, le=10)] = 3.0,
    seed: SeedArg = None,
    wait_seconds: WaitArg = None,
) -> CallToolResult:
    """Turns one picture of one object into a 3D Gaussian-splat object (GLB) with Idol; returns its link."""
    try:
        m = catalog.find("idol")
        _require_installed(m)
        params = {"image": _input_image(image), "mode": kind, "num_gaussians": gaussians - gaussians % 32,
                  "steps": steps, "guidance": guidance, "seed": seed}
    except (ValueError, KeyError, OSError) as e:
        return _error(str(e))
    return await _run(_submit(m, "3d", params), ctx, wait_seconds)


# ---------------------------------------------------------------- tools: jobs

@mcp.tool()
async def get_job(
    job_id: str,
    ctx: Context,
    wait_seconds: WaitArg = None,
    include_image: IncludeImageArg = False,
    details: Annotated[bool, Field(description="When done, also return the seed, size, timings and peak memory "
                                               "(text only). Default: the link only.")] = False,
) -> CallToolResult:
    """Status of a generation job; waits for it to finish (up to wait_seconds) and returns its result."""
    job = engine.jobs.get(job_id)
    if job is None:
        return _error(f"no job {job_id}")
    return await _run(job, ctx, wait_seconds, details, include_image)


@mcp.tool()
async def list_jobs() -> CallToolResult:
    """Recent and queued jobs, newest first."""
    rows = [j.summary() for j in sorted(engine.jobs.values(), key=lambda j: j.created, reverse=True)][:50]
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(rows, indent=1))])


@mcp.tool()
async def cancel_job(job_id: str) -> CallToolResult:
    """Cancels a queued or running job."""
    if job_id not in engine.jobs:
        return _error(f"no job {job_id}")
    job = await engine.cancel(job_id)
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(job.summary()))])


@mcp.tool()
async def unload_model() -> CallToolResult:
    """Frees the GPU memory held by the loaded model (when no job is running)."""
    await engine.unload()
    return CallToolResult(content=[TextContent(type="text", text="unloaded")])


# ---------------------------------------------------------------- HTTP routes

@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> Response:
    return JSONResponse({"ok": True, "engine": engine.info,
                         "running": [j.id for j in engine.jobs.values() if j.status == "running"]})


@mcp.custom_route("/upload", methods=["POST"])
async def upload(request: Request) -> Response:
    """Stores a picture sent as the raw request body (curl --data-binary @pic.png) and returns its URL, for use as
    an input to edit_image, upscale_image, generate_video or image_to_3d."""
    data = await request.body()
    if not data:
        return JSONResponse({"error": "send the picture as the request body"}, status_code=400)
    if len(data) > MAX_INPUT_BYTES:
        return JSONResponse({"error": "larger than 64 MB"}, status_code=413)
    ext = _guess_ext(data)
    if not (data.startswith(b"\x89PNG") or data[:3] == b"\xff\xd8\xff" or data[:4] == b"RIFF" or ext == ".heic"):
        return JSONResponse({"error": "not a PNG, JPEG, WebP or HEIC picture"}, status_code=415)
    path = cfg.output_dir / f"upload-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(16).hex()}{ext}"
    path.write_bytes(data)
    return JSONResponse({"url": _file_url(path)})


@mcp.custom_route("/files/{name}", methods=["GET"])
async def files(request: Request) -> Response:
    name = request.path_params["name"]
    path = cfg.output_dir / name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or not path.is_file():
        return Response(status_code=404)
    return FileResponse(path)


class BearerAuth:
    """Requires `Authorization: Bearer <token>` on everything but /health and /files (whose names are
    unguessable: they carry the job's 128-bit id)."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not scope["path"].startswith(("/health", "/files/")):
            headers = dict(scope.get("headers") or [])
            given = headers.get(b"authorization", b"")
            if not hmac.compare_digest(given, b"Bearer " + self.token):
                await JSONResponse({"error": "unauthorized"}, status_code=401,
                                   headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("TURBO_HOST", "0.0.0.0"),
                        help="Interface to listen on (default 0.0.0.0: the whole local network).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TURBO_PORT", "8765")))
    parser.add_argument("--engine", default=cfg.engine, help="Path to the turbo-engine binary.")
    parser.add_argument("--output-dir", default=str(cfg.output_dir))
    parser.add_argument("--public-url", default=cfg.public_url,
                        help="Base URL clients reach this server at, for result links (default http://<LAN IP>:<port>).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg.engine, cfg.output_dir = args.engine, Path(args.output_dir).expanduser()
    engine.binary = cfg.engine
    (cfg.output_dir / "inputs").mkdir(parents=True, exist_ok=True)
    if not Path(cfg.engine).is_file():
        raise SystemExit(f"turbo-engine not found at {cfg.engine}; build it with scripts/build-engine.sh")
    cfg.public_url = args.public_url or f"http://{_lan_ip() if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"

    import uvicorn

    # Stateless: no MCP session lives in this process, so clients survive a server restart.
    app = mcp.streamable_http_app(max_request_body_size=MAX_INPUT_BYTES, host=args.host, stateless_http=True)
    if cfg.token:
        app = BearerAuth(app, cfg.token)
    else:
        log.warning("TURBO_MCP_TOKEN is not set: anyone on the network can use this server")
    log.info("MCP endpoint: %s/mcp  (results at %s/files/...)", cfg.public_url, cfg.public_url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
