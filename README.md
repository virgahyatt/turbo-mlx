# Turbo MLX MCP Server

Local image, video, music, sound-effect, upscaling and image-to-3D generation on an Apple-silicon Mac,
served as an [MCP](https://modelcontextprotocol.io) server on your local network. Any MCP client on the
LAN (Claude Code, Claude Desktop, Cursor, your own agents) can call it.

This is a fork of [rinste/turbo-mlx](https://github.com/rinste/turbo-mlx) with the macOS app (SwiftUI front
end) removed. Its native MLX Swift engine, `turbo-engine`, is unchanged; a small Python server
(`mcp-server/`) drives it over its JSON-lines protocol and exposes it over Streamable HTTP. The original
app's README, with the measurements and model notes, is kept at
[docs/upstream-app-README.md](docs/upstream-app-README.md).

```
MCP client ──HTTP──▶ turbo-mlx-mcp (Python, :8765/mcp) ──stdin/stdout JSON lines──▶ turbo-engine serve (Swift + MLX, GPU)
                         │
                         └─ /files/<name>  results (PNG, MP4, WAV, GLB) for clients to download
```

## Requirements

- A Mac with Apple silicon, macOS 15 or later.
- Xcode 26, with its Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain` once). `xcode-select`
  may keep pointing at the Command Line Tools: the scripts set `DEVELOPER_DIR` to `/Applications/Xcode.app`.
- [uv](https://docs.astral.sh/uv/) (`brew install uv`).
- Disk space for models: 3 GB to 38 GB each, in the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`).

## Run it

```bash
scripts/run-mcp-server.sh
```

This builds `turbo-engine` the first time (a few minutes), then listens on `0.0.0.0:8765` and prints the
endpoint, such as `http://192.168.1.20:8765/mcp`. macOS may ask whether Python may accept incoming
connections: allow it.

To start it at login and keep it running, with a random bearer token:

```bash
scripts/install-launch-agent.sh
```

The token goes in `~/.config/turbo-mlx-mcp/env` (mode 600); logs go to `~/Library/Logs/turbo-mlx-mcp/`.
Remove the agent with `scripts/install-launch-agent.sh uninstall`.

### Settings

Environment variables, or `KEY=value` lines in `~/.config/turbo-mlx-mcp/env`:

| Variable | Default | |
|---|---|---|
| `TURBO_MCP_TOKEN` | none | Bearer token clients must send. Without one, anyone on the network can use the server. |
| `TURBO_HOST` / `TURBO_PORT` | `0.0.0.0` / `8765` | Where to listen. |
| `TURBO_OUTPUT_DIR` | `~/TurboMLX/outputs` | Where results (and fetched input pictures, in `inputs/`) are written. |
| `TURBO_PUBLIC_URL` | `http://<LAN IP>:<port>` | Base of the result links, if clients reach the Mac by another name. |
| `TURBO_WAIT_SECONDS` | `600` | How long a tool call waits before returning a `job_id` to poll. |
| `TURBO_LOW_MEMORY` | `auto` | The app's "Save memory": on below 64 GB of RAM. `1` / `0` to force. |
| `TURBO_CACHE_LIMIT_GB` | RAM / 8 | MLX's buffer cache between allocations (16 GB on a 128 GB Mac). Uncapped, it grew to fill MLX's whole limit and stalled the GPU. |
| `TURBO_ALLOW_LOCAL_PATHS` | off | `1` lets clients pass input pictures as file paths on the server. |
| `HF_TOKEN` | none | Needed for gated models (FLUX 3 Image Klein, LTX-2.5): accept their licence on Hugging Face first. |

## Connect a client

Claude Code, from any machine on the network:

```bash
claude mcp add --transport http turbo-mlx http://192.168.1.20:8765/mcp --header "Authorization: Bearer YOUR_TOKEN"
```

Clients that take a JSON config (Claude Desktop through a remote connector, Cursor, and others):

```json
{
  "mcpServers": {
    "turbo-mlx": {
      "type": "http",
      "url": "http://192.168.1.20:8765/mcp",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```

Check it from another machine with `curl http://192.168.1.20:8765/health`, or run the smoke test, which
generates one image with the first downloaded model:

```bash
cd mcp-server && uv run python tests/smoke_client.py http://192.168.1.20:8765/mcp YOUR_TOKEN
```

## Tools

| Tool | What it does |
|---|---|
| `list_models` | The catalog, what is downloaded, each model's defaults and step range. |
| `download_model` | Fetch a model from Hugging Face (pinned to the commit the engine was checked with). Needs `confirm=true`: without it, it only returns the size, so an agent asks the user first. |
| `get_download`, `cancel_download` | Follow a download (GB, %, speed, ETA) or stop it. |
| `generate_image` | Text to image (or edits with `reference_images`). |
| `edit_image` | Change a picture as an instruction says, keeping the rest (FLUX.2 Klein, FLUX 3, Qwen-Image Edit, SenseNova). |
| `generate_video` | LTX-2.3 / 2.5: a clip with sound from a prompt, optionally from a first and/or last frame (MP4). |
| `generate_song` | MiniMax-Music3: description + lyrics (or instrumental) to a song up to 6 minutes. WAV URL. |
| `generate_sound_effect` | MOSS-SoundEffect: up to 30 s of sound from a description. WAV URL. |
| `upscale_image` | SeedVR2 3B: enlarge a picture 2-4x. |
| `image_to_3d` | Idol: one picture of an object to a Gaussian-splat GLB. |
| `get_job`, `list_jobs`, `cancel_job` | Follow, list and cancel jobs. `get_job(details=true)` adds seed, size, timings and peak memory. |
| `unload_model` | Free the GPU memory of the loaded model. |

A finished generation answers with its link alone, which keeps small models' context clean. Pass
`include_image: true` (on `generate_image`, `edit_image`, `generate_video`, `upscale_image` or `get_job`)
to also get a preview inline: a JPEG of at most 1024 px (a video's first frame), well under the 1 MB
message limit of standard MCP clients. The link is always the full-quality file.

Input pictures can be http(s) URLs, `data:` URIs, base64, or the URL of an earlier result from this server
(so a client can generate, then edit or upscale the result). To use a local file, upload it first; the
response's `url` can then be passed to any tool:

```bash
curl --data-binary @picture.png -H "Authorization: Bearer YOUR_TOKEN" http://192.168.1.20:8765/upload
```

A picture pasted into a chat client is only in the language model's context and can't be passed to the
tools; the agent has to upload its file as above. The GPU runs one job at a time; others queue.
Long calls wait on the server instead of making the client poll: `download_model`, `get_download`, the
generation tools and `get_job` each wait up to `wait_seconds` (default 600) and send MCP progress
notifications meanwhile (a download's say "1.3 of 5.9 GB (22%), 32 MB/s, about 2 min left"). If the work
is not done by then, the result says so and the client calls `get_download` / `get_job` once more.

Downloads run in a child process, so `cancel_download` stops them at once. Files already finished are kept;
the file in progress is deleted and fetched again by the next download.

OpenCode: set `"codemode": false` on this server. With code mode on, the model must write JavaScript to call
tools, which small local models often get wrong.

Result files are served at `/files/<name>` without the token: their names carry the job's random 128-bit
id. Everything else needs the token when one is set.

## Models

The catalog in `mcp-server/turbo_mlx_mcp/catalog.py` is the app's, with the same pinned revisions:

| id | Makes | Download | RAM |
|---|---|---|---|
| `flux2-klein-4b` | image, edits | 4.6 GB | 24 GB |
| `z-image-turbo-q4` / `-q8` | image | 5.9 / 11 GB | 16 / 24 GB |
| `ming-image` | image (transparent background) | 19 GB | 24 GB |
| `flux3-klein` (gated) | image, edits | 20 GB | 16 GB |
| `sensenova-u1.5` | image, edits | 12 GB | 16 GB |
| `moss-image` | image (slow) | 27 GB | 16 GB |
| `qwen-image-2512` | image | 28 GB | 32 GB |
| `qwen-image-edit-2511` | edits | 29 GB | 32 GB |
| `ltx-2.3-q4` / `-q8` | video with sound | 28 / 38 GB | 32 / 64 GB |
| `ltx-2.5-q4` (gated) | video with sound | 29 GB | 32 GB |
| `seedvr2-3b` | upscaling | 7.3 GB | 16 GB |
| `idol` | image to 3D | 3 GB | 16 GB |
| `minimax-music3-4bit` / `-8bit` | songs | 9.2 / 14 GB | 16 / 24 GB |
| `moss-soundeffect` | sound effects | 8.4 GB | 16 GB |

Models the app already downloaded into `~/.cache/huggingface` are found as they are.

## Layout

```
Engine/          turbo-engine: the MLX Swift engine, unchanged from upstream (see Engine/README.md)
mcp-server/      the MCP server (Python, uv project)
scripts/         build-engine.sh, run-mcp-server.sh, install-launch-agent.sh
docs/            engine design notes; the upstream app's README
```

To pull engine improvements from upstream: `git fetch upstream && git merge upstream/main`. Conflicts will
only be in files this fork deleted (the app); keep them deleted.

## License

MIT, as upstream (see [LICENSE](LICENSE)). Each model has its own licence, listed by `list_models`.
