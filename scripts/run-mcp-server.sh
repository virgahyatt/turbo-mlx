#!/bin/zsh
# Builds turbo-engine if needed, then runs the MCP server on the local network.
#
# Settings come from the environment, or from ~/.config/turbo-mlx-mcp/env (KEY=value lines):
#   TURBO_MCP_TOKEN   bearer token clients must send (strongly recommended on a shared network)
#   TURBO_PORT        default 8765           TURBO_HOST   default 0.0.0.0 (all interfaces)
#   TURBO_OUTPUT_DIR  default ~/TurboMLX/outputs
#   TURBO_PUBLIC_URL  base URL for result links (default http://<LAN IP>:<port>)
#   HF_TOKEN          for gated models (FLUX 3, LTX-2.5)
#   HF_HOME           where models are cached (default ~/.cache/huggingface)
set -euo pipefail
cd "${0:A:h}/.."

ENV_FILE="${TURBO_ENV_FILE:-$HOME/.config/turbo-mlx-mcp/env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi

# xcodebuild needs full Xcode, even when xcode-select points at the Command Line Tools.
if [[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app ]]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

if [[ ! -x build/bin/turbo-engine || -n "$(find Engine/Sources Engine/Package.swift -newer build/bin/turbo-engine -print -quit)" ]]; then
  scripts/build-engine.sh
fi

UV="$(command -v uv || echo /opt/homebrew/bin/uv)"
exec "$UV" run --project mcp-server --frozen turbo-mlx-mcp "$@"
