#!/bin/zsh
# Installs a LaunchAgent that starts the MCP server at login and restarts it if it stops.
# Creates ~/.config/turbo-mlx-mcp/env with a random bearer token the first time.
#
# The agent runs a copy of the server and engine in ~/Library/Application Support/turbo-mlx-mcp,
# not this checkout: macOS does not let background agents read external or removable volumes
# without a privacy grant. Run this script again after changing the engine or the server.
#
#   scripts/install-launch-agent.sh            install / update
#   scripts/install-launch-agent.sh uninstall  remove it
set -euo pipefail
cd "${0:A:h}/.."
LABEL="com.turbo-mlx.mcp"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/Library/Logs/turbo-mlx-mcp"
ENV_FILE="$HOME/.config/turbo-mlx-mcp/env"
RUNTIME="$HOME/Library/Application Support/turbo-mlx-mcp"
UV="$(command -v uv || echo /opt/homebrew/bin/uv)"

launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
# bootout returns before the job is gone; bootstrapping too soon fails with "Input/output error".
for _ in {1..20}; do launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1 || break; sleep 0.5; done
if [[ "${1:-}" == "uninstall" ]]; then
  rm -f "$PLIST"
  rm -rf "$RUNTIME"
  print "Removed $LABEL (settings kept in $ENV_FILE, models in the Hugging Face cache)"
  exit 0
fi

if [[ ! -f "$ENV_FILE" ]]; then
  mkdir -p "${ENV_FILE:h}"
  (umask 077; print "TURBO_MCP_TOKEN=$(openssl rand -hex 24)\nTURBO_PORT=8765" > "$ENV_FILE")
  print "Created $ENV_FILE with a new token."
fi

# Build the engine if it is missing or older than its sources.
[[ -z "${DEVELOPER_DIR:-}" && -d /Applications/Xcode.app ]] && export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
if [[ ! -x build/bin/turbo-engine || -n "$(find Engine/Sources Engine/Package.swift -newer build/bin/turbo-engine -print -quit)" ]]; then
  scripts/build-engine.sh
fi

print "Installing the runtime in $RUNTIME"
mkdir -p "$RUNTIME"
rm -rf "$RUNTIME/bin"
cp -R build/bin "$RUNTIME/bin"
# A non-editable install of the server, with the locked dependency versions.
"$UV" venv -q --allow-existing --python 3.12 "$RUNTIME/venv"
"$UV" export -q --project mcp-server --frozen --no-dev --no-emit-project -o "$RUNTIME/requirements.txt"
"$UV" pip install -q --python "$RUNTIME/venv/bin/python" -r "$RUNTIME/requirements.txt"
"$UV" pip install -q --python "$RUNTIME/venv/bin/python" --reinstall-package turbo-mlx-mcp --no-deps ./mcp-server

cat > "$RUNTIME/run.sh" <<RUN
#!/bin/zsh
set -euo pipefail
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi
exec "$RUNTIME/venv/bin/turbo-mlx-mcp" --engine "$RUNTIME/bin/turbo-engine" "\$@"
RUN
chmod +x "$RUNTIME/run.sh"

mkdir -p "$LOGS" "${PLIST:h}"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>$RUNTIME/run.sh</string></array>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOGS/server.log</string>
  <key>StandardErrorPath</key><string>$LOGS/server.log</string>
</dict>
</plist>
PL
: > "$LOGS/server.log"
launchctl bootstrap "gui/$UID" "$PLIST"
source "$ENV_FILE"
print "Started $LABEL. Logs: $LOGS/server.log"
print "Endpoint: http://$(ipconfig getifaddr en0 2>/dev/null || hostname):${TURBO_PORT:-8765}/mcp"
print "Token:    stored in $ENV_FILE"
