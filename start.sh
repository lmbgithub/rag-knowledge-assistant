#!/usr/bin/env bash
#
# Start the stack on the fastest inference backend this machine has.
#
# The order is not a preference. Docker Desktop on macOS runs a Linux VM with
# no Metal passthrough, so ollama inside a container on a Mac is CPU-only
# however much GPU the machine has — a host ollama therefore beats a bundled
# one on Apple silicon every time, and the container is the fallback rather
# than the default.
#
#   ./start.sh              pick automatically
#   ./start.sh host         force the ollama already running on this machine
#   ./start.sh gpu          force the bundled ollama with NVIDIA passthrough
#   ./start.sh cpu          force the bundled ollama on CPU
#   ./start.sh offline      no model and no database — fakes, for the UI only
#   ./start.sh --down       stop everything
set -euo pipefail

cd "$(dirname "$0")"

CHAT_MODEL="${CHAT_MODEL:-qwen3:1.7b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
HOST_OLLAMA="${HOST_OLLAMA:-http://localhost:11434}"

say() { printf '  %s\n' "$*"; }

if [[ "${1:-}" == "--down" ]]; then
  docker compose --profile bundled down
  exit 0
fi

host_ollama_up() { curl -fsS -m 3 "$HOST_OLLAMA/api/version" >/dev/null 2>&1; }

# /api/tags reports an untagged model as "name:latest", so a bare "qwen3:1.7b"
# matches and a bare "nomic-embed-text" does not. Without the normalisation
# below, every start re-pulls the embedding model.
host_has_model() {
  local want="$1"
  [[ "$want" == *:* ]] || want="$want:latest"
  curl -fsS -m 5 "$HOST_OLLAMA/api/tags" 2>/dev/null | grep -q "\"$want\""
}

nvidia_ready() {
  command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi -L >/dev/null 2>&1 &&
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}

mode="${1:-auto}"
if [[ "$mode" == "auto" ]]; then
  if host_ollama_up; then
    mode=host
  elif nvidia_ready; then
    mode=gpu
  else
    mode=cpu
  fi
fi

case "$mode" in
host)
  say "using the ollama already running at $HOST_OLLAMA"
  say "on Apple silicon that is Metal; in a container it would be CPU-only"
  for model in "$EMBED_MODEL" "$CHAT_MODEL"; do
    if host_has_model "$model"; then
      say "$model is already pulled"
    else
      say "pulling $model into the host's ollama"
      ollama pull "$model"
    fi
  done
  OLLAMA_HOST="http://host.docker.internal:11434" \
    CHAT_MODEL="$CHAT_MODEL" EMBED_MODEL="$EMBED_MODEL" \
    docker compose up --build -d
  ;;
gpu)
  say "no host ollama; using the bundled one with the NVIDIA GPU"
  OLLAMA_HOST="http://ollama:11434" CHAT_MODEL="$CHAT_MODEL" EMBED_MODEL="$EMBED_MODEL" \
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile bundled up --build -d
  ;;
cpu)
  say "no GPU and no host ollama: the bundled ollama will run on CPU."
  say "expect answers to take a minute or more. This fallback is loud on"
  say "purpose — a silent one is how a large slowdown goes unnoticed."
  OLLAMA_HOST="http://ollama:11434" CHAT_MODEL="$CHAT_MODEL" EMBED_MODEL="$EMBED_MODEL" \
    docker compose --profile bundled up --build -d
  ;;
offline)
  say "fakes only: a hashing embedder and an echoing model, in memory."
  say "the pipeline and the UI work end to end; the answers are not real."
  OFFLINE=true docker compose up --build -d api web
  ;;
*)
  echo "usage: $0 [host|gpu|cpu|offline|--down]" >&2
  exit 2
  ;;
esac

say ""
say "web  http://localhost:3000"
say "api  http://localhost:8000/health"
