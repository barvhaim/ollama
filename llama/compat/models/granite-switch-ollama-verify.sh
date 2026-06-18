#!/usr/bin/env bash
# granite-switch-ollama-verify.sh
#
# Build Ollama from THIS fork (with the granite-switch compat patch) and verify
# `ollama run granite-switch` end-to-end: the two crisp mid-sequence adapter
# demos that were verified in the llama.cpp fork (answerability, query_rewrite).
#
# The granite-switch arch is added the laguna way — a compat patch + a model
# .cpp under llama/compat/models/ — so `cmake -B build .` fetches the pinned
# llama.cpp (b9672), applies our patch, and builds the llama-server runner with
# the arch registered. No fork of llama.cpp is pulled.
#
# Works on Vela (CPU or CUDA) and on an Apple-Silicon Mac (Metal, darwin preset).
#
# Usage:
#   ./granite-switch-ollama-verify.sh build     # cmake + go build only
#   ./granite-switch-ollama-verify.sh convert   # HF -> gs-f16.gguf (needs python deps)
#   ./granite-switch-ollama-verify.sh create    # ollama serve + ollama create
#   ./granite-switch-ollama-verify.sh demo      # the two raw /api/generate demos
#   ./granite-switch-ollama-verify.sh all        # everything in order
#
# Env:
#   GGUF=/path/to/gs-f16.gguf   reuse an existing GGUF (skips convert)
#   SRC_MODEL=ibm-granite/granite-switch-4.1-3b-preview   HF source for convert
#   OLLAMA=./ollama             path to the built ollama binary
#   PORT=11434                  ollama server port

set -euo pipefail

# Repo root is two levels up from llama/compat/models/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SRC_MODEL="${SRC_MODEL:-ibm-granite/granite-switch-4.1-3b-preview}"
GGUF="${GGUF:-$ROOT/gs-f16.gguf}"
OLLAMA="${OLLAMA:-$ROOT/ollama}"
PORT="${PORT:-11434}"
HOST="127.0.0.1:${PORT}"
export OLLAMA_HOST="$HOST"

build() {
  echo "=== cmake configure (fetches b9672 + applies compat patches incl. granite-switch) ==="
  # Top-level CMake pulls llama/server, which FetchContents the pinned llama.cpp
  # and runs apply-patch.cmake over llama/compat/**/*.patch. Our patch registers
  # the granite-switch arch; the models/*.cpp glob compiles granite_switch.cpp.
  cmake -B build .
  echo "=== cmake build (llama-server runner) ==="
  cmake --build build --parallel "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  echo "=== go build (ollama CLI/server) ==="
  go build -o "$OLLAMA" .
  echo "=== confirm the patch actually landed in the fetched tree ==="
  find build -path '*src/models/granite_switch.cpp' -print -quit | grep -q . \
    && echo "OK: granite_switch.cpp present in fetched llama.cpp source" \
    || echo "WARN: granite_switch.cpp not found under build/ (check apply-patch output above)"
}

convert() {
  if [ -f "$GGUF" ]; then echo "=== GGUF present at $GGUF — skip convert ==="; return; fi
  echo "=== convert $SRC_MODEL -> $GGUF using the llama.cpp fork converter ==="
  echo "    (clone github.com/barvhaim/llama.cpp @ feature/granite-switch and run"
  echo "     convert_hf_to_gguf.py — see that repo's granite-switch-mac-demo.sh.)"
  echo "    Or set GGUF=/path/to/existing/gs-f16.gguf and re-run."
  [ -f "$GGUF" ] || { echo "FATAL: no GGUF at $GGUF"; exit 1; }
}

serve() {
  if curl -fsS "http://$HOST/api/version" >/dev/null 2>&1; then
    echo "=== ollama server already up on $HOST ==="; return; fi
  echo "=== start ollama serve (background) ==="
  "$OLLAMA" serve > "$ROOT/ollama-serve.log" 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS "http://$HOST/api/version" >/dev/null 2>&1 && { echo "  up."; return; }
    sleep 1
  done
  echo "FATAL: ollama server did not come up — see ollama-serve.log"; exit 1
}

create() {
  [ -f "$GGUF" ] || { echo "FATAL: $GGUF missing — run convert (or set GGUF=)"; exit 1; }
  serve
  echo "=== ollama create granite-switch ==="
  # Modelfile lives next to this script; rewrite FROM to the resolved GGUF path.
  local mf="$ROOT/llama/compat/models/Modelfile.granite-switch"
  sed "s|^FROM .*|FROM ${GGUF}|" "$mf" > "$ROOT/Modelfile.granite-switch.resolved"
  "$OLLAMA" create granite-switch -f "$ROOT/Modelfile.granite-switch.resolved"
}

# gen PROMPT N — raw completion (raw:true bypasses chat templating so the
# mid-sequence control token reaches the model, matching llama-completion -no-cnv).
gen() {
  local prompt="$1" n="${2:-16}"
  curl -fsS "http://$HOST/api/generate" -d "$(cat <<JSON
{"model":"granite-switch","raw":true,"stream":false,
 "options":{"temperature":0,"num_predict":${n}},
 "prompt":$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}
JSON
)" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("response",""))'
}

demo() {
  serve
  echo
  echo "############ DEMO 1: answerability (<|answerability|>, id 100356) ############"
  echo "# Doc is about the Eiffel Tower; question asks Australia's capital."
  echo "# Base ANSWERS; adapter judges UNANSWERABLE from the doc."
  local ANS='<|start_of_role|>user<|end_of_role|>Document: The Eiffel Tower is in Paris. Question: What is the capital of Australia?<|end_of_role|><|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
  echo "--- OFF (no control) ---"; gen "${ANS}" 12; echo
  echo "--- ON  (<|answerability|> mid-seq) ---"; gen "${ANS}<|answerability|>" 12; echo

  echo
  echo "############ DEMO 2: query_rewrite (<|query_rewrite|>, id 100353) ############"
  echo "# Follow-up 'What other movies has he made?' — base answers; adapter REWRITES."
  local QR='<|start_of_role|>user<|end_of_role|>Who directed Inception?<|end_of_text|><|start_of_role|>assistant<|end_of_role|>Christopher Nolan directed Inception.<|end_of_text|><|start_of_role|>user<|end_of_role|>What other movies has he made?<|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
  echo "--- OFF (no control) ---"; gen "${QR}" 24; echo
  echo "--- ON  (<|query_rewrite|> mid-seq) ---"; gen "${QR}<|query_rewrite|>" 24; echo
  echo "=== DONE. In each pair only the mid-sequence control token differs. ==="
}

case "${1:-all}" in
  build)   build ;;
  convert) convert ;;
  create)  create ;;
  demo)    demo ;;
  all)     build; convert; create; demo ;;
  *) echo "usage: $0 [all|build|convert|create|demo]"; exit 1 ;;
esac
