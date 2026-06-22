# Granite Switch on Ollama

Run **Granite Switch** — a dense Granite-4.1 model with **12 embedded LoRA adapters
selected per-token by control tokens** — via real `ollama run`.

This is added the same way Ollama adds its own custom models (the `laguna`
precedent): a **compat patch** that registers the `granite-switch` arch into the
pinned llama.cpp source, plus a **model `.cpp`** that implements the switched-LoRA
graph. No fork of llama.cpp is pulled — Ollama still FetchContents the pinned
`LLAMA_CPP_VERSION` (`b9672`) and applies our patch on top.

## Files (all under `llama/compat/models/`)

| File | Purpose |
|------|---------|
| `003-llama-cpp-granite-switch.patch` | Registers the `granite-switch` arch into fetched llama.cpp: arch enum + name, 5 KV keys, 14 stacked-LoRA tensor enums/names/infos, model dispatch + Granite scale/rope wiring, the `llama_layer_switch_lora` struct, and the `llama_model_granite_switch` / `llm_graph_input_switch` decls in `models.h`. |
| `granite_switch.cpp` | The switched-LoRA implementation: `load_arch_hparams`, `load_arch_tensors`, and `build_arch_graph`. Identical to the llama.cpp fork's `src/models/granite_switch.cpp` except the include is `"models/models.h"` (Ollama compat layout). Auto-compiled by the `models/*.cpp` glob. |
| `Modelfile.granite-switch` | Minimal `FROM ./gs-f16.gguf` + `temperature 0`. Relies on the GGUF's embedded tokenizer/chat-template. |
| `granite-switch-ollama-verify.sh` | Build + convert + create + demo, on Vela (CPU/CUDA) or Mac (Metal). |

### How the switch works (one paragraph)

The 12 adapters are stored as **stacked** LoRA tensors (stacked dim = 12 + 1; slot 0
is a zero delta for base tokens). The per-token adapter index is recovered **in the
graph** by a single-head causal **router attention** — a faithful copy of the vLLM
(`single.py`) / HF `SingleSwitch` mechanism. `llm_graph_input_switch::set_input` only
fills per-token signals (router K = ±gain for control/normal, router V = the adapter
slot, router Q = 1) and **substitutes** each control token's id before embedding; the
causal softmax over those signals then yields each token's slot. That router K/V lives
in the model's KV cache at an extra layer (`hparams.router_layer == n_layer`), so the
selection is **per-sequence** — concurrent requests are isolated with no global state
to leak (the bug the prior CPU-side sticky index had). The decoder then adds the
per-token LoRA delta on qkv/o/gate/up/down via `ggml_mul_mat_id` over the stacked
tensors (MoE-style indexed matmul, repurposed: "expert = adapter"). B is pre-scaled by
`alpha/rank` at compose time, so the runtime LoRA scale is 1.0.

**Single-switch contract (known limitation).** Like vLLM/HF, the router uses flat gain
(no recency), so within one sequence an adapter, once fired by a control token, stays
selected for the rest of that sequence — there is "no mechanism to transition back to
base mid-sequence" (vLLM's own wording). vLLM/HF never observe this because each served
request is a fresh sequence; a chat client that continues one KV cache across turns
(`ollama run`) carries the selection into later turns. To get base behavior in a new
turn, start a fresh sequence.

## Build & run

Ollama's top-level `cmake -B build .` pulls `llama/server`, which FetchContents the
pinned llama.cpp and runs `apply-patch.cmake` over `llama/compat/**/*.patch` — our
patch lands automatically, and the `models/*.cpp` glob compiles `granite_switch.cpp`.

```bash
# from the repo root
cmake -B build .                       # fetches b9672, applies compat patches, configures
cmake --build build --parallel 8       # builds the llama-server runner (arch registered)
go build -o ollama .                   # builds the ollama CLI/server

# bring up the model (reuse the verified gs-f16.gguf, or convert — see below)
GGUF=/path/to/gs-f16.gguf ./llama/compat/models/granite-switch-ollama-verify.sh create
./ollama run granite-switch
```

On an **Apple-Silicon Mac**, Metal is selected by the `darwin` preset automatically;
the same three commands apply. Requirements: Xcode CLT, CMake ≥ 3.24, Go, and
~16 GB unified memory for the f16 model (8.4 GB on disk).

### Getting the GGUF

The converter lives in the llama.cpp fork, not here. Either:

- **Reuse** an existing `gs-f16.gguf` (842 tensors, stacked dim 13) and point `GGUF=` at it, or
- **Convert** from `ibm-granite/granite-switch-4.1-3b-preview`: clone
  `github.com/barvhaim/llama.cpp` @ `feature/granite-switch` and run
  `PYTHONPATH=gguf-py python convert_hf_to_gguf.py <hf_dir> --outfile gs-f16.gguf --outtype f16`
  (see that repo's `granite-switch-mac-demo.sh`).

`ollama create` preserves all GGUF metadata (the custom `granite-switch.*` keys, the
stacked LoRA tensors, and the tokenizer/chat-template) verbatim. The arch is validated
at **run** time by the patched llama-server runner.

## Verifying the per-token switch

Use the **raw** completion endpoint (`raw:true`) so the mid-sequence control token
reaches the model untemplated (equivalent to `llama-completion -no-cnv`). Greedy
decode (`temperature 0`). `granite-switch-ollama-verify.sh demo` runs both:

**Demo 1 — answerability** (`<|answerability|>`, id 100356). Document is about the
Eiffel Tower; the question asks Australia's capital. Base answers; the adapter judges
it **unanswerable** from the document.

**Demo 2 — query_rewrite** (`<|query_rewrite|>`, id 100353). Follow-up "What other
movies has he made?" — base answers it; the adapter **rewrites** it into a standalone
query (resolving "he" → Christopher Nolan).

In each pair only the mid-sequence control token differs, so the divergent output is
the per-token switch firing. Outputs match the llama.cpp fork's `llama-completion
--temp 0` reference. If the switch doesn't fire, the prompt was templated — confirm
`raw:true` and that the control-token ids appear in the server logs.

## Regenerating the patch (only if `LLAMA_CPP_VERSION` changes)

The patch is authored against `b9672` with exact context, so it applies (and
reverse-checks for idempotency) cleanly. If the pin moves, re-transcribe the arch
edits onto the new tag, rebuild to confirm it compiles, and regenerate with
`git diff` over the five files
(`src/llama-arch.{h,cpp}`, `src/llama-model.{cpp,h}`, `src/models/models.h`).
