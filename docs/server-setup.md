# Server setup from a source checkout (uv-managed)

Run OpenJet as a server straight from a git checkout — the supported path for fork
branches and for the experimental CUDA megakernel (Turing `sm_75`, RTX 30/40/50).
The repository is fully uv-managed: one lockfile (`uv.lock`), one project
environment (`.venv`), one interpreter pin (`.python-version` = **3.12**;
`requires-python >=3.12` — one dependency does not build on 3.10).

## Prerequisites

- Linux (x86-64), NVIDIA driver for your GPU
- `nvcc` on `PATH` — CUDA 12.x or 13.x (CUDA 13 supports Turing through Grace
  Blackwell; `nvcc 13.3` verified on a TITAN RTX, see
  [`docs/benchmarks/titan-rtx-sm75-investigation.md`](benchmarks/titan-rtx-sm75-investigation.md))
- `git`, `make`, `cmake`, `ninja`
- [uv](https://docs.astral.sh/uv/) — it installs the Python 3.12 interpreter itself
- `bun` — builds the TypeScript TUI (`scripts/build_tui.py`); only needed for the
  interactive UI, the CLI paths work without it
- Free disk: ~17 GB model, ~2 GB build trees; 24 GB+ VRAM for the megakernel path

## Install

```bash
git clone -b consolidated-turing https://github.com/victorgabr/open-jet.git
cd open-jet
uv sync                          # creates .venv from uv.lock, dev group included
uv run python scripts/build_tui.py
uv run openjet setup
```

## Provisioning (`openjet setup`)

Type `/setup` in the TUI (or `/setup recommended`). In order:

1. **Consent gate** — asks before the ~17 GB model download (skipped when the
   `.gguf` is already in `models/`); progress streams in the footer while it works.
2. **llama.cpp** — clones `ggerganov/llama.cpp` into `<repo>/llama.cpp` pinned to
   **`b10246`** and builds it `GGML_CUDA=ON` in `Release` (CMake's default build
   type; ~15–25 min — this is the slow step, cached afterwards via the checkout's
   build tag).
3. **Megakernel** — `nvcc` is probed from `PATH`; builds `megakernel/libqwen38mk.so`
   for your GPU's arch (`sm_75` on Turing) against that checkout, tagging the
   result in `megakernel/.openjet-build.json`.
4. Writes `config.yaml` and starts the runtime. Context is sized to fit real VRAM:
   the VRAM branch reserves 2,560 MB for compute buffers/CUDA context, and when
   the model+GPU are megakernel-eligible the context is additionally capped for
   the megakernel's fp16 K/V (≈72k tokens on a 24 GB card, more on 48 GB).

Re-running setup is cheap (builds are reused unless ref/ABI/arch changed). A full
provision from zero: `rm config.yaml` and re-run; nothing else needs deleting.

## Verify

```bash
uv run openjet fix                    # healthy local runtime, not "none detected"
grep arch megakernel/.openjet-build.json   # {"arch": "sm_75"} on Turing
uv run openjet chat                   # first turn pays ~10 s model load
```

For the Turing path specifically:

```bash
make -C megakernel NVCC=$(which nvcc) ARCH=sm_75 bench/turing_check
megakernel/bench/turing_check          # 12/12 PASS expected on Turing hardware
```

## Build and serve the megakernel manually (no wizard)

The megakernel is the inference engine; llama.cpp provides the model loader and
tokenizer it links against (`libllama`). Any compiled llama.cpp tree qualifies if it
has `include/llama.h` + `build/bin/libllama.so`.

```bash
# 1. llama.cpp with CUDA (any tree; b10246 is openjet's pin)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
git checkout b10246
cmake -G Ninja -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build

# 2. megakernel against that tree, plus the CLI binary
cd <open-jet>
make -C megakernel NVCC=$(which nvcc) ARCH=sm_75 LLAMA=/path/to/llama.cpp libqwen38mk.so qwen38_mk
ldd megakernel/libqwen38mk.so | grep llama   # must resolve inside $LLAMA/build/bin

# 3. serve (OpenAI-compatible)
uv run python megakernel/server/mk_server.py \
    -m models/Qwen3.8-27B-Q4_K_M.gguf --host 127.0.0.1 --port 18080 -c 72729 --kv auto
curl -s http://127.0.0.1:18080/health
curl -s http://127.0.0.1:18080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

# 4. or one-shot CLI, no server
./megakernel/qwen38_mk -m models/Qwen3.8-27B-Q4_K_M.gguf -p "Explain KV caches in one line" \
    -n 256 -d 4 --think xhigh --kv auto
```

- `--kv auto` falls back to q8_0 (~37 KB/token) when fp16 K/V (~73 KB/token) would
  not fit; `--kv q8` forces it. Context budget = weights (~16.7 GB) + KV bytes x
  tokens; 72,729 is the safe 24 GB fp16 number.
- The engine targets exactly `Qwen3.8-27B-Q4_K_M` (filename-matched);
  `draft_vocab_32k.bin` is located automatically, `--lib` overrides the `.so`.
- To let `openjet chat` reuse a manually built tree instead of rebuilding, write the
  matching tag: `echo '{"arch": "sm_75"}' > megakernel/.openjet-build.json`.
- `qwen38_mk` must be relinked whenever you swap `LLAMA=` trees — do not mix binaries
  across llama.cpp builds.

## Development loop

```bash
uv run python -m pytest tests -q       # dev group carries pytest; no extra flags
```

Protocol/schema changes: `cd ui && npm run generate:protocol` regenerates
`ui/src/protocol.generated.ts` from `protocol/openjet-tui-v1.schema.json`.

## Gotchas

- **Arch-family distros**: the megakernel Makefile defaults to
  `NVCC ?= /usr/local/cuda/bin/nvcc`, which does not exist there (`/opt/cuda`).
  `openjet setup` probes `PATH` and is unaffected; manual `make -C megakernel`
  must pass `NVCC=$(which nvcc)`.
- **`<repo>/llama.cpp` is owned by setup** (managed checkout at `b10246`). Do not
  symlink a personal llama.cpp tree onto it — setup will fetch/checkout into that
  path. For manual megakernel compiles against another tree use
  `make LLAMA=/path/to/llama.cpp` (needs `include/llama.h` + `build/bin/libllama.so`).
- **Multi-GPU**: pin by UUID (`CUDA_VISIBLE_DEVICES=<uuid>`); setup and the
  runtime both honor it.
- **uv**: keep `uv.lock` and `.python-version` committed; `uv sync` is the only
  install step. Activating a foreign `VIRTUAL_ENV` makes uv warn — run through
  `uv run` instead.
