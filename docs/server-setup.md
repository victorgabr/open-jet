# Server setup from a source checkout (uv-managed)

Run OpenJet as a server straight from a git checkout — the supported path for fork
branches and for the experimental CUDA megakernel (Turing `sm_75`, RTX 30/40/50).
The repository is fully uv-managed: one lockfile (`uv.lock`), one project
environment (`.venv`), one interpreter pin (`.python-version` = **3.12**;
`requires-python >=3.12` — one dependency does not build on 3.10).

## Prerequisites

- Linux (x86-64), NVIDIA driver for your GPU
- `nvcc` on `PATH` — **CUDA 12.x or 13.x**; CUDA 13 keeps Turing through
  Grace Blackwell, the "Turing needs CUDA 12.x" claim is wrong (verified with
  `nvcc 13.3` on a TITAN RTX, see
  [`docs/benchmarks/titan-rtx-sm75-investigation.md`](benchmarks/titan-rtx-sm75-investigation.md))
- `git`, `make`, `cmake`
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
   **`b10246`** and builds it with `GGML_CUDA=ON` (~15–25 min; this is the slow
   step, cached afterwards via the checkout's build tag).
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
