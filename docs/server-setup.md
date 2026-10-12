# Server setup from a source checkout

OpenJet runs as a server from a git checkout. This is the supported path for
fork branches and for the experimental CUDA megakernel (Turing `sm_75`,
RTX 30/40/50). uv manages the whole repository: one lockfile (`uv.lock`), one
project environment (`.venv`), and one interpreter pin (`.python-version`,
Python 3.12). The project needs Python 3.12 or later because one dependency
does not build on 3.10.

Two install paths exist. The headless path runs one script,
`scripts/setup_server.sh`, with no interactive UI. It performs every step this
document describes in one pass and ends by printing the command that starts the
API server on your local network. The wizard path is the default TUI flow. The
sections below cover the wizard path first, then the headless script, then the
manual steps.

## Prerequisites

Before you start, the host must have:

- Linux (x86-64) and an NVIDIA driver for your GPU.
- `nvcc` in `PATH`, from CUDA 12.x or 13.x. CUDA 13 supports Turing through
  Grace Blackwell. `nvcc 13.3` ran on a TITAN RTX, see
  [`docs/benchmarks/titan-rtx-sm75-investigation.md`](benchmarks/titan-rtx-sm75-investigation.md).
- `git`, `make`, `cmake`, and `ninja`.
- [uv](https://docs.astral.sh/uv/). uv installs the Python 3.12 interpreter by
  itself.
- `bun`. It builds the TypeScript TUI through `scripts/build_tui.py`. You need
  it only for the interactive UI. The CLI paths work without it.
- About 17 GB of free disk for the model and about 2 GB for the build trees.
  The megakernel path needs 24 GB or more of VRAM.

## Install

Clone the branch, create the environment, build the TUI, then run the setup
wizard:

```bash
git clone -b consolidated-turing https://github.com/victorgabr/open-jet.git
cd open-jet
uv sync                          # creates .venv from uv.lock, dev group included
uv run python scripts/build_tui.py
uv run openjet setup
```

### Headless setup without the TUI

`scripts/setup_server.sh` repeats every Install step without the wizard. You
can run the script twice. A second run changes nothing.

The script prints an elapsed-time marker and a time estimate for each step, and
it streams the tool's own progress while a step runs. On a terminal the script
asks before it downloads the ~17 GB model. Pass `--yes` to skip that question.
When you pipe the script's output, it proceeds without asking.
After the setup, the script prints how to keep the virtual environment active in
your shell: `source .venv/bin/activate`.

The script creates the uv environment, clones llama.cpp at `b10246`, and builds
the `llama` and `llama-server` targets in Release mode with Ninja. It downloads
the model over plain HTTPS and resumes a stopped download. It builds the
megakernel for the GPU in the host, writes the build tag, and runs the
`turing_check` gate on Turing hardware. Then the script prints the
`mk_server.py` start command. That command carries a context size that fits the
VRAM of the host, and the printout shows the LAN address of the host.

## Provisioning with `openjet setup`

The wizard runs when you type `/setup` in the TUI. Type `/setup recommended`
for the default choices. The wizard works in four steps:

1. The wizard asks for consent before it downloads the ~17 GB model. It skips
   the question when the `.gguf` file already sits in `models/`. Download
   progress streams in the footer while the download works.
2. The wizard clones `ggerganov/llama.cpp` into `<repo>/llama.cpp` at `b10246`.
   It builds that tree with `GGML_CUDA=ON` in Release mode. This step takes 15
   to 25 minutes. Later runs reuse the checkout and its build tag.
3. The wizard probes `nvcc` in `PATH` and builds `megakernel/libqwen38mk.so`
   for the arch of your GPU, for example `sm_75` on Turing. It records the
   build in `megakernel/.openjet-build.json`.
4. The wizard writes `config.yaml` and starts the runtime. It sizes the context
   to the real VRAM. The calculation keeps 2,560 MB free for the compute
   buffers and the CUDA context. On a megakernel GPU it also caps the context
   for the fp16 K/V cache. That cap gives about 72,000 tokens on a 24 GB card,
   and more on 48 GB.

You can run the setup wizard again without cost. It reuses the builds unless
the ref, the ABI, or the arch changed. To provision from zero, delete
`config.yaml` and run the wizard again. Nothing else needs deleting.

## Make sure the install works

Run these commands from the repo root. `openjet fix` must report a healthy
local runtime, not "none detected". The first chat turn pays about 10 seconds
of model load.

```bash
uv run openjet fix                    # healthy local runtime, not "none detected"
grep arch megakernel/.openjet-build.json   # {"arch": "sm_75"} on Turing
uv run openjet chat                   # first turn pays ~10 s model load
```

For the Turing path, build and run the regression gate. The gate tests the
kernel math without loading the model. All twelve checks must report PASS on
Turing hardware.

```bash
make -C megakernel NVCC=$(which nvcc) ARCH=sm_75 bench/turing_check
megakernel/bench/turing_check          # 12 of 12 PASS on Turing hardware
```

## Build and serve the megakernel manually (no wizard)

`openjet setup` and `scripts/setup_server.sh` normally build llama.cpp and the
megakernel for you. Use this section when you want to run the steps by hand.
Use it also when you want to link the megakernel against your own llama.cpp
tree, for example a fork or a different version pin. The megakernel is the
inference engine. llama.cpp gives it the model loader and the tokenizer through
the `libllama` library. A llama.cpp tree works when it has `include/llama.h`
and `build/bin/libllama.so`. Run every command in this section from the
open-jet repo root. Before step 2, the file
`models/Qwen3.8-27B-Q4_K_M.gguf` must exist. The wizard or
`scripts/setup_server.sh` downloads it for you.

### 1. Build

Clone llama.cpp at the version the repository pins, build the two targets the
megakernel links against, then build the megakernel:

```bash
# Skip the clone, checkout, and build when <repo>/llama.cpp exists already.
git clone https://github.com/ggerganov/llama.cpp llama.cpp
git -C llama.cpp checkout b10246
cmake -G Ninja -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama llama-server

# The Makefile reads LLAMA ?= ../llama.cpp, which matches the clone above.
# Set LLAMA=/path only for another tree. Re-run this make command after every
# tree swap, because qwen38_mk and the .so must match one llama.cpp build.
make -C megakernel NVCC=$(which nvcc) ARCH=sm_75 libqwen38mk.so qwen38_mk
ldd megakernel/libqwen38mk.so | grep llama    # the path must sit in llama.cpp/build/bin
```

### 2. Serve on localhost

Start the server on the loopback address. Only processes on this host can reach
it.

```bash
uv run python megakernel/server/mk_server.py \
    -m models/Qwen3.8-27B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 72729 --kv auto
curl -s http://127.0.0.1:8080/health
```

### 3. Serve on the local network

Start the server on every interface with `--host 0.0.0.0`. Do not expose port
8080 to the public network. The server has no authentication and no TLS.

```bash
uv run python megakernel/server/mk_server.py \
    -m models/Qwen3.8-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -c 72729 --kv auto
```

A client on the network sends requests to
`http://<server-lan-ip>:8080/v1/chat/completions`. The placeholder
`<server-lan-ip>` is the LAN address of the server host. The script
`scripts/setup_server.sh` prints that address. Any OpenAI SDK works when you
set `base_url` to `http://<server-lan-ip>:8080/v1`. This example request runs
on another machine:

```bash
curl -s http://<server-lan-ip>:8080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Choose one of these two protections before you start the server. Bind one
interface instead of all of them, for example `--host 192.168.1.20`. Or
restrict the port to your subnet with
`sudo ufw allow from 192.168.0.0/16 to any port 8080 proto tcp`.

On a host with more than one GPU, start the server with
`CUDA_VISIBLE_DEVICES=<GPU-UUID>`. Run `nvidia-smi -L` on that host to list the
UUIDs. The setup wizard honors the same variable.

### 4. One-shot CLI without a server

Run one prompt through the megakernel without a server:

```bash
./megakernel/qwen38_mk -m models/Qwen3.8-27B-Q4_K_M.gguf -p "Explain KV caches in one line" \
    -n 256 -d 4 --think xhigh --kv auto
```

### Runtime notes

About the runtime:

- `--kv auto` picks q8_0, about 37 KB per token, when the fp16 K/V cache of
  about 73 KB per token does not fit the free VRAM. `--kv q8` forces q8_0. The
  context budget is the model weights (about 16.7 GB) plus the KV bytes times
  the token count. For a 24 GB card with fp16, 72,729 tokens is the safe
  number.
- The engine runs one model, `Qwen3.8-27B-Q4_K_M`, and matches it by file name.
  It finds `draft_vocab_32k.bin` on its own. Use `--lib` to point the loader at
  a different `.so` file.
- To make `openjet setup` and `openjet chat` reuse a manually built megakernel
  instead of rebuilding it, write the matching tag with
  `echo '{"arch": "sm_75"}' > megakernel/.openjet-build.json`.

## Development loop

Run the test suite from the repo root. The dev group carries pytest, so the
command needs no extra flags.

```bash
uv run python -m pytest tests -q       # dev group carries pytest; no extra flags
```

After you change `protocol/openjet-tui-v1.schema.json`, regenerate the
TypeScript types. Run `npm --prefix ui run generate:protocol` from the repo
root. The command rewrites `ui/src/protocol.generated.ts` from the schema file.

## Known problems

On Arch-family distros, CUDA lives in `/opt/cuda`, and the path
`/usr/local/cuda/bin/nvcc` does not exist. The megakernel Makefile uses that
missing path as its default. `openjet setup` probes `PATH` first, so the wizard
is not affected. For a manual build, pass `NVCC=$(which nvcc)` to the
`make -C megakernel` command.

The setup wizard owns `<repo>/llama.cpp`. The wizard clones that path and
checks out `b10246` inside it. Do not put a symlink to a personal llama.cpp
tree at that path. The wizard will fetch and check out over it. To build the
megakernel against a different tree, run
`make -C megakernel LLAMA=/path/to/llama.cpp NVCC=$(which nvcc) ARCH=sm_75`.
That tree needs `include/llama.h` and `build/bin/libllama.so`.

Keep `uv.lock` and `.python-version` committed. `uv sync` is the only install
step. When a foreign `VIRTUAL_ENV` is active, uv prints a warning and ignores
it. Run commands through `uv run` instead.
