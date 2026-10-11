# Runtime: llama.cpp

`openjet` uses `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) as its local/offline inference backend.

In this simplified build, `llama.cpp` is the only supported local runtime.

Pre-built binaries are available for x86, while Jetson/ARM64 commonly requires source build with device-appropriate flags.

Use the deployment-specific build guides:

- [Jetson](../deployment/jetson.md)
- [Linux x86 + NVIDIA](../deployment/linux-x86-nvidia.md)
- [CPU-only](../deployment/cpu-only.md)

The `llama-server` binary can be added to `PATH` or left in `~/llama.cpp/build/bin/`.

## Qwen3.8-27B megakernel backend

For one specific setup, Qwen3.8-27B Q4_K_M on an RTX 30, 40 or 50-series GPU with 24 GB+ (sm_86, sm_89,
sm_120), OpenJet ships a
specialised CUDA engine in [`megakernel/`](../../megakernel/README.md) and serves the model through it
instead of `llama-server`.

There is also an experimental Turing (`sm_75`, including RTX 8000) path with 24 GB+. It requires
CUDA 12.x to compile (CUDA 13 removed `sm_75` compilation support).
Its short reasoning benchmark measured 54.14 tok/s (`-d 3`) and 53.69 tok/s (`-d 4`) on GPU 3,
versus llama.cpp MTP 2 at 32.88 tok/s on GPU 2, both Quadro RTX 8000 cards. No Turing KLD or
long-context validation is claimed; see [Turing build and evidence details](../../megakernel/README.md#turing-rtx-8000).

The following measurements are **RTX 3090 only**, against `llama-server` with OpenJet's flags:

| | llama.cpp | megakernel |
|---|---|---|
| reasoning mode, OpenJet's thinking preset (decode) | 51 tok/s | 78 tok/s (1.5x) |
| greedy decode, chat prompts (mean) | 63 tok/s | 107 tok/s (1.7x) |
| prompt processing, 2-8K tokens | 1050-1140 tok/s | 1530-1600 tok/s (1.3-1.5x) |
| pasting a 12K-token file into the next turn | 10.5 s | 8.1 s |

`openjet setup` builds it when it selects this model on a supported GPU: it compiles the megakernel
source that ships in the wheel for that GPU's architecture, against the CUDA `llama.cpp` build setup
makes (for its tokenizer). That needs the CUDA toolkit (`nvcc`) and takes about 2 minutes; without
it, setup says so and OpenJet uses `llama-server`. In a source checkout you can also build by hand:

```bash
cd megakernel && make
```

With `llama_backend: auto` (the default) OpenJet then starts `megakernel/server/mk_server.py`
whenever the configured model is Qwen3.8-27B Q4_K_M, the megakernel is built, the GPU is sm_75, sm_86, sm_89 or
sm_120 with at least 23,000 MiB of VRAM and `jinja2` is installed; in every other case, or if the megakernel fails to start,
it starts `llama-server` as usual. `llama_backend: llama` turns it off, `llama_backend: megakernel`
makes it required. `OPENJET_MEGAKERNEL_DIR` points at a megakernel build elsewhere.

On multi-GPU hosts, pin `CUDA_VISIBLE_DEVICES` to the GPU's full UUID for both `openjet setup`
and `openjet`; put the CUDA 12.x toolkit first on `PATH` when setting up Turing. For a manual
Turing build, run `make ARCH=sm_75 NVCC=/path/to/cuda-12.x/bin/nvcc LLAMA=/path/to/llama.cpp`
inside `megakernel/`. Use `llama_backend: megakernel` during validation to rule out silent
fallback, then restore `auto` for normal use. When `nvidia-smi`/NVML fails but CUDA is usable,
GPU discovery falls back to the CUDA driver and respects device visibility.

The server speaks the parts of the OpenAI API OpenJet uses: streaming chat completions rendered with
the model's own chat template, reasoning as `reasoning_content`, tool calls as structured
`tool_calls`, and reuse of the processed conversation between turns. Differences from
`llama-server`:

- sampling supports temperature, top_k (at most 32), top_p and min_p; presence and repetition
  penalties are ignored (OpenJet's non-thinking Qwen3.8 preset uses a presence penalty of 1.5)
- the KV cache is fp16, or q8_0 when the configured context does not fit in fp16 (above about 80K
  tokens on a 24 GB card; ~1-2% slower in the RTX 3090 measurements)
- no slot save / restore: the low-memory shell swap below falls back to re-prompting
- editing an earlier part of a conversation (e.g. context compaction) re-processes it
  (~1500 tok/s in the RTX 3090 measurements)

## What setup automates

When you run `openjet setup`, OpenJet can:

- reuse an existing `llama-server` binary
- clone and build `llama.cpp` if `llama-server` is missing
- use a local `.gguf`
- download a recommended GGUF if you do not have a local model ready

Automatic provisioning still depends on local prerequisites like network access, `git`, `cmake`, and a working compiler toolchain.

## Current OpenJet integration

When `llama.cpp` is the active runtime, OpenJet can:

- stream chat and tool calls through `llama-server` (or the megakernel server, see above)
- toggle reasoning mode per request with `/reasoning`
- clear or rebuild the KV cache with `/clear`
- save and restore slot state for low-memory shell workflows

## Low-memory shell swap behavior

For memory-heavy shell commands, OpenJet can temporarily unload the active `llama-server` process, run the command, then reload the model.

Current behavior:

- this path is only wired up for the `llama.cpp` runtime
- OpenJet starts `llama-server` with `--slot-save-path .openjet/state/swap`
- before unload, it saves conversation messages and attempts KV-cache save through the llama slot API
- after the shell command completes, it reloads the server and attempts KV-cache restore
- if KV restore fails, OpenJet keeps the restored messages and falls back to re-prompting the conversation history

Swap state is stored under `.openjet/state/swap/`.
