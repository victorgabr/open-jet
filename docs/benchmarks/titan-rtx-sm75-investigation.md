# TITAN RTX (sm_75) verification of the Turing port

Investigation of PR #18 ("experimental Turing sm_75") claims, run entirely on a
**24 GB NVIDIA TITAN RTX** (compute capability 7.5, 72 SMs) under **CUDA 13.3** — a
Turing card the PR's own matrix never tested. Date: 2026-10-11.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA TITAN RTX, CC 7.5, 24 GB, 72 SMs (single GPU, PCI 05:00.0) |
| OS | Manjaro Linux (Arch), kernel 6.1 |
| Toolkit | nvcc 13.3.73 (`/opt/cuda`), **not** CUDA 12.x |
| llama.cpp baseline | standalone checkout with CUDA `llama-server`, `--spec-type draft-mtp`; later re-verified end-to-end against setup's managed `b10246` checkout |
| Model | unsloth `Qwen3.8-27B-Q4_K_M.gguf`, revision `566347e9…`, SHA256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` (byte-exact match with the PR's published hash) |

## Claim: "Turing requires CUDA 12.x to compile; CUDA 13 removed sm_75 compilation support" — FALSE

Reproduced locally on three levels; NVIDIA's own release notes say CUDA 13.0
"supports all NVIDIA architectures **from Turing** through Grace Blackwell" — the
removed architectures are those **before** CC 7.5 (Maxwell, Pascal, Volta).

1. `nvcc -gencode arch=compute_75,code=sm_75` on a probe kernel compiles; `cuobjdump` shows an `sm_75` cubin.
2. The full megakernel builds: `make ARCH=sm_75 NVCC=/opt/cuda/bin/nvcc LLAMA=<llama.cpp> libqwen38mk.so qwen38_mk` succeeds with CUDA 13.3 (~6 min); `cuobjdump libqwen38mk.so` reports `arch = sm_75`.
3. Everything runs on real Turing hardware: `bench/turing_check` **11/11 PASS**, `mk_server.py` serves `/v1/chat/completions` at 60–93 tok/s decode, and every benchmark below ran on this build.

The docs should instead state: *any CUDA 12.x or 13.x toolkit works; only
pre-Turing architectures lost support in CUDA 13.* The wrong sentence is repeated in
`README.md` (§Turing install/requirements), `megakernel/README.md` (Requirements,
Turing build notes), `docs/benchmarking.md`, `docs/configuration.md` and
`docs/runtimes/llama-cpp.md`.

## Benchmark reproduction (TITAN RTX, this build)

Method matches the PR's harnesses exactly (`megakernel/bench/`, same prompts, seeds,
sampling presets; llama launched with open-jet's flags `-fa on -ctk q8_0 -ctv q8_0
-ngl 99 --spec-type draft-mtp --spec-draft-n-max 2`).

**Reasoning mode (`think_bench.py -n 1024 --seeds 2`, decode tok/s):**

| set | llama.cpp MTP 2 | megakernel d3 | megakernel d4 |
|---|---:|---:|---:|
| all prompts, tokens/time | **40.5** | **54.7** (1.35x) | **56.7** (1.40x) |
| draft acceptance | 59.6% | 43.7% | 44.3% |

**Short-prompt greedy decode (`bench.py -n 512 --ctx 8192` and `code_bench.py`):**

| workload | llama.cpp | megakernel d4 | ratio |
|---|---:|---:|---:|
| bench.py 4 prompts, mean | 49.1 | 71.2 | 1.45x |
| code_bench short chat prompts | 47.2 | 76.3 | 1.62x |

**Context sweep (`code_bench.py`, prompt tok/s / decode tok/s):** prefill parity or
ahead up to 8K (246–739 vs 213–663 tok/s; parity at 15.8K), decode 1.35–1.7x across
0–16K. **Second-turn append:** batched append matches llama's prompt cache
(18.4 s vs 18.8 s for ~12K new tokens); the decode-kernel fallback path is 5.6x
slower (103 s) and should never be reached in normal operation.

The PR's 1.63–1.65x reasoning speedups on a Quadro RTX 8000 (48 GB, 96 SMs) are
directionally confirmed on a 24 GB Turing card at 1.35–1.40x; absolute throughput is
lower as expected from the memory-bandwidth and tensor-core gap (673 GB/s vs the
3090's 936 GB/s). Note the megakernel reaches these margins **despite lower draft
acceptance** than llama.cpp's MTP.

## Findings for the PR amendment

1. **Fix the CUDA-12-only requirement everywhere** (see Claim section); `sm_86`
   remains unaffected. The Makefile default `NVCC ?= /usr/local/cuda/bin/nvcc` is
   additionally wrong on Arch-family distros (`/opt/cuda/bin/nvcc`); document the
   `NVCC=` override, or probe both.
2. **Prefix reuse does not engage through `mk_server.py` on the sm_75 build.**
   A clean append-only conversation replay (exact assistant text fed back) reports
   `usage.prompt_tokens_details.cached_tokens: 0` — the `load_prompt` truncate /
   checkpoint paths in `megakernel/server/mk_server.py` never trigger, so every HTTP
   request re-prefills fully. The CLI `qwen38_mk` batched-append path works (above).
   Suspect `mk_committed` / `mk_checkpoint_pos` bookkeeping in the Turing port
   (`qwen38_mk.cu`); llama.cpp's prompt cache does not have this gap, so multi-turn
   agent sessions lose the reuse the server docstring advertises.
3. **Model download robustness:** setup execs the system `hf` CLI with
   `HF_XET_HIGH_PERFORMANCE=1`; the xet backend stalled at 16 MB with zero bytes for
   minutes on this network, while plain HTTPS (`HF_HUB_DISABLE_XET=1`) sustained
   11.5 MB/s. Suggest a stall watchdog or fallback.
4. Sanity checks confirmed: the PR's 70 focused Python tests pass unchanged; its
   published JSON records are arithmetically self-consistent (all 30 decode samples
   and 18 prefill samples recompute to the summary ratios exactly); model file
   hash matches the PR's provenance.

5. **`openjet setup` oversizes the profile context on 24 GB cards and the
   runtime then dies silently behind the TUI.** The wizard sized
   `context_window_tokens` at `(vram − model)` with a 10% ratio reserve:
   **210,767** tokens here. Both backends fail at launch: the megakernel OOMs
   on the fp16 KV `cudaMalloc` (its kernels read K/V as fp16 — 2 bytes/element —
   while sizing uses the catalog's q8_0 ~1 B/element figure), and llama-server's
   `--fit on` refuses to adapt because openjet pins `-ngl 99` ("n_gpu_layers
   already set by user to 99, abort"), so the 1.1 GiB compute-buffer allocation
   kills the server and the frontend re-exits 1 with no error on screen.
   `openjet fix` targets 32k on this same box — wizard and doctor disagree by
   6.6x. Fixed on this branch (`ac3c3d9`, `6084611`): subtract a 2,560 MB GPU
   runtime reserve in the VRAM sizing branch, and when the model/GPU is
   megakernel-eligible also cap context at double the KV bytes per token.
   Verified clean run: fresh `openjet setup` (managed `b10246` + `nvcc 13.3`)
   writes 72,729 tokens, runtime serves at 23,303/24,576 MiB with ~1.2 GB
   spare, and a live chat turn completes. 145,458 (GGUF-sized, no fp16 cap)
   loads with zero headroom; 210,767 dies at startup.

## Reproduce

```bash
git checkout consolidated-turing
openjet setup            # clones llama.cpp pinned at b10246 into <repo>/llama.cpp,
                         # builds it (GGML_CUDA=ON), then builds the sm_75 megakernel
make -C megakernel NVCC=$(which nvcc) ARCH=sm_75 bench/turing_check  # LLAMA ?= ../llama.cpp now resolves
megakernel/bench/turing_check   # 12/12 PASS against the managed b10246 build
cd megakernel
python bench/bench.py llama --spec 2 --ctx 8192 -n 512 --port 8091
python bench/bench.py mk    --draft 4 --ctx 8192 -n 512
python bench/think_bench.py -n 1024 --seeds 2
python bench/code_bench.py --ctx-sizes 0,2000,4000,8000,16000 --reps 2
```
The benchmark numbers above were measured with the standalone llama.cpp baseline;
the managed `b10246` `llama-server` accepts the same `--spec-type draft-mtp` flags.

Machine-local artifacts (raw harness logs `/tmp/think_bench.out`,
`/tmp/code_bench.out`, `bench/llama_server.log`) are not committed.
