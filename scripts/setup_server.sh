#!/usr/bin/env bash
# Headless server setup for the consolidated-turing branch — no TUI, no wizard.
# Idempotent: every step is skipped when already satisfied. All paths are
# relative to the open-jet repository root (this script lives in scripts/).
#
#   scripts/setup_server.sh
#
# Does: uv env -> llama.cpp (pinned b10246, Ninja Release) -> model download
# (plain HTTPS, resumable) -> megakernel build linked to that llama.cpp ->
# turing_check sanity gate -> prints the LAN API-server start command.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

die() { echo "setup-server: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

for bin in git cmake ninja nvidia-smi curl; do
    command -v "$bin" >/dev/null || die "$bin not found on PATH"
done
NVCC=$(command -v nvcc || true)
if [ -z "$NVCC" ]; then
    # Arch-family installs CUDA under /opt/cuda; keep the fallback ordering
    # the rest of the repo uses.
    for c in /opt/cuda/bin/nvcc /usr/local/cuda/bin/nvcc; do
        [ -x "$c" ] && NVCC=$c && break
    done
fi
[ -n "$NVCC" ] || die "nvcc not found (CUDA 12.x or 13.x — CUDA 13 supports Turing)"
PATH="$(dirname "$NVCC"):$PATH"; export PATH
command -v uv >/dev/null || die "uv not found; install it: https://docs.astral.sh/uv/"

# GPU arch: derive from nvidia-smi compute capability (megakernel-supported set).
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
case "$CAP" in
    7.5)  ARCH=sm_75  ;;
    8.6)  ARCH=sm_86  ;;
    8.9)  ARCH=sm_89  ;;
    12.0) ARCH=sm_120 ;;
    *) die "compute capability $CAP has no megakernel path (need 7.5/8.6/8.9/12.0, 24 GB+ VRAM)" ;;
esac
echo "==> GPU compute capability $CAP -> ARCH=$ARCH"

step "Python environment (uv, pinned 3.12)"
uv sync

LLAMA_DIR=llama.cpp
step "llama.cpp (managed path: $LLAMA_DIR @ b10246)"
if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi
WANT=$(git -C "$LLAMA_DIR" rev-parse -q --verify refs/tags/b10246 || true)
if [ -z "$WANT" ]; then
    git -C "$LLAMA_DIR" fetch -q origin tag b10246
    WANT=$(git -C "$LLAMA_DIR" rev-parse refs/tags/b10246)
fi
[ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" = "$WANT" ] || git -C "$LLAMA_DIR" checkout -q "$WANT"
GEN=(-G Ninja)
if [ -f "$LLAMA_DIR/build/CMakeCache.txt" ]; then GEN=(); fi   # reuse an existing generator
cmake "${GEN[@]}" -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$LLAMA_DIR/build"

MODEL_DIR=models
step "Model"
eval "$(uv run python -c 'from src.config import QWEN38_27B_FILENAME as f, QWEN38_27B_URL as u; print(f"MODEL={f!r}"); print(f"URL={u!r}")')"
MODEL_PATH="$MODEL_DIR/$MODEL"
if [ ! -f "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_DIR"
    echo "downloading $MODEL (~17 GB, resumable plain HTTPS)..."
    curl -L -C - --fail --retry 5 -o "$MODEL_PATH.part" "$URL"
    mv "$MODEL_PATH.part" "$MODEL_PATH"
else
    echo "already present: $MODEL_PATH"
fi

step "Megakernel ($ARCH, linked against $LLAMA_DIR)"
make -C megakernel "NVCC=$NVCC" "ARCH=$ARCH" libqwen38mk.so qwen38_mk bench/turing_check
printf '{"arch": "%s"}\n' "$ARCH" > megakernel/.openjet-build.json
ldd megakernel/libqwen38mk.so | grep -q "$LLAMA_DIR/build/bin" \
    || die "libqwen38mk.so did not link into $LLAMA_DIR/build/bin — check the LLAMA default"

step "Turing/kernel regression gate"
megakernel/bench/turing_check

CTX=$(uv run python - <<'PY'
from src.hardware import detect_hardware_info
from src.setup import _recommended_context_for_payload
h = detect_hardware_info()
payload = {"model_source": "local", "llama_model": "models/Qwen3.8-27B-Q4_K_M.gguf"}
print(_recommended_context_for_payload(payload, device="cuda", fallback_tokens=8192, total_vram_mb=h.vram_mb))
PY
)

LAN_IP=$(ip -4 -o addr show scope global 2>/dev/null | awk -F'inet ' '{split($2,a,"/"); print a[1]; exit}' || true)
step "Done. Start the API server on the local network with:"
cat <<EOF

  uv run python megakernel/server/mk_server.py \\
      -m $MODEL_PATH --host 0.0.0.0 --port 18080 -c $CTX --kv auto

  LAN endpoint:  http://${LAN_IP:-<this-host-ip>}:18080/v1/chat/completions
  Health check:  curl -s http://127.0.0.1:18080/health
  OpenAI client: base_url="http://${LAN_IP:-<this-host-ip>}:18080/v1"

  --host 0.0.0.0 exposes the server on every interface: it has no auth, so
  firewall port 18080 to your LAN (e.g. ufw allow from 192.168.0.0/16 to any port 18080)
  or use --host <LAN address> to bind one interface only.

  Multi-GPU: pin with CUDA_VISIBLE_DEVICES=<GPU-UUID> (nvidia-smi -L).
EOF
