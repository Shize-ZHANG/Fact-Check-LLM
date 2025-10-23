#!/usr/bin/env bash
set -euo pipefail

# ===== 可配参数（按需改/或用环境变量覆盖） =====
MODEL_NAME="${MODEL_NAME:-/path/to/Qwen3-32B}"  # 你的本地权重路径或 HF 名称
PORT="${PORT:-8009}"                             # 对外端口
TP_SIZE="${TP_SIZE:-1}"                          # Tensor Parallel 大小；H200 单卡=1，多卡可设为 2/4/8
HOST="${HOST:-0.0.0.0}"                          # 监听地址
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local}"     # 占位即可，方便下游 SDK 连接

# ===== 友情提示 =====
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] PORT=${PORT}  TP_SIZE=${TP_SIZE}  HOST=${HOST}"
echo "[INFO] GPU(s):"
nvidia-smi || true
echo

# ===== H200 单卡是否需要 TP？=====
# - H200 141GB 显存，Qwen3-32B bf16/fp16 单卡可装下 -> TP 不必需（TP_SIZE=1）
# - 只有在 “模型装不下 / 想吃满多卡吞吐” 时，再把 TP_SIZE 调大，并确保已设置 CUDA_VISIBLE_DEVICES
#   例如：CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 bash serve_qwen32b_vllm.sh

# ===== 启动 vLLM OpenAI 兼容服务（极简 + Qwen 必需项）=====
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --port "${PORT}" \
  --host "${HOST}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE}"

# 下面是你可能“偶尔需要”的可选参数（先注释掉，按需解开）：
#  --dtype bfloat16 \                     # 强制 bf16；默认 auto 已够用
#  --max-model-len 4096 \                 # 控输入长度上限；默认够用，长上下文需求时再调
#  --gpu-memory-utilization 0.95 \        # 占用更高显存（吞吐 ↑，OOM 风险 ↑）
#  --kv-cache-dtype fp8 \                 # H200/新卡可尝试 FP8 KV cache（显存更省），需 vLLM 支持
#  --seed 42 \                            # 可复现
#  --max-num-seqs 64 \                    # 并发序列，吞吐相关
#  --max-num-batched-tokens 8192          # 批内 token 上限，吞吐相关
