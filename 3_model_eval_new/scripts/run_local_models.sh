#!/bin/bash
# =============================================================================
# 本地模型评估脚本
#
# 用法: ./scripts/run_local_models.sh
# 按顺序评估所有本地 Qwen2.5-7B 模型
# =============================================================================

set -eo pipefail

# =============================================================================
# 配置
# =============================================================================
TEST_DATA="../hydrogen_benchmark_gen/output/sft_format/test.json"
MAX_SAMPLES="500"        # 留空使用全部数据，测试时可设为 50 或 100
BATCH_SIZE="8"        # 并发数，已修复线程安全问题，可以用更大的值

# =============================================================================
# 模型队列（按顺序执行）
# =============================================================================
MODEL_QUEUE=(
    # "qwen25_7b"         # Qwen2.5-7B-Instruct (Base)
    # "qwen25_7b_sft"     # Qwen2.5-7B-Instruct (SFT)
    "qwen25_7b_rl"      # Qwen2.5-7B-Instruct (RL)
)

# =============================================================================
# 执行
# =============================================================================
cd "$(dirname "$0")/.."  # 切换到项目根目录

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/local_models_${TIMESTAMP}.log"
mkdir -p ./logs

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 本地模型评估系统"
echo "日志文件: ${LOG_FILE}"
echo "模型队列: ${#MODEL_QUEUE[@]} 个"
echo "评估样本: ${MAX_SAMPLES:-全部}"
echo "=========================================="

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估" | tee -a "$LOG_FILE"

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=""

for model in "${MODEL_QUEUE[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 评估模型: $model" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # 构建命令
    cmd="python3 main.py --models $model --test-data \"$TEST_DATA\" --batch-size $BATCH_SIZE"
    [ -n "$MAX_SAMPLES" ] && cmd="$cmd --max-samples $MAX_SAMPLES"

    echo "执行: $cmd" | tee -a "$LOG_FILE"

    # 执行并捕获返回值
    if eval $cmd 2>&1 | tee -a "$LOG_FILE"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ 完成: $model" | tee -a "$LOG_FILE"
        ((SUCCESS_COUNT++)) || true
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ 失败: $model" | tee -a "$LOG_FILE"
        ((FAIL_COUNT++)) || true
        FAILED_MODELS="$FAILED_MODELS $model"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 评估完成" | tee -a "$LOG_FILE"
echo "成功: ${SUCCESS_COUNT} 个" | tee -a "$LOG_FILE"
echo "失败: ${FAIL_COUNT} 个" | tee -a "$LOG_FILE"
[ -n "$FAILED_MODELS" ] && echo "失败模型:${FAILED_MODELS}" | tee -a "$LOG_FILE"
echo "结果目录: ./results/raw/ 和 ./results/reports/" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
