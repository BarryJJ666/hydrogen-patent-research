#!/bin/bash
# =============================================================================
# 氢能模型评估脚本（极简版）
#
# 用法: ./run_evaluation.sh
# 直接运行，按队列顺序评估模型
# =============================================================================

set -e

# =============================================================================
# 配置
# =============================================================================
TEST_DATA="../hydrogen_benchmark_gen/output/sft_format/test.json"
MAX_SAMPLES="500"      # 留空使用全部数据，测试时可设为 50 或 100
BATCH_SIZE="8"      # 并发数，建议 4-16

# =============================================================================
# 模型队列（按顺序执行，不在队列里的不执行）
# 编辑此数组来控制评估哪些模型及其顺序
# =============================================================================
MODEL_QUEUE=(
    # Direct 模式（Text-to-Cypher）
    "deepseek_v32_direct"
    "deepseek_v31_direct"
    "qwen35_35b_direct"
    "qwen3_235b_direct"
    "glm5_direct"
    "ernie45_direct"

    # Tool Single 模式（一轮工具调用）
    "deepseek_v32_tool_single"
    "deepseek_v31_tool_single"
    "qwen35_35b_tool_single"
    "qwen3_235b_tool_single"
    "glm5_tool_single"
    "ernie45_tool_single"

    # Tool Multi 模式（多轮工具调用）
    "deepseek_v32_tool_multi"
    "deepseek_v31_tool_multi"
    "qwen35_35b_tool_multi"
    "qwen3_235b_tool_multi"
    "glm5_tool_multi"
    "ernie45_tool_multi"
)

# =============================================================================
# 执行
# =============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/run_${TIMESTAMP}.log"
mkdir -p ./logs

echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 氢能模型评估系统"
echo "日志文件: ${LOG_FILE}"
echo "模型队列: ${#MODEL_QUEUE[@]} 个"
echo "=========================================="

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估" | tee -a "$LOG_FILE"

for model in "${MODEL_QUEUE[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 评估模型: $model" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # 构建命令
    cmd="python3 main.py --models $model --test-data \"$TEST_DATA\" --batch-size $BATCH_SIZE"
    [ -n "$MAX_SAMPLES" ] && cmd="$cmd --max-samples $MAX_SAMPLES"

    echo "执行: $cmd" | tee -a "$LOG_FILE"
    eval $cmd 2>&1 | tee -a "$LOG_FILE"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成: $model" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部完成" | tee -a "$LOG_FILE"
echo "结果目录: ./results/raw/ 和 ./results/reports/" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
