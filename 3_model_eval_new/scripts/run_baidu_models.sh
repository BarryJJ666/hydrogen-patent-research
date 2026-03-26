#!/bin/bash
# =============================================================================
# 氢能模型评估脚本
#
# 用法: ./run_baidu_models.sh
# 直接运行，按队列顺序评估模型
#
# 两种评估模式:
# - _single: 单轮，直接输出Cypher
# - _multi: 多轮，执行反馈循环，模型自主判断结果正确性
# =============================================================================

set -e

# =============================================================================
# 配置
# =============================================================================
TEST_DATA="../hydrogen_benchmark_gen/output/sft_format/test.json"
MAX_SAMPLES="50"      # 留空使用全部数据，测试时可设为 50 或 100
BATCH_SIZE="8"      # 并发数，建议 4-16

# =============================================================================
# 模型队列（按顺序执行，不在队列里的不执行）
# 编辑此数组来控制评估哪些模型及其顺序
# =============================================================================
MODEL_QUEUE=(
    # =========================================================================
    # 单轮 (_single)
    # =========================================================================
    "deepseek_v32_single"
    # "deepseek_v31_single"
    # "deepseek_r1_single"
    # "qwen35_35b_single"
    # "qwen3_235b_single"
    # "glm5_single"
    # "ernie45_single"

    # =========================================================================
    # 多轮 (_multi)
    # =========================================================================
    "deepseek_v32_multi"
    # "deepseek_v31_multi"
    # "deepseek_r1_multi"
    # "qwen35_35b_multi"
    # "qwen3_235b_multi"
    # "glm5_multi"
    # "ernie45_multi"
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
