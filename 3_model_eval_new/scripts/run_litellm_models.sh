#!/bin/bash
# LiteLLM 模型评估脚本
#
# 支持评估基于 LiteLLM 的外部模型（GPT-5、Gemini、Claude 等）

set -e

# ==============================================================================
# 配置
# ==============================================================================
TEST_DATA="../hydrogen_benchmark_gen/output/sft_format/test.json"
MAX_SAMPLES="5"        # 可选: 50, 100, 500 或留空使用全部数据
BATCH_SIZE="8"          # 并发数: 建议 4-16

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 日志文件（使用新的输出组织方式）
OUTPUT_DIR="../output/run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/execution.log"

# 模型队列
MODEL_QUEUE=(
    # GPT-5 系列
    "gpt5_mini"
    "gpt54"

    # # Gemini 系列（需要设置 GEMINI_API_KEY 环境变量）
    # "gemini_31_pro"

    # # Claude 系列（需要设置 ANTHROPIC_API_KEY 环境变量）
    # "claude_sonnet_46"
)

# ==============================================================================
# 切换到项目根目录
# ==============================================================================
cd "$(dirname "$0")/.."

# ==============================================================================
# 打印配置信息
# ==============================================================================
echo "========================================" | tee -a "$LOG_FILE"
echo "  LiteLLM 模型评估" | tee -a "$LOG_FILE"
echo "  时间: $(date)" | tee -a "$LOG_FILE"
echo "  测试数据: $TEST_DATA" | tee -a "$LOG_FILE"
echo "  最大样本数: ${MAX_SAMPLES:-全部}" | tee -a "$LOG_FILE"
echo "  并发数: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  评估模型: ${MODEL_QUEUE[*]}" | tee -a "$LOG_FILE"
echo "  输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# ==============================================================================
# 评估每个模型
# ==============================================================================
SUCCESS_COUNT=0
FAILED_MODELS=()

for model in "${MODEL_QUEUE[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "正在评估模型: $model ..." | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    # 构建命令（注意：时间戳会在 main.py 中自动生成）
    CMD="python3 main.py --models $model --test-data \"$TEST_DATA\" --batch-size $BATCH_SIZE"
    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max-samples $MAX_SAMPLES"
    fi

    # 执行评估
    if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ $model 评估成功" | tee -a "$LOG_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "✗ $model 评估失败" | tee -a "$LOG_FILE"
        FAILED_MODELS+=("$model")
    fi
done

# ==============================================================================
# 打印摘要
# ==============================================================================
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "  评估完成" | tee -a "$LOG_FILE"
echo "  成功: $SUCCESS_COUNT / ${#MODEL_QUEUE[@]}" | tee -a "$LOG_FILE"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "  失败模型: ${FAILED_MODELS[*]}" | tee -a "$LOG_FILE"
fi
echo "  输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 设置退出码
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    exit 1
fi
