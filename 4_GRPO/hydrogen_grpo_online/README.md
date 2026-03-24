# hydrogen_grpo_online

氢能专利知识图谱 Text-to-Cypher GRPO 训练（在线执行奖励函数版本）

## 与离线版本的区别

| 特性 | 离线版本 (hydrogen_grpo) | 在线版本 (hydrogen_grpo_online) |
|------|--------------------------|--------------------------------|
| 奖励函数 | 文本相似度（token F1, 子句相似度等）| **执行正确性 + 语法正确性** |
| 是否连接 Neo4j | 否 | 是（训练时执行预测 Cypher）|
| 优化目标 | 文本相似 | **执行准确** |
| 预期效果 | 可能 reward hacking | 直接优化真实目标 |

## 奖励函数设计

```
总分 = 0.7 × execution_score + 0.3 × syntax_score
```

| 情况 | execution | syntax | 总分 |
|------|-----------|--------|------|
| 语法错误 | 0.0 | 0.0 | **0.00** |
| 可执行但结果为空 | 0.1 | 1.0 | **0.37** |
| 可执行，结果非空但行数不同 | 0.3 | 1.0 | **0.51** |
| 可执行，行数相同但内容不同 | 0.7 | 1.0 | **0.79** |
| 完全匹配 | 1.0 | 1.0 | **1.00** |

## 目录结构

```
hydrogen_grpo_online/
├── README.md                      # 本文件
├── data/
│   ├── gold_answers.json          # 预执行的金标答案
│   ├── train.parquet              # 训练数据
│   └── test.parquet               # 测试数据
├── precompute_gold_answers.py     # Step 1: 预执行金标 Cypher
├── data_preprocess.py             # Step 2: 生成 parquet 数据
├── reward_fn_online.py            # 在线奖励函数
├── run_3b_grpo_4gpu.sh            # 3B 模型训练脚本
└── run_7b_grpo_4gpu.sh            # 7B 模型训练脚本
```

## 使用方法

### Step 1: 预执行金标答案

```bash
cd /ssd1/zhangyuzhe/verl-release-v0.7.0
python hydrogen_grpo_online/precompute_gold_answers.py
```

输出：`hydrogen_grpo_online/data/gold_answers.json`

### Step 2: 生成训练数据

```bash
python hydrogen_grpo_online/data_preprocess.py
```

输出：
- `hydrogen_grpo_online/data/train.parquet`
- `hydrogen_grpo_online/data/test.parquet`

### Step 3: 开始训练

```bash
# 3B 模型
bash hydrogen_grpo_online/run_3b_grpo_4gpu.sh

# 7B 模型
bash hydrogen_grpo_online/run_7b_grpo_4gpu.sh
```

### Step 4: 查看 TensorBoard

```bash
tensorboard --logdir=tensorboard_log/hydrogen_patent_grpo_online --port=6010 --bind_all
```

关注指标：
- `critic/rewards/mean`: 应该上升
- `execution_score`: 应该上升（核心指标）
- `syntax_score`: 应该维持高位
- `actor/kl_loss`: 应该缓慢上升（不要太快）

## 训练参数调整

相比离线版本，主要调整：

| 参数 | 离线版本 | 在线版本 | 原因 |
|------|----------|----------|------|
| `kl_loss_coef` | 0.001 | **0.01** | 增强 KL 约束，防止策略漂移过快 |
| `lr` (7B) | 2e-6 | **1e-6** | 降低学习率稳定训练 |
| `lr` (3B) | 3e-6 | **1.5e-6** | 降低学习率稳定训练 |

## 依赖

- Python 3.10+
- neo4j (Python driver)
- pandas
- tqdm
- VERL v0.7.0

## Neo4j 配置

默认连接：
- URI: `bolt://10.223.3.13:7687`
- User: `neo4j`
- Password: `zhangyuzhe`

如需修改，编辑 `precompute_gold_answers.py` 和 `reward_fn_online.py` 中的 `NEO4J_CONFIG`。
