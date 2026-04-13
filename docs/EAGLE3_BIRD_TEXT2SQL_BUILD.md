# EAGLE3 + BIRD Text-to-SQL 构建文档（可执行版）

## 0. 工作方式约定（协作模式）

1. 开发与策略优化在本地环境完成（与 AI 助手协作修改代码、脚本和文档）。
2. 功能与实验验证在公司 Web IDE 环境执行（运行脚本、检查日志、记录结果）。
3. 任何“本地可跑、Web IDE 报错”的问题，优先按“环境差异（Python/依赖版本/GPU 驱动/路径）”排查。
4. 变更提交前必须在 Web IDE 做一次最小复现（smoke test），并把报错与修复记录回本文档。

## 1. 目标与范围

1. 基于 EAGLE3 在 BIRD 上构建单阶段 direct SQL generation 推理加速链路。
2. 模型范围：Qwen2.5-Coder-14B / Qwen3-Coder-30B。
3. 训练路线：微调（B方案）。
4. 准确率主指标：Execution Accuracy（EX）。
5. 第一版解码参数：temperature=0。
6. Prompt 强制包含 BIRD evidence 字段。

`假设`：已具备 BIRD 数据使用权限和本地 sqlite 执行环境。

## 2. 系统架构与加速策略

1. 数据层：BIRD 原始样本与数据库文件整理为统一 JSONL。
2. 检索剪裁层：词法召回 + 结构重排 + 动态 top-k，控制 schema 注入 token。
3. Prompt 层：system + pruned schema + evidence + question + SQL 输出约束。
4. 推理层：EaModel(EAGLE3) 生成 SQL。
5. 评测层：SQL 后处理 + EX 评测。
6. 监控层：记录 new_tokens、wall_time、执行成功率。

## 3. 代码改动清单（文件级+函数级）

### 已落地（当前）

1. `EAGLE-main/eagle/text2sql/bird/prompt_builder.py`
2. `EAGLE-main/eagle/text2sql/bird/postprocess_sql.py`
3. `EAGLE-main/eagle/text2sql/bird/prep_bird.py`
4. `EAGLE-main/eagle/evaluation/gen_ea_answer_qwen3_bird.py`
5. `EAGLE-main/scripts/run_bird_eagle3_infer.sh`
6. `EAGLE-main/eagle/text2sql/bird/eval_exec.py`
7. `EAGLE-main/scripts/run_bird_eval_exec.sh`
8. `docs/BIRD_EX_REPORT_TEMPLATE.md`
9. `EAGLE-main/eagle/traineagle3/data_utils.py`（chat template 无关的监督 mask 构建）
10. `EAGLE-main/eagle/traineagle3/main.py`（接入通用 text2sql 对话样本预处理）
11. `EAGLE-main/eagle/traineagle3/cnets.py`（Qwen2 目标模型加载 + 训练数据 cache 隔离）
12. `EAGLE-main/eagle/text2sql/bird/build_bird_eagle3_sft.py`
13. `EAGLE-main/scripts/run_bird_eagle3_train.sh`

### 待落地（后续批次）

1. `EAGLE-main/eagle/text2sql/bird/schema_index.py`
2. `EAGLE-main/eagle/text2sql/bird/schema_prune.py`
3. 训练后 checkpoint 自动挑选与回归评测联动脚本
4. 失败样本自动归因（按 schema/hint/evidence/SQL 模板分类）

## 4. 实现步骤（可执行）

1. 安装依赖。
2. 用 BIRD train 集构建 EAGLE3 训练数据并训练 `Eagle Head`（Qwen2.5-Coder-14B-Instruct）。
3. 产出 head checkpoint 后，用 BIRD dev 集做推理（temperature=0）。
4. 对推理结果执行 SQL 后处理（必要时）并统计 acceptance 指标。
5. 运行 EX 评测并汇总结果。
6. 在“训练稳定 + 推理可复现”后，再进入 schema 检索剪裁优化。

## 5. 评测设计

1. 主指标：Execution Accuracy。
2. 辅助记录：平均 latency、tokens/sec、SQL 可执行率。
3. 第一版固定 deterministic 解码（temperature=0）。
4. 第二版扩展温度扫描实验。

## 6. 脚本与复现命令（当前主线：先训后测）

```bash
cd EAGLE-main

# 1) 训练（Qwen2.5-Coder-14B-Instruct 对应 Eagle Head）
# 产物：workdir 下训练集 JSONL + 评估集 JSONL + save_dir checkpoint
# 可选：ENABLE_WANDB=1 开启 wandb 记录；默认关闭避免环境依赖导致训练中断
bash scripts/run_bird_eagle3_train.sh \
  /path/to/Qwen2.5-Coder-14B-Instruct \
  /path/to/bird/train/train.json \
  /path/to/bird/train/train_databases \
  /path/to/workdir/train_qwen25 \
  /path/to/checkpoints/eagle3_qwen25_head

# 2) 推理集预处理（BIRD dev）
python -m eagle.text2sql.bird.prep_bird \
  --input-json /path/to/bird/dev_20240627/dev.json \
  --db-root /path/to/bird/dev_20240627/dev_databases \
  --output-jsonl /path/to/workdir/bird_dev_infer.jsonl

# 3) 推理（temperature=0）
bash scripts/run_bird_eagle3_infer.sh \
  /path/to/Qwen2.5-Coder-14B-Instruct \
  /path/to/checkpoints/eagle3_qwen25_head \
  /path/to/workdir/bird_dev_infer.jsonl \
  /path/to/workdir/pred_dev.jsonl

# 4) 统计推理 acceptance（来自 pred_dev.jsonl）
python - <<'PY'
import json, statistics
rows = [json.loads(x) for x in open("/path/to/workdir/pred_dev.jsonl", "r", encoding="utf-8") if x.strip()]
acc = [float(r.get("acceptance_rate", 0.0)) for r in rows]
tok = [int(r.get("accepted_tokens", 0)) for r in rows]
print({"samples": len(rows), "avg_acceptance_rate": statistics.mean(acc) if acc else 0.0, "accepted_tokens_sum": sum(tok)})
PY

# 5) EX评测（执行准确率 + 失败样本 + 报告）
bash scripts/run_bird_eval_exec.sh \
  /path/to/workdir/bird_dev_infer.jsonl \
  /path/to/workdir/pred_dev.jsonl \
  /path/to/bird/dev_20240627/dev_databases \
  /path/to/workdir/eval_summary.json \
  /path/to/workdir/eval_failures.jsonl \
  /path/to/workdir/eval_report.md \
  15
```

## 7. 风险与回滚

1. 30B 显存不足：先 14B 完成闭环。
2. 输出混入解释文字：后处理阶段严格提取 SQL。
3. schema 过长截断：后续批次加入动态剪裁。
4. 评测超时：设置 query timeout 并单独统计。
5. 基座模型与 head 不匹配：训练和推理必须使用同一 base model/tokenizer（本阶段固定 Qwen2.5-Coder-14B-Instruct）。

## 8. 验收标准与里程碑

1. M1：可以从 BIRD JSON 生成推理输入。
2. M2：可以从 BIRD train 自动构建 SFT 数据并启动 Eagle Head 训练。
3. M3：EAGLE3+Qwen2.5 head 能输出结构化 SQL 预测文件。
4. M4：EX 与 acceptance 评测链路跑通并产出结果。
5. M5：接入检索剪裁优化，继续提升稳定性与效果。

## 9. 完整构建过程 TODO（可以动态优化）

### Batch-1（已开始，优先打通）

- [x] 新增 Text-to-SQL prompt 构建模块（含 evidence）
- [x] 新增 SQL 后处理模块
- [x] 新增 BIRD 推理脚本（独立于原 mt_bench）
- [x] 新增最小数据转换脚本（JSON -> JSONL）
- [x] 新增一键推理 shell 脚本
- [ ] 在真实 BIRD dev 子集做首轮 smoke test

### Batch-2（评测闭环）

- [x] 新增 EX 评测封装脚本
- [x] 统一输出格式（db_id/question_id/pred_sql/latency）
- [x] 增加失败样本收集（语法错/执行错/空结果）
- [x] 形成可复现评测报告模板

### Batch-3（检索剪裁优化）

- [ ] 新增 schema 索引构建
- [ ] 新增词法召回 + 结构重排
- [ ] 加入动态 top-k（token budget 控制）
- [ ] 对比“全schema vs 剪裁schema”在 EX 上的变化

### Batch-4A（当前最高优先级：Qwen2.5 Eagle Head 训练）

- [x] 训练侧改造为 chat template 无关的监督 mask（避免模型模板不匹配）
- [x] `traineagle3/cnets.py` 支持 Qwen2 目标模型加载分支
- [x] 加入 BIRD SFT 数据构建脚本 `build_bird_eagle3_sft.py`
- [x] 加入一键训练脚本 `run_bird_eagle3_train.sh`
- [ ] 在 Web IDE 启动完整训练并记录 loss/acc 曲线
- [ ] 产出可用 checkpoint（用于后续推理评测）

### Batch-4B（训练后推理评测闭环）

- [ ] 用训练出的 Qwen2.5 Eagle head 跑 BIRD dev 推理
- [ ] 汇总 acceptance_rate / accepted_tokens / wall_time
- [ ] 运行 EX 评测并出 `eval_summary.json + eval_report.md`
- [ ] 固化“最佳 checkpoint + 对应评测结果”映射

### Batch-5（实验扩展）

- [ ] temperature 扫描实验
- [ ] 多卡策略与吞吐优化
- [ ] 稳定性压测（长 schema、长 SQL、超时）
- [ ] 最终发布版文档与实验结论

> 说明：上述 TODO 为“可以动态优化”，允许在每批结束后重排优先级、拆分任务或替换实现细节。
