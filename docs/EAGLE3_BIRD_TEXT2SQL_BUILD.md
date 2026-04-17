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

# 4) EX评测（执行准确率 + 接受率聚合 + 失败样本 + 报告）
bash scripts/run_bird_eval_exec.sh \
  /path/to/workdir/bird_dev_infer.jsonl \
  /path/to/workdir/pred_dev.jsonl \
  /path/to/bird/dev_20240627/dev_databases \
  /path/to/workdir/eval_summary.json \
  /path/to/workdir/eval_failures.jsonl \
  /path/to/workdir/eval_report.md \
  15

# 5) 一键全流程（dev预处理 + 推理 + 评测）
# 5.1 默认内嵌参数（0 参数直接运行）
bash scripts/run_bird_eagle3_full_eval.sh

# 5.2 推荐：通过环境变量覆盖关键路径（无需改脚本）
BASE_MODEL_PATH=/path/to/Qwen2.5-Coder-14B-Instruct \
EA_MODEL_PATH=/path/to/checkpoints/eagle3_qwen25_head \
WORKDIR=/path/to/workdir/full_eval \
TIMEOUT_SEC=15 \
bash scripts/run_bird_eagle3_full_eval.sh

# 5.2.1 训练后 head 目录仅包含 state_x/pytorch_model.bin 的兼容说明
# 如果 EA_MODEL_PATH 指向 head 根目录（例如 .../Qwen2.5-Coder-14B-Instruct_eagle3_head），
# run_bird_eagle3_full_eval.sh 会自动：
# 1) 选择最新且包含模型文件的 state_x；
# 2) 补齐 config.json（优先从 head 目录查找，否则回退到 eagle/traineagle3/config.json）；
# 3) 在 WORKDIR/resolved_ea_model 下组装可加载目录并用于推理。
# 你也可以直接显式指定某个 state：
# EA_MODEL_PATH=/path/to/checkpoints/eagle3_qwen25_head/state_39 bash scripts/run_bird_eagle3_full_eval.sh

# 5.3 兼容：位置参数模式（旧方式仍可用）
bash scripts/run_bird_eagle3_full_eval.sh \
  /path/to/Qwen2.5-Coder-14B-Instruct \
  /path/to/checkpoints/eagle3_qwen25_head \
  /path/to/bird/dev_20240627/dev.json \
  /path/to/bird/dev_20240627/dev_databases \
  /path/to/workdir/full_eval \
  15
```

## 6A. 审核用细粒度流程（Data -> Generate -> SpecDecode -> Eval）

本节用于“代码级审核”，按真实实现拆成 4 层。每层都给出：
1. 输入
2. 关键处理逻辑
3. 输出
4. 审核检查点

### 6A-1. 数据处理过程（prep）

代码入口：
`eagle/text2sql/bird/prep_bird.py`

输入：
1. `--input-json`：BIRD 原始样本（list 或 `{"data": [...]}`）
2. `--db-root`：数据库根目录（可选，但建议必填）
3. `--max-db-desc-chars`：数据库描述裁剪上限

关键处理逻辑：
1. 字段归一化：兼容 `question/nl/utterance`、`evidence/external_knowledge/hint/hints`、`SQL/sql/query` 等多键名。
2. 数据库定位：按以下顺序查找 sqlite 文件。
`{db_root}/{db_id}/{db_id}.sqlite` -> `{db_root}/{db_id}.sqlite` -> `{db_root}/{db_id}/*.sqlite`
3. schema 自动构建：
通过 `sqlite_master` 枚举表，再通过 `PRAGMA table_info` 枚举列，生成
`TABLE xxx (col type, ...)` 文本。
4. database_description 自动构建：
读取 `{db_root}/{db_id}/database_description/*.csv`，抽取列描述、数据格式、值描述。
5. 文本清洗：
去 BOM、压缩空白、空值回退。
6. 统计空字段：
输出 `empty_schema_context`、`empty_evidence`、`empty_database_description` 计数。

输出：
每行一个 JSON，核心字段如下。
`question_id, db_id, question, evidence, schema_context, database_description, gold_sql`

审核检查点：
1. `schema_context` 空比例是否异常升高。
2. `database_description` 是否被过度截断。
3. `gold_sql` 是否存在空值或明显脏数据。

### 6A-2. 生成过程（inference）

代码入口：
`eagle/evaluation/gen_ea_answer_qwen3_bird.py`

输入：
1. `question_file`：6A-1 的输出 JSONL
2. `base_model_path`：基础模型
3. `ea_model_path`：EAGLE head 目录
4. 解码参数：`temperature/max_new_token/total_token/depth/top_k`

关键处理逻辑：
1. Prompt 构建：
`prompt_builder.py` 使用固定 `SYSTEM_PROMPT` + 分段 user prompt。
分段为 `[DB_ID]/[SCHEMA]/[DATABASE DESCRIPTION]/[EVIDENCE]/[QUESTION]/[OUTPUT]`。
2. 聊天模板：
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` 生成最终输入。
3. warmup：
对首条样本先跑一次 `model.eagenerate` 预热 KV/cache 路径。
4. 单样本生成：
`generate_once` 返回
`raw_text/new_token/stats/wall_time/prompt_tokens`。
5. SQL 后处理：
`postprocess_sql.extract_sql` 强制提取首条 SQL 语句，若无 SQL 起始关键字则判无效。
6. 无效重试：
默认 `retry_invalid_sql=True`，使用严格重试提示再生成一次。
7. 结果落盘：
每条样本写入 `pred_dev.jsonl`，包含预测 SQL 和加速统计字段。

输出（核心字段）：
`pred_sql, raw_output, new_tokens, tree_steps, accepted_tokens, proposed_tokens, acceptance_rate, wall_time, prompt_tokens, is_valid_sql, invalid_reason, retry_used, final_attempt`

审核检查点：
1. `is_valid_sql` 比例是否合理。
2. `retry_used` 占比是否过高（提示 prompt 或 schema 问题）。
3. `raw_output` 与 `pred_sql` 差异是否异常（后处理过于激进或模型输出噪声过大）。

### 6A-3. 投机解码过程（speculative decoding）

核心代码：
`eagle/model/ea_model.py` 中 `EaModel.eagenerate`
`eagle/model/utils.py` 中 `initialize_tree/tree_decoding/evaluate_posterior/update_inference_inputs`

算法流程（与实现对应）：
1. Prefill：
基础模型先前向，得到首 token 分布。
2. Draft 树生成：
`ea_layer.topK_genrate` 基于隐藏状态构建 draft tree（由 `total_token/depth/top_k` 控制宽度和深度）。
3. Target 验证：
`tree_decoding` 用基础模型对候选路径并行打分。
4. 接受判定：
`evaluate_posterior` 在 `temperature=0` 下走 greedy 接受规则：
按位置比较候选 token 是否等于 target argmax，取最长前缀接受。
5. 状态更新：
`update_inference_inputs` 把接受前缀并入输入，拷贝对应 KV 区段，继续下一轮树扩展。
6. 统计累加：
`proposed_tokens += 候选树中提议 token 数`
`accepted_tokens += 本轮接受长度 + 1`
最终 `acceptance_rate = accepted_tokens / proposed_tokens`

停止条件：
1. 命中 EOS / eot
2. `new_token > max_new_tokens`
3. 序列超 `max_length`

审核检查点：
1. `acceptance_rate` 是否与速度提升趋势一致。
2. `proposed_tokens_sum` 很大但 `accepted_tokens_sum` 很小，通常表示 draft 分布与 target 偏差大。
3. `tree_steps` 异常大且 wall_time 高，需检查 `depth/top_k/total_token` 配置。

### 6A-4. Evaluation 过程（EX + 接受率聚合）

代码入口：
`eagle/text2sql/bird/eval_exec.py`

输入：
1. `question-jsonl`：含 `gold_sql`
2. `pred-jsonl`：含 `pred_sql` 与投机解码统计
3. `db-root`：sqlite 数据库目录

关键处理逻辑：
1. 样本对齐：
按 `question_id` 将预测与题目索引对齐。
2. DB 定位：
优先显式 `db_path` 字段，否则按 `db_id` 自动查找 sqlite。
3. SQL 执行：
以 `sqlite3` 只读模式执行，支持 `signal.setitimer` 超时中断。
4. 结果比较：
默认 `--ignore-row-order`，先排序后比较结果集。
5. EX 口径：
`ex_denominator = gold 可执行样本数`
`ex_matches = gold/pred 都可执行且结果集一致`
`exec_accuracy = ex_matches / ex_denominator`
6. 接受率聚合（来自 pred jsonl）：
`acceptance_rate_mean`：样本均值
`acceptance_rate_token_weighted = accepted_tokens_sum / proposed_tokens_sum`
同时统计 `wall_time_avg_sec`。
7. 失败分类：
`question_not_found/db_not_found/gold_sql_empty/pred_timeout/pred_syntax_error/exec_mismatch/...`

输出产物：
1. `eval_summary.json`
2. `eval_failures.jsonl`
3. `eval_report.md`

审核检查点：
1. `ex_denominator` 是否接近预期样本数（否则 gold 执行质量有问题）。
2. `pred_executable_rate` 与 `exec_accuracy` 差距过大时，重点看 `exec_mismatch`。
3. 失败集中在 `pred_syntax_error` 时优先看 prompt 和后处理。

### 6A-5. 一键链路产物追踪（建议审核顺序）

脚本入口：
`scripts/run_bird_eagle3_full_eval.sh`

阶段与产物：
1. Stage-1 `prep_bird` -> `bird_dev_infer.jsonl`
2. Stage-2 `gen_ea_answer_qwen3_bird` -> `pred_dev.jsonl`
3. Stage-3 `eval_exec` -> `eval_summary.json/eval_failures.jsonl/eval_report.md`
4. Stage-4 控制台打印关键指标（EX、接受率、token 加权接受率）

建议审核顺序：
1. 先看 `eval_summary.json` 的全局指标。
2. 再看 `eval_failures.jsonl` 的 Top reason。
3. 最后回溯 `pred_dev.jsonl` 里的 `raw_output/pred_sql/retry_used/acceptance_rate`。

### 6A-6. 单独 coding 验证脚本（对应审核意见 1/2）

#### A) 生成 prompt 引导能力验证（审核意见 1）

目的：
1. 完全基于当前 `SYSTEM_PROMPT + build_bird_user_prompt` 做小批量直推。
2. 使用 Qwen2.5-Coder-14B（非 EAGLE）验证“能否稳定产出标准 SQL”。
3. 可选联动 6A-4 evaluation，直接看 EX 与失败分类。

脚本：
`eagle/text2sql/bird/validate_prompt_smallbatch.py`

示例命令（推荐）：
```bash
cd EAGLE-main
python -m eagle.text2sql.bird.validate_prompt_smallbatch \
  --base-model-path /path/to/Qwen2.5-Coder-14B-Instruct \
  --question-jsonl /path/to/workdir/bird_dev_infer.jsonl \
  --output-dir /path/to/workdir/prompt_validate_smallbatch \
  --num-samples 20 \
  --temperature 0 \
  --max-new-tokens 128 \
  --run-eval \
  --db-root /path/to/bird/dev_20240627/dev_databases \
  --eval-timeout-sec 15
```

关键输出：
1. `prompt_smallbatch_pred.jsonl`：直推 SQL 预测
2. `prompt_smallbatch_summary.json`：`valid_sql_rate/invalid_breakdown` + 可选 `eval_summary`
3. `prompt_eval_summary.json`：若 `--run-eval` 则生成

审核建议阈值（可按项目调整）：
1. `valid_sql_rate` 先达到可接受基线（例如 >0.8）
2. `invalid_breakdown` 中 `missing_sql_keyword` 不应占主导
3. 若 `pred_executable_rate` 明显低于 `valid_sql_rate`，优先检查后处理与 schema 注入

#### B) evaluation 严格逻辑验证（审核意见 2）

目的：
1. 用最小正负样本严格覆盖 4 个环节：
`DB 定位 / SQL 执行 / 结果比较 / 失败分类`
2. 同时验证 `ignore_row_order` 与 `strict_row_order` 两种比较口径。

脚本：
`eagle/text2sql/bird/validate_eval_exec_minimal.py`

示例命令：
```bash
cd EAGLE-main
python -m eagle.text2sql.bird.validate_eval_exec_minimal
```

可选：保留中间产物便于人工复核
```bash
python -m eagle.text2sql.bird.validate_eval_exec_minimal \
  --output-dir /path/to/workdir/eval_logic_validate
```

当前脚本内置断言覆盖点：
1. `db_not_found`：验证 DB 定位失败分支
2. `pred_syntax_error`：验证 SQL 执行异常分类
3. `exec_mismatch`：验证结果比较分支
4. `question_not_found`：验证样本对齐失败分支
5. 行顺序开关差异：`ignore_row_order` 下匹配，`strict_row_order` 下不匹配

脚本通过标准：
1. 程序退出码为 0
2. 控制台打印 `status=PASS`
3. `ignore_row_order` 与 `strict_row_order` 的 EX 指标符合脚本断言

### 6A-7. 联动一键验证脚本（Prompt直推 + Eval严格逻辑）

目的：
1. 一次执行同时完成：
`Prompt 小批量直推验证` + `Evaluation 严格逻辑验证`
2. 自动聚合关键审核指标，减少手工串联命令。

脚本：
`scripts/run_bird_prompt_eval_audit.sh`

默认运行（内嵌参数）：
```bash
cd EAGLE-main
bash scripts/run_bird_prompt_eval_audit.sh
```

推荐运行（覆盖关键路径）：
```bash
cd EAGLE-main
BASE_MODEL_PATH=/path/to/Qwen2.5-Coder-14B-Instruct \
BIRD_DEV_JSON=/path/to/bird/dev_20240627/dev.json \
BIRD_DEV_DB_ROOT=/path/to/bird/dev_20240627/dev_databases \
WORKDIR=/path/to/workdir/prompt_eval_audit \
NUM_SAMPLES=20 \
TEMPERATURE=0 \
MAX_NEW_TOKENS=128 \
EVAL_TIMEOUT_SEC=15 \
bash scripts/run_bird_prompt_eval_audit.sh
```

关键产物：
1. `WORKDIR/prompt_validate/prompt_smallbatch_summary.json`
2. `WORKDIR/eval_logic_validate/eval_ignore_summary.json`
3. `WORKDIR/eval_logic_validate/eval_strict_summary.json`
4. `WORKDIR/run_bird_prompt_eval_audit.log`

联动脚本默认行为：
1. 若 `QUESTION_JSONL` 不存在，会自动调用 `prep_bird` 构建（可用 `AUTO_PREP_IF_MISSING=0` 关闭）。
2. Prompt 小批量验证默认带 `--run-eval`，会产出小批量 EX。
3. 严格逻辑验证会断言失败分类与 EX 口径，任何不一致直接非 0 退出。

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
