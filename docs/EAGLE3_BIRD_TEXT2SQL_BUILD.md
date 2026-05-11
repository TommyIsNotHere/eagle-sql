# EAGLE + BIRD Text-to-SQL 持久化文档

## 0. 文档定位

本文件是本项目的唯一持久化记忆，用于沉淀：
1. 项目目标与边界
2. 当前真实实现（以仓库代码为准）
3. 历史故障与修复策略
4. 当前焦点与下一步动作
5. 协作方式与环境规格

维护原则：
1. 只保留当前仍然有效的信息，过期内容必须删除。
2. 每次 Web IDE 实验后，必须回填结果到本文件（进度看板/故障台账/当前焦点）。
3. 新会话开始前，先阅读本文件再动手。

---

## 1. 工作方式约定

1. **环境基准**：编写代码时以本文档「2. 环境规格」中的 Web IDE 环境为准，不得假设本地路径或依赖版本。
2. **禁止猜测修复**：没有真实报错输出时，不得凭推测修改代码。若同一问题连续两轮未解决，切换为诊断模式（加日志、打印中间变量、缩小复现范围）。
3. **一致性优先**：修改任何模块前，先阅读该模块及其上下游接口的现有实现，确保新改动在数据流、命名、参数传递上与已有逻辑一致。
4. **不做无关重构**：除非任务明确要求，不扩大修改范围。涉及多文件联动时先列计划再执行。
5. **历史代码警觉**：本项目经历多轮 AI 修改，核心模块（`main.py`、`cnets.py`、`data_utils.py`、训练脚本）可能存在接口不匹配、逻辑未闭合等遗留问题。接触这些文件时先做一致性扫描，发现矛盾应主动指出而非绕过。

---

## 2. 环境规格（Web IDE 基准）

以下规格来自 2026-04-29 的真实环境输出，作为代码与排障基线。

### 2.1 基础软硬件

1. 操作系统：Ubuntu 22.04.4 LTS
2. Python：3.11.11
3. NVIDIA Driver：535.54.03
4. CUDA Version（nvidia-smi）：12.4
5. GPU：H800 x2
6. 内存：总 1.8Ti，已用 50Gi，可用约 1.8Ti
7. Swap：0

### 2.2 存储与持久化约束

1. `/dev/shm`（tmpfs，72G）：容器内临时高速区，容器回收后数据丢失。
2. `/mnt/nj-aigc/usr/wangtong2`（JuiceFS 挂载）：跨容器持久化，项目正式产物必须落这里。
3. 当前项目代码目录 `/mnt/nj-aigc/usr/wangtong2/eagle_sql` 本质位于远程挂载，不是本机 NVMe 直连盘。

### 2.3 关键路径约定

1. 项目根：`/mnt/nj-aigc/usr/wangtong2/eagle_sql`
2. 代码根：`/mnt/nj-aigc/usr/wangtong2/eagle_sql/EAGLE-main`
3. **原版参考代码**：`EAGLE-main-original/`（EAGLE 官方仓库快照，**只读**，用于行为对照与算法校验，不参与训练推理）
4. 基座模型：`/mnt/nj-aigc/usr/wangtong2/eagle_sql/model/Qwen2.5-Coder-14B-Instruct`
5. BIRD train：`/mnt/nj-aigc/usr/wangtong2/eagle_sql/bird/train`
6. BIRD dev：`/mnt/nj-aigc/usr/wangtong2/eagle_sql/bird/dev_20240627`
7. 训练 artifacts 默认：`/mnt/nj-aigc/usr/wangtong2/eagle_sql/artifacts`

### 2.4 核心依赖（动态）

1. 训练脚本已做运行前预检：`pydantic`、`deepspeed`、`datasets`、`transformers`、`torch`。
2. 版本以 Web IDE 实际 `pip show` 为准；如出现环境变更，优先更新本节。


---

## 3. 当前焦点（2026-05-11）

**Stage A 训练已完成；ShareGPT/Alpaca/HumanEval 通用任务推理验证已通过；下一步在 BIRD text-to-SQL 任务上跑 Stage A checkpoint 的推理验证，作为 Stage B 训练的基线。**

整体推进顺序：
```
[已完成] Stage A 训练 (ShareGPT)
   └─ Eval Acc 84.27%, Top-1 K-acc 0.840

[已完成] Stage A 通用任务推理验证
   ├─ ShareGPT       : Chain α=0.883, Tree τ=4.59, speedup 3.13x
   ├─ Alpaca         : ✅ 通过（符合预期）
   └─ HumanEval      : ✅ 通过（符合预期）

[当前焦点] Stage A 在 text-to-SQL (BIRD) 上的推理验证 ← 我们在这里
   └─ 用通用 head 直接跑 BIRD dev，看 SQL 域下的 α/τ/EX 基线

[未开始] Stage B 训练 (BIRD SQL domain adaptation)
[未开始] Stage B 推理 + EX 评测
```

- Checkpoint：`artifacts/eagle2/checkpoints/state_20`，推理产物 `artifacts/eagle2/infer/`
- Stage A → Stage B 的判定依据：BIRD Stage A 基线 α/τ + EX 数据，作为 Stage B 训练后改进幅度的参照

---

## 4. 项目目标与边界

### 4.1 总体目标

在 BIRD Text-to-SQL 任务中，形成 EAGLE2 head 的完整工程闭环：
1. 训练：Qwen2.5-Coder-14B-Instruct + EAGLE2 head（两阶段：ShareGPT → BIRD SQL）
2. 推理：speculative decoding 生成可执行 SQL
3. 评测：输出 EX、可执行率、acceptance 等指标

### 4.2 当前边界

1. 主线切换为 EAGLE2，`traineagle3/` 代码暂挂不删。
2. 第一阶段用 ShareGPT 验证训练 pipeline，指标对齐后再接 SQL fine-tuning。
3. 文档聚焦”已实现 + 已验证路径”，不记录纯设想方案。

---

## 5. EAGLE2 vs EAGLE3 差异对比与切换方案

### 5.1 训练架构对比

| 维度 | EAGLE2 (`eagle/train/main.py`) | EAGLE3 (`eagle/traineagle3/main.py`) |
|---|---|---|
| 框架 | `accelerate` | DeepSpeed (ZeRO) |
| 数据格式 | 预抽取 `.pt`（hidden_state + input_ids + loss_mask） | 原始 JSONL → 在线/离线 tokenization |
| 数据准备 | 先跑基座 forward 将 hidden states 存盘 | 训练时在线过 teacher 取 hidden states |
| Head 结构 | `fc(2h→h)` + N 层 LlamaDecoderLayer | `fc(3h→h)` + 单层 midlayer |
| 损失函数 | SmoothL1(velocity) + CE(prediction)，v_w=1.0 / p_w=0.1 | Hybrid: gold CE + distillation |

### 5.2 推理对比

| 维度 | EAGLE2 | EAGLE3 |
|---|---|---|
| Draft vocab | 全词表 | 子集词表 + d2t/t2d 映射 |

### 5.3 切换方案

| 阶段 | 动作 | 说明 |
|---|---|---|
| Phase 0 | 新写 `ge_data_qwen25.py` | EAGLE2 核心前置：基座 forward 抽取 hidden states 存 .pt |
| Phase 1 | 新建 `eagle/train2/main.py` | 从原始 `train/main.py` 复制并适配 Qwen2.5 路径与超参 |
| Phase 1 | 复制 `cnets1.py` 到 `eagle/model/` | EAGLE2 训练模型结构（已在 original 中，当前 EAGLE-main 缺失） |
| Phase 1 | 新建 `scripts/run_eagle2_train.sh` | 复用现有环境预检，训练命令改为 `accelerate launch` |
| Phase 2 | 修改推理脚本 `use_eagle3=False` | `ea_model.py` 已内置切换逻辑，改动最小 |
| 暂挂 | `traineagle3/` 目录保留不删 | 后续如需回退有据可依 |

### 5.4 两阶段训练策略

```
━━━ Stage A: ShareGPT 通用预训练 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  目标：训练对齐 EAGLE 论文，head 在通用对话域学到 draft 能力
  数据：ShareGPT 通用对话（~68k 样本）
  超参：20 epoch, lr=3e-5, warmup_ratio=0.03
  结果：✅ 完成，Eval Acc 84.27%, Top-1 K-acc 0.840

━━━ Stage A 推理验证（多任务） ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  目的：验证 head 工程闭环可用 + 拿到 Stage B 前的基线指标
  在通用任务上：
    ✅ ShareGPT  (eval_sharegpt.py) — Chain α=0.883 / Tree τ=4.59 / 3.13x
    ✅ Alpaca    (eval_bench.py)    — 符合预期
    ✅ HumanEval (eval_bench.py)    — 符合预期
  在 text-to-SQL 任务上（当前进行）：
    🔄 BIRD dev  (gen_ea_answer_qwen3_bird.py + eval_exec.py)
       目标：用 Stage A 通用 head 跑 BIRD dev，得到 α/τ/EX 基线
       意义：Stage B 训练后对比此基线，量化 SQL domain adaptation 增益

━━━ Stage B: BIRD SQL Domain Adaptation ━━━━━━━━━━━━━━━━━━━━━━━
  前置：Stage A 在 BIRD 上的基线指标已采集
  目标：在 Stage A 基础上，让 head 适应 SQL token 分布
        → α/τ 上升 + EX accuracy 提升
  数据：BIRD train chat JSONL（复用现有 prep → SFT 管线）
  超参：40-60 epoch, lr=1e-5~2e-5, warmup_ratio=0.05~0.1
  状态：⬚ 未开始

━━━ Stage B 推理验证 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  脚本：同 Stage A BIRD 推理流程（同一 `run_bird_eagle3_full_eval.sh`）
  对比：Stage B 的 α/τ/EX vs Stage A 同任务基线
  状态：⬚ 未开始
```

---

## 6. 端到端流程图（切换前后对比）

### 6.1 旧流程（EAGLE3，已暂挂）

```
BIRD JSON → prep_bird.py → build_bird_eagle3_sft.py → chat JSONL
    │
    ▼
traineagle3/main.py (DeepSpeed)
  ├─ 在线加载基座 teacher (冻结)
  ├─ fc(3h→h) + 单层 midlayer
  └─ hybrid loss
    │
    ▼  EAGLE3 checkpoint
    │
gen_ea_answer_qwen3_bird.py (use_eagle3=True) → eval_exec.py
```

### 6.2 新流程（EAGLE2，当前主线）

按时间顺序展开，左侧为 Stage A 路径，右侧为 Stage B 路径，二者共用同一套推理评测脚本。

```
━━━ Phase 0: Hidden States 抽取（每个 Stage 各执行一次） ━━━━━━━━

  Stage A: ShareGPT JSONL          Stage B: BIRD chat JSONL
        │                                │
        └──────────┬─────────────────────┘
                   ▼
  ┌──────────────────────────────────────────────┐
  │  ge_data_qwen25.py                           │
  │  基座 forward → 逐样本存 .pt                   │
  │  (hidden_state, input_ids, loss_mask)        │
  └──────────────────────────────────────────────┘
                   │
                   ▼  data/{sharegpt,bird}/*.pt

━━━ Phase 1: EAGLE2 训练 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  run_eagle2_train.sh → accelerate launch train2/main.py
                   │
                   ▼
  ┌──────────────────────────────────────────────┐
  │  train2/main.py (accelerate + 多卡)          │
  │  ├─ 加载 .pt hidden states                   │
  │  ├─ cnets1.Model: fc(2h→h) + N层 decoder    │
  │  ├─ SmoothL1(velocity) + CE(prediction)     │
  │  └─ checkpoint 落盘 state_X/                 │
  └──────────────────────────────────────────────┘
                   │
                   ▼

━━━ Phase 1.5: Checkpoint 格式转换 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  state_X/model/  →  artifacts/eagle2/infer/
                       ├─ config.json
                       └─ pytorch_model.bin

━━━ Phase 2: Stage A 推理验证（多任务） ━━━━━━━━━━━━━━━━━━━━━━━

  base_model + Stage A EAGLE2 head (use_eagle3=False)
                   │
       ┌───────────┼───────────┬───────────────┐
       ▼           ▼           ▼               ▼
  通用对话     Alpaca      HumanEval       text-to-SQL
  (ShareGPT)   (eval_      (eval_         (BIRD dev)
  eval_        bench.py)   bench.py)      gen_ea_answer
  sharegpt.py                              _qwen3_bird.py
       │           │           │               │
       └───────────┴───────────┴───────────────┘
                                 │
                                 ▼  指标：α / τ / speedup
                                       BIRD 额外：EX / 可执行率
                                 │
                            ✅ 通用任务通过
                            🔄 BIRD 进行中（Stage A baseline）

━━━ Phase 3: Stage B 训练 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  返回 Phase 0 → BIRD chat JSONL 抽 hidden states
        →  Phase 1 训练（用 Stage A checkpoint 作为初始化或从零）
        →  Phase 1.5 转换

━━━ Phase 4: Stage B 推理验证 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  base_model + Stage B EAGLE2 head（SQL fine-tuned）
                   │
                   ▼
  ┌──────────────────────────────────────────────┐
  │  gen_ea_answer_qwen3_bird.py → eval_exec.py  │
  │  pred_dev.jsonl                              │
  │  ├─ pred_sql                                 │
  │  ├─ acceptance_rate / mean_accepted_length   │
  │  └─ wall_time                                │
  │            │                                 │
  │            ▼                                 │
  │  对 SQLite DB 执行 pred 与 gold → EX 比对    │
  └──────────────────────────────────────────────┘
                   │
       ┌───────────┴───────────┬───────────────┐
       ▼                       ▼               ▼
  eval_summary.json    eval_failures.jsonl   eval_report.md

  对比：Phase 2 BIRD 基线 vs Phase 4 Stage B 结果 → 量化 SQL adaptation 增益
```

关键产物汇总：

| 阶段 | 产物 | 路径约定 |
|---|---|---|
| Phase 0 Hidden states | `*.pt` | `data/sharegpt/`、`data/bird/` |
| Phase 1 训练 | checkpoint（accelerate 格式） | `artifacts/eagle2/checkpoints/state_X` |
| Phase 1.5 转换 | 推理格式 | `artifacts/eagle2/infer/` |
| Phase 2 Stage A 验证 | 推理输出 + α/τ/speedup | `artifacts/eval_{sharegpt,alpaca,humaneval}/`、`artifacts/bird_dev_full_eval/`（Stage A baseline） |
| Phase 4 Stage B 验证 | `pred_dev.jsonl` + EX summary | `artifacts/bird_dev_full_eval_stageB/` |

---

## 7. 当前进度看板与执行清单

### 7.1 进度看板

**基建（一次性，已稳定）：**

| 状态 | 事项 | 备注 |
|---|---|---|
| ✅ 已完成 | BIRD 数据标准化与 SFT 数据构建链路 | 可复用于 Stage B |
| ✅ 已完成 | 推理与 EX 评测脚本闭环（SQL 路径） | `gen_ea_answer_qwen3_bird.py` + `eval_exec.py` |
| ✅ 已完成 | Prompt / Eval 逻辑审计脚本 | |
| ✅ 已完成 | EAGLE3→EAGLE2 切换改造 | `cnets1.py`、`train2/main.py`、`run_eagle2_train.sh` |
| ✅ 已完成 | `ge_data_qwen25.py` hidden states 抽取脚本 | 通用脚本，按数据源切换即可 |
| ✅ 已完成 | 推理代码 bug 修复 | `cnets1.py` self.config + `ea_model.py` draft_vocab_size |
| ✅ 已完成 | Checkpoint 格式转换（accelerate → 推理） | `artifacts/eagle2/infer/` |
| ✅ 已完成 | 通用任务统一评测脚本 | `eval_bench.py`（alpaca/humaneval/mt_bench） |
| ⏸️ 暂挂 | EAGLE3 训练全套（traineagle3/） | 保留代码不删，暂不使用 |

**Stage A（ShareGPT 通用预训练 + 多任务验证）：**

| 状态 | 事项 | 备注 |
|---|---|---|
| ✅ 已完成 | ShareGPT hidden states 抽取 | |
| ✅ 已完成 | ShareGPT Stage A 训练 | Eval Acc 84.27%, Top-1 K-acc 0.840 |
| ✅ 已完成 | Stage A 推理验证 - ShareGPT | Chain α=0.883, Tree τ=4.59, speedup 3.13x |
| ✅ 已完成 | Stage A 推理验证 - Alpaca | 符合预期 |
| ✅ 已完成 | Stage A 推理验证 - HumanEval | 符合预期 |
| 🔄 进行中 | **Stage A 推理验证 - BIRD dev** | **当前焦点**：用通用 head 跑 BIRD，拿 Stage B 前基线 |

**Stage B（BIRD SQL domain adaptation）：**

| 状态 | 事项 | 备注 |
|---|---|---|
| ⬚ 未开始 | BIRD SQL hidden states 抽取 | 复用 `ge_data_qwen25.py` 换数据源 |
| ⬚ 未开始 | BIRD SQL Stage B fine-tuning | 前置：Stage A BIRD 基线已采集 |
| ⬚ 未开始 | Stage B 推理 + EX 评测 | 同 Stage A BIRD 流程，对比基线 |

### 7.2 执行优先级

1. **Stage A BIRD 推理基线（当前）** — `run_bird_eagle3_full_eval.sh` 在 Stage A checkpoint 上跑 BIRD dev，记录 α/τ/EX/可执行率，作为 Stage B 改进的对照基线。
2. **Stage B 数据准备** — `ge_data_qwen25.py` 换 BIRD chat JSONL，抽 SQL 域 hidden states。
3. **Stage B 训练** — 同 pipeline，SQL domain adaptation（lr ↓, epoch ↑, warmup ↑，参见第 8 节）。
4. **Stage B 评测** — 重跑 BIRD full eval，对比第 1 步基线，量化 SQL 适配增益。

---

## 8. 训练超参推荐

EAGLE 原文默认：EAGLE2 20 epoch，EAGLE3 40 epoch（均基于 ShareGPT ~68k 样本）。Epoch 数应根据数据规模调整——核心依据是总优化步数是否足够收敛。

| 参数 | ShareGPT（~68k 样本） | BIRD SQL（~9.4k 样本） |
|---|---|---|
| epoch | 20 | 40–60 |
| lr | 3e-5 | 1e-5 ~ 2e-5 |
| warmup_ratio | 0.03 | 0.05 ~ 0.1 |
| 策略 | 当前设置即可 | 数据量小，降 lr + 加 warmup 防过拟合；监控 eval loss，连续 5 epoch 不降则 early stop |

依据：`micro_batch=1, grad_accum=2` 下，ShareGPT 20 epoch ≈ 68 万步（已充分收敛），BIRD SQL 20 epoch 仅 ~9.4 万步（偏少）。SQL token 分布集中，需更多 epoch 学习 draft 模式，但过拟合风险高，必须配合降 lr。

---

## 9. Stage A 推理验证基线

### 9.1 ShareGPT（state_5，100 条随机样本，seed=42）

| 模式 | DEPTH | TOP_K | TOTAL_TOKEN | 核心指标 | wall_speedup |
|---|---|---|---|---|---|
| Chain (314) | 3 | 1 | 4 | **α = 0.883** | 2.09x |
| Tree (EAGLE2 默认) | 6 | 10 | 50 | **τ = 4.588** | 3.13x |

辅助指标：prefix_match_ratio ≈ 0.48，exact_match_ratio = 0.09~0.16，quality_issue_ratio = 0。

指标解读：
- Chain `acceptance_rate` 即 α（per-token 接受概率），衡量 head 质量
- Tree `acceptance_length` 即 τ（每次 base forward 接受 token 数），衡量端到端价值
- bf16 推测解码存在已知数值漂移，prefix_match 50% 左右属正常范围

**数据局限**：随机样本来自 `sharegpt_train.jsonl` 全量，与训练数据存在 ~95% 重叠（训练时按 95/5 顺序切分），指标含 in-distribution 偏高。Stage B 评测须使用真正 held-out 数据集。

### 9.2 Alpaca / HumanEval

- Alpaca：✅ 通过，加速指标符合预期
- HumanEval：✅ 通过，加速指标符合预期
- 评测脚本：`eval_bench.py` + `run_eval_alpaca.sh` / `run_eval_humaneval.sh`
- 默认 Chain 配置（top_k=1, depth=3, total_token=4），主指标 α
- 详细数据落在 `artifacts/eval_alpaca/`、`artifacts/eval_humaneval/`

### 9.3 BIRD（Stage A 基线 — 进行中）

待回填：
- α (chain) / τ (tree)
- EX accuracy
- 可执行率（pred_executable_rate）
- 失败模式 breakdown

跑通后，这一节作为 Stage B 评估增益的对照参照。

### 9.4 阶段性结论

Stage A head 在通用对话域（ShareGPT/Alpaca/HumanEval）已达 EAGLE2 论文水平的加速能力（speedup 2-3x）。当前在 SQL 域的 baseline 采集是 Stage B 训练策略选型的依据。

---

## 10. 故障台账

| 日期 | 问题 | 根因 | 修复 |
|---|---|---|---|
| 2026-05-08 | 推理加载 `self.ea_layer.config` 崩溃 | `cnets1.Model.__init__` 缺少 `self.config = config` | 补上赋值 |
| 2026-05-08 | EAGLE2 config 无 `draft_vocab_size` 导致 `AttributeError` | `ea_model.py` 直接访问不存在的属性 | 改为 `getattr` 回退到 `vocab_size` |
| 2026-05-08 | Checkpoint 格式不匹配 | `accelerator.save_state()` 存 `state_X/model/` 目录结构，推理期望 `config.json` + `pytorch_model.bin` 平铺 | 手动转换到 `artifacts/eagle2/infer/` |

---

## 11. 变更记录

1. 2026-04-28：首次重构为持久化文档，清理冗余与过时内容。
2. 2026-04-29：按协作规范补齐「工作方式约定」「环境规格」「当前焦点」「触发条件台账」「模块 I/O 契约」「路径索引」。
3. 2026-05-06：方向转换 EAGLE3→EAGLE2，重写第 3-7 节（焦点、目标、差异对比、流程图、进度看板）。删除原模块契约/故障台账/运行策略/路径索引（已过时）。
4. 2026-05-08：Stage A 训练完成。更新焦点为推理验证；流程图加入 Phase 1.5（checkpoint 转换）和 Phase 2a（Stage A 验证）；补充训练记录和故障台账；进度看板全面对齐实际状态。
5. 2026-05-11：Stage A 推理验证完成。补充 9.2 节推理验证结果（Chain α=0.883 / Tree τ=4.59 / speedup 3.13x）；进度看板和执行优先级前移至 Stage B；记录测试数据与训练集 ~95% 重叠的方法学局限。
6. 2026-05-11：补充 Stage A 在 Alpaca/HumanEval 上的多任务验证（通过）；重写 3/5.4/6.2/7 节，把"Stage A 推理验证"拆为「通用任务（已完成）」+「BIRD 基线（进行中）」；当前焦点改为 BIRD Stage A 基线采集（用于对照 Stage B 增益）；删除 9.1 训练 epoch 流水（信息密度低）。
