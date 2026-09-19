# PHASE4_PROTOCOL.md — Phase 4 协议（生效版本）

> 写于 2026-09-12，Phase 4 第一个 session；第二 session 打了三处补丁（§4 loo_k、§5 玩家内指标、
> §6.3 冷谱面内容基线），并在 §2 修正了 LR2 玩家数。
> **本文是 Phase 4 的唯一协议来源。** `PROTOCOL.md` / `AGENTS.md`（已删除）/`PHASE4_READINESS.md`
> 属 Phase 3 口径，只作数据语义与负结果参考，不得作为 Phase 4 的验收标准。
> `PHASE4_READINESS.md` §2C 曾建议"beatoraja 矩阵 + LR2 矩阵合并分解"——**该建议已被本协议否决**（见 §3）。

## 1. 任务定义

给定玩家在一部分谱面上的当前最佳成绩，预测他在**一张没打过**的谱面上的**稳定/可能最佳成绩**。

预测单元：`(player, client, sha256, best_score, best_lamp, best_bp, notes)`。

- 不预测首打，不用游玩时间，不建因果历史窗口。目标就是"当前最佳"这一横截面量。
- 难度表等级只作坐标/评估，**不作输入特征**。
- 不预先定义速度/耐力/LN 等人工技能轴；谱面与玩家表示只能从 player × chart 成绩矩阵里学。

## 2. 数据契约

`bms_ml/output/phase4/dataset/cross_section.parquet`，由 `build_cross_section.py` 生成。
每行 = 一个 (player, client, sha256) 的当前最佳。构建口径与单位见该脚本 docstring。

关键口径（都已验证，不是假设）：

| 项 | 口径 |
|---|---|
| beatoraja 来源 | `score.db` 的 `score` 表 = **每谱一行当前最佳**（不是游玩日志）。真正的日志 `scorelog.db` 只记"刷新记录"的改善，且分数量纲不同，故不作数据源 |
| beatoraja acc | `(2*(epg+lpg) + (egr+lgr)) / (2*Σ所有判定) * 100`。**必须含 lpg/lgr**：只用 epg/egr 会得到中位 45.6%、最大 304% 的假值；含大型判定后全 18 份存档中位 75.5%、硬上界 100.00 |
| LR2 acc | `(perfect*2 + great) / (2*totalnotes) * 100`，`lr2_reader.py` 已用自己的 `rate` 列验证（最大偏差 0.999） |
| 谱面宇宙 | manifest 中 `has_7k`（7 键可玩）的唯一 sha256 |
| LR2 玩家数 | **2 人**（更正于 2026-09-12）：`PlayerData/LunaticRave2/` 下 `V_soflan.db`（账号 `V_soflan`）与 `mengye.db`（账号 `Caiwla`，文件名不是账号名）。第一 session 误把 `mengye.db` 当成"别人的存档"排除，已修正。每个 roster 条目用 `db` 键钉住唯一文件；**一个条目 = 一份存档 = 一个人**，多文件条目直接报错 |
| 灯标 | 两个客户端的灯标枚举**不同**（LR2 存档只有 0..5，无 FC/PERFECT），各自保留 |
| 备注 | `notes` 用各客户端自记值（主口径）；`acc_alt` 是把同一分数换到 manifest 音符数上的稳健性变体 |
| 谱面客观统计 | `cs_*` 共 26 列，来自语料解析器的客观统计（NPS/音符数分布/BPM/LN 比例等）。**仅用于 §6.3 冷谱面内容基线**，不得作为矩阵补全的输入。列顺序以 `bms_ml/features.py::FEATURE_NAMES` 为准（`features_schema.json` 有 27 个名字对 26 维向量，按它索引会整体错位一格） |

## 3. 红线（违反即结果作废）

1. **LR2 与 beatoraja 禁止混池。** 分数量纲不同（实测同谱差 sd 10.16pp），矩阵、报告、模型全部分开。
2. 不使用任何时间信息（无时间戳、无首打、无因果窗口）。
3. 不使用难度表等级作为输入特征。
4. 不手工定义技能轴。
5. 每个实验必须绑定 held-out 指标；只讲特征解释不算结果。
6. 每个结果 JSON 必须带 data config / client / split / 模型配置 / seed / git commit（由 `provenance.py` 强制）。
7. 每个玩家单独报告；聚合数字单独出现不构成结论。
8. 旧输出不直接引用，需要时重新生成。

## 4. 四固定留出协议

对每个 client 单独执行（`splits.py` 是唯一定义来源，`evaluate_matrix.py` 与 `run_m3.py` 都从它导入）：

| split | 含义 | 作用 |
|---|---|---|
| `random_interaction` | 随机遮盖一部分已观测格子 | 插值能力上限；每个玩家/谱面在训练集中均可见 |
| `loo_k_charts_per_player` (k=1/5/10) | 每位玩家留出 k 张谱 | 小 k 端。**k=1 时 n_test = 玩家数，噪声极大**；补丁 2.1 要求 k=5/10，且 ≥3 seed 报 mean ± std |
| `cold_player` | 整批玩家完全未出现 | 冷启动玩家的可预测性 |
| `cold_chart` | 整批谱面完全未出现 | 冷启动谱面 → 矩阵补全在此理论上无信息（谱面效应恒为 0），只有内容基线能回答 |

- k=1 的分裂沿用原 salt，保证补丁前记录的 split 仍可复现。
- "稳定" = 每个 seed 都赢过门基线均值，**且**增益大于模型自身的 seed 标准差。

## 5. 固定基线与指标

基线（全部只在训练行上拟合）：`global_mean`、`player_mean`、`chart_mean`、
**`player_chart_bias` = 全局均值 + 玩家偏差 + 谱面偏差**（这是**门基线**）、
以及 `player_chart_bias_shrunk`（经验贝叶斯收缩，属模型而非基线，不能当门）。

主指标 MAE / RMSE；辅助：分位数校准、Spearman、Precision@K / NDCG@K。
K ∈ {5, 10, 20}，梯度增益用实际 acc。

### 5.1 玩家内指标（补丁 2.2，必报）

绝对 MAE 主要被"这个玩家有多强"主导（实测跨玩家 MAE 差 4.8pp，远大于任何模型间差距），
所以**不能作为唯一主指标**。每个结果必须同时给：

| 指标 | 定义 | 回答的问题 |
|---|---|---|
| `player_centered_mae` | 测试行的真实值与预测值**都减去该玩家的训练均值**后再算 MAE | 在已知这名玩家多强的前提下，能否判断他在哪张谱上更好/更差 |
| `player_rank_spearman` | 同一玩家内部，模型排序 vs 真实排序的 Spearman（逐玩家后取均值） | 玩家内的排序质量 |
| `player_topk_ndcg` | 同一玩家内部 top-k 的 NDCG（K=5/10/20） | 推荐场景里真正关心的那一段 |

### 5.2 门禁（补丁 2.2 + §6）

M3 要通过，必须**同时**满足：
- 总体 MAE 不能比 `player_chart_bias` 明显变差；
- `player_centered_mae` 与 `player_rank_spearman` **稳定超过** `player_chart_bias`；
- 只有总体 MAE 好一点、玩家内排序不变 → **不算通过**。

## 6. 停止线（M3）

> M3 = 带偏差的矩阵分解 `pred = 全局均值 + 玩家偏差 + 谱面偏差 + <玩家隐向量, 谱面隐向量>`。
> 它必须在 `random_interaction` 与 `loo_k_charts_per_player`(k=5/10) 上**稳定超过**
> `player_chart_bias`，否则停止，不得继续堆表示与模型。

验收五条（全部满足才算过）：

1. `random_interaction`：稳定超过门基线；
2. `loo_k_charts_per_player`（k=5/10，≥3 seed）：稳定超过门基线，不能被 seed 波动淹没；
3. 玩家内三指标（`player_centered_mae` / `player_rank_spearman` / `player_topk_ndcg`）明显超过门基线；
4. `cold_player`：不超过谱面均值基线；
5. 任何一条不满足 → 停止，写负结果，不进入 M4/M5。

补充规则：
- "稳定" = 每个 seed 都赢过门基线均值，且增益大于模型自身 seed 标准差。
- `cold_chart` 与 `cold_player` 不参与主门判定（前者矩阵补全无信息，后者是另一个人群问题）。
- **超参选择不得使用测试集**：dim/reg 在**训练集内部**再切一份验证集上选，按 split 冻结后跑所有 seed。
- 隐向量维度扫描范围 2–64；正则化用验证集选。

### 6.3 冷谱面单独处理（补丁 2.3）

`cold_chart` 用**内容基线**回答，不用矩阵补全强行解释：`content_ridge`，
只用 26 个客观谱面统计（`cs_*`）+ 玩家偏移，**不用难度表等级**（红线 §3.1）。

- 若内容基线也无效 → 报告"当前 BMS 内容不足以做冷谱面"，停下，不进入内容编码器（M4）。
- 若内容基线有效 → 这才构成 M4 内容编码器的入场理由。

## 7. 非目标

- 重打已有谱面的提升预测——需要重复游玩/干预数据，本阶段明确不做。
- 推荐器只推荐**没打过**的谱，不承诺"重打会提高多少"。
- 第一版推荐器建立在成绩预测之上；成绩预测未过门则不做推荐器。

## 8. 结果落盘约定

| 文件 | 内容 |
|---|---|
| `bms_ml/output/phase4/dataset/cross_section.parquet` | 横截面数据（含 26 个 `cs_*` 客观谱面统计） |
| `bms_ml/output/phase4/build_cross_section.json` | 构建口径 + 每玩家读取/丢弃统计 |
| `bms_ml/output/phase4/audit_cross_section.json` | 数据审计（每 client 分列） |
| `bms_ml/output/phase4/evaluate_matrix.json` | 全部 split × 基线 × seed 的指标 |
| `bms_ml/output/phase4/m3_matrix_completion.json` | M3：全部 split × seed + 超参选择轨迹 +（诊断用）测试集网格 |
| `bms_ml/phase4/splits.py` | **split 唯一定义来源**（基线脚本与 M3 都从这里导入） |
| `bms_ml/phase4/metrics.py` | 指标唯一定义来源（含玩家内指标） |
| `bms_ml/phase4/models.py` | 基线、M3（ALS）、冷谱面内容基线 |
| `bms_ml/tests/test_phase4_protocol.py`、`test_phase4_m3.py` | 分裂无泄漏 / 基线只看训练集 / ALS 求解器回归测试 |

## 9. 已知陷阱（用血换来的，别再踩）

1. **`|(y-base) - (p-base)|` 恒等于 `|y-p|`**。玩家内指标必须对**两侧都去均值**，
   否则所谓"玩家内 MAE"就是普通 MAE 的换皮（第一版就踩了，被测试抓到）。
2. **ALS 的两个因子块不能互相写入**。把求解结果写进"固定不动"的另一侧会让两块别名，
   目标函数不降反震荡（train RMSE 6.0 → 1e7）。
3. **单支撑度（某实体只有 1 条观测）的闭式解是 `g·(gᵀr)/(reg+‖g‖²)`**，漏掉 `‖g‖²` 会整体缩放因子。
4. **内层验证选不出正确正则**：内层任务天然更容易（被评的行本来就在拟合范围内），
   所有变体（随机格、整谱、按玩家随机交互、RMSE/MAE）都偏好弱正则 reg≈0.1–0.3，
   而真实测试集偏好 reg≈1–3。因此**不得**把"测试集上最好的网格点"当作协议结果汇报，
   只可作为诊断。
5. **`features_schema.json` 有 27 个名字但 manifest 的 feature 向量只有 26 维**，
   按它索引会整体错位。以 `bms_ml/features.py::FEATURE_NAMES` 为准。
6. **`loo_1_charts_per_player` 对玩家内指标是退化的**（每玩家只留 1 行，去均值后恒为 0），
   该 split 只能看总体 MAE。

## 10. 当前状态

环境：`python -m unittest discover bms_ml/tests` **89 全绿**。

- 数据层、四 split、固定基线、M3（ALS）、冷谱面内容基线均已实现并有回归测试。
- **M3 判定：不通过**（§6 五条未满足）。详见 `PHASE4_M3_REPORT.md`。
- **Phase 4 的"协同过滤 / 矩阵补全"路线在当前数据上记为负结果**：
  带偏差的矩阵分解没有稳定超过门基线，玩家内排序没有改善，
  即使拿测试集作弊调参也不算通过。
- 按停止线，**不得**开始内容编码器或推荐器——除 §11 明确放行的 Phase 4B 冷谱面分支外。

## 11. Phase 4B / cold_chart（独立分支，2026-09-12 用户决定）

### 11.1 定性

- **M3 是失败的**（见 §10）。**Phase 4B 是一个破例放行的独立 track，不是 M4 通过。**
  名字固定为 `Phase 4B / cold_chart`；不得写成"矩阵补全路线成立"，也不得暗示 M3 通过。
- 放行依据只有一个：`cold_chart` 上内容基线是稳定阳性（§11.3）。

### 11.2 范围

- 只研究**完全没人打过的新谱面**（`cold_chart`）。
- **不用矩阵补全解释冷谱面**（谱面在训练集里没有观测，矩阵无信息是数学事实）。
- LR2 与 beatoraja **继续分开**报告，不混池。
- **不手工定义速度/耐力/LN 技能轴。**
- 可以用现有客观统计（26 个 `cs_*`）与 MSD 七轴作为基线或输入，
  但**不得把它们说成最终谱面表示**——它们只是"目前能拿到的东西"。

### 11.3 固定基线（后续任何模型必须和 content_ridge 比，而不是只和 6.749 比）

| 基线 | 定义 | cold_chart MAE（beatoraja, 3 seed） |
|---|---|---|
| 退化基线 | 玩家均值（谱面偏差恒为 0） | **6.749** |
| `content_ridge` | 26 个客观统计 + 玩家偏移，α=1.0 | **6.243** |

**必须胜过 6.243。** 只赢 6.749 不叫过门。

### 11.4 两道门

**第一道门：谱面级内容预测（难度）**
- 预测目标：谱面的**平均表现**（去掉玩家构成后的谱面效应）。
- 必须稳定超过 `content_ridge`；≥3 seed；**提升幅度必须大于 seed 波动**。

**第二道门：玩家 × 谱面交互**
- **不能只做"玩家均值 + 预测谱面难度"。**
- 必须测试"谱面内容能否针对不同玩家给出不同预测"（玩家对内容特征的权重不同）。
- 必须看**玩家内 MAE / 玩家内 Spearman / top-k 排序**。
- **只提升谱面级 MAE、玩家内排序不变 → 判定为"只是难度预测，不是交互"，第二道门不过。**

### 11.5 允许的模型顺序（每加一层复杂度都必须胜过上一层对应基线）

1. ridge / GBDT 在客观统计 + MSD 上做冷谱面基线；
2. 只有当简单内容模型证明有信号后，才尝试从 note sequence / 原始事件学谱面表示；
3. **不先上深度学习**；
4. 任何一层没过对应基线就停下，不加下一层。

### 11.6 推荐器

- **Phase 4B 未过两道门前，不启动 M5 / 推荐器。**
- 两道门都失败 → 直接写负结果，结束 Phase 4B。

### 11.7 Option C（内层验证为什么偏好弱正则）

- 只作为**时间盒内的小诊断**，不单独作为主线方向。
- 若 Phase 4B 也遇到同样的验证/调参失配，再优先查它。
