# Phase 3 Evaluation Protocol（固化版，2026-09-03）

任何 Phase 3 实验改动协议前必须先改本文档并在报告中说明理由。

## 1. 样本定义

- 样本 = (player u, cutoff T, target chart c)；c 在 T 前 u 从未游玩（scorelog 无该 sha256 行）；
- c 的首打发生在 (T, T_end] 内；首打行 = 该玩家 scorelog 中该 sha256 的最早行；
- 排除：course 行（len(sha256)≠64 或 mode≥100）、首打 clear==NO_PLAY(0)、首打 ex==0、
  BP > notes+5、无 manifest 统计的谱面；
- active 玩家名单由 `players.json` 管理（include=true），经 `ingest_player.py` 进入；
- **样本空间（用户决定 2026-09-03）**：目标限定 sl/st/発狂2018 三表并集内的谱面——表外
  谱面质量不可控，作为质量围栏使用；表等级本身仍不作特征（特征契约见 chart_repr.py），
  若未来放开围栏需重新审计表外谱面质量；
- **难度表等级不进入任何训练特征**。

## 2. 时间切分（strict temporal extrapolation）

- 每玩家独立取其首打时间线分位：T_train = q50、T_test = q75；
- train 目标 = (T_train, T_test]，特征只用 ≤ T_train 的历史；
- test 目标 = (T_test, end]，特征用 ≤ T_test 的全部历史；
- 禁止随机打散；禁止任何来自 target 首打之后的信息进入特征；
- timestamp 缺失的玩家（LR2）使用 players.json 的 `time: synthetic`（按游玩序号生成
  伪时间），其内部排序有效、跨玩家绝对时间语义无效——含绝对时间的特征
  （h_days_*、h_plays_last30d）在 synthetic 玩家上按序号日解释，报告中必须声明。

## 3. 特征与表征（chart_repr.py 为唯一登记处）

- objective_stats（26 维解析统计 + `c_jrank` 判定窗等级，共 27 维）：当前强基线 encoder；
  c_jrank 于 2026-09-05 加入（消除 tight-rank 谱面 acc 被高估 5-11pp 的系统偏差）；
  已知限制：parser 对未写 #RANK 的谱默认 rank=2；有效判定窗还受玩家 config 影响；
- phase2a_t1_pooled（64 维）：learned encoder 候选；
- 新 encoder 一律注册进 `chart_encoder_registry()` 并在同一任务上比较；
- 评价标准唯一：**是否提升对未见 player × chart 首打表现的预测**，不以其与人工
  难度维度的对应性定义价值。

## 4. 目标与指标

| 目标 | 训练形式 | 主指标 | 辅助 |
|---|---|---|---|
| acc | 连续（0-100） | MAE、R² | **centered R²**（玩家内去均值，检验 interaction） |
| lamp | 有序 1-9 | ordinal MAE | QWK（quadratic weighted kappa）；regression 式优于 classification 式（3.1 已验证） |
| BP | **log1p(raw BP)** | raw MAE | ratio MAE×1000（BP/notes）作报告视角；MedAE 看长尾 |

- NO_PLAY 不作为 lamp 等级；禁止把三目标合成单一分数。

## 5. 必备对照与分析

- H（手工历史统计）、B（chart+history）为固定参照；chart-only（A）作下界；
- **centered R²** 必报：interaction 的主判据；
- **fixed-chart 多玩家子集**（test 内 ≥2 玩家首打的谱面）必报；
- nanji（及未来任一分布外玩家）单独报告全部目标；
- 逐玩家指标必报，aggregate 不得单独作为结论。

## 6. 可信度规范

- C 类（学习型）模型**至少 3 seeds**，报 mean±std；**单 seed 结果不得作为结论**；
- HGB 等确定性模型固定 random_state；
- 每个结果 JSON 必须含 `features_used`（由 chart_repr.feature_manifest 生成）；
- 比较必须在同一行集上进行；跨 scope 比较必须声明样本空间差异。

## 7. 当前已验证的关键结论（改动协议前先读）

1. scorelog 是破纪录日志，首打行精确可靠（PHASE3_AUDIT.md）；
2. 玩家历史 >> 谱面侧信息（acc）；lamp 主要由谱面决定；BP 是两者交互（3.1/3.2）；
3. 难度表特征可被客观 kNN 特征完全替代（3.2 §7/§8）；
4. 手工统计仍是 player state 最强表示；learned encoder 的条件化随数据扩大在改善，
   尚未反超（3.2 §8）。
