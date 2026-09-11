# Phase 3 Evaluation Protocol（固化版，2026-09-03）

任何 Phase 3 实验改动协议前必须先改本文档并在报告中说明理由。

## 1. 样本定义

- 样本 = (player u, cutoff T, target chart c)；c 在 T 前 u 从未游玩（scorelog 无该 sha256 行）；
- c 的首打发生在 (T, T_end] 内；首打行 = 该玩家 scorelog 中该 sha256 的最早行；
- 排除：course 行（len(sha256)≠64 或 mode≥100）、首打 clear==NO_PLAY(0)、首打 ex==0、
  BP > notes+5、无 manifest 统计的谱面；
- active 玩家名单由 `players.json` 管理（include=true），经 `ingest_player.py` 进入；
- **不变量：每个 (player, chart) 恰好一行**。2026-09-11 修过违反：语料库同一谱文件存在多个路径副本
  （manifest 有 329 个重复 sha256），merge 后样本被双计、且副本时间戳相同会让目标出现在自己的历史
  窗口里。`data.py` 已加不变量断言，回归会直接报错；样本空间随之收缩（12,859 → 12,725 行）；
- **客户端是声明维度（2026-09-11，用户决策：未来区分对待 LR2 与 beatoraja）**：`beatoraja` 与 `lr2`
  **不得混入同一份 player state 特征集**。实测 vsoflan 两个存档共有的 **4,576 张谱面**：两客户端 acc
  差 **sd 10.16pp**、\|差\|>5pp 占 **29.4%**（大于模型自身 acc MAE 6.5），且与 LN 比例相关仅 +0.016
  （是判定/记录差异，非计分口径）；lamp 枚举也不同（beatoraja 在 2/3 插入 ASSIST_EASY / L_ASSIST，
  即同一档位差 +2）。`lr2_reader.py` 提供显式映射仅供对比，**不代表可混用**；
  `data.py` 对 client≠beatoraja 的 include=true 玩家**直接拒绝构建**（守卫，已单测）；
  LR2 档案无首打、无时间戳，只可作 player state；
- **样本空间（用户决定 2026-09-03；2026-09-11 两次扩围）**：目标限定 sl/st/発狂2018 +
  **normal（通常☆）+ overjoy（★★）五表并集**内的谱面——表外谱面质量不可控，作为质量围栏使用；
  表等级本身仍不作特征（特征契约见 chart_repr.py）。**扩围规则**：新表在 `load_tables` 优先级
  **追加在最后**（satellite > stella > insane > normal > overjoy），已有标签**零改动**，
  只有净新增谱面进入；扩围链 12,725 → 13,446（+normal）→ **14,011 行**（train 7,001 / test 7,010，
  +overjoy），**新旧空间的数字不可直接比较**（差异来自样本构成而非模型退化；
  各空间同配置数字见台账）；normal2 与 ln 解析器就绪但**暂屏蔽**（`data.py` admit flag）；
- **推荐工具的候选空间比围栏更窄（2026-09-11，用户决定）**：`recommend.py` 对候选**额外屏蔽
  overjoy**（★★ 分级过不均匀，不宜据此推荐）；仅限推荐侧——overjoy 首打仍作历史与协议目标；
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
- `objective_stats_v2`（+14 维无阈值分布统计，`chart_stats_v2.py`）：**谱面侧当前最优**；
- `perm_space`（22 维排列空间手部位移几何，`chart_perm_space.py`）：**已被 v2 支配**（胜 v1 基线但
  叠加 v2 无增益），2026-09-11 记录，勿重复投入；
- **`HISTORY_RESPONSE_COLS`（24 维个人分轴响应剖面，`history_response.py`）：玩家侧当前最优**
  （11 轴 × resp/slope，每玩家每轴一元 OLS，严格因果前缀和；17/18 玩家改善，已用"同列数打乱配对"
  对照排除维度效应）；
- **`HISTORY_RESPONSE_LAMP_COLS`（24 维 lamp 响应块，同脚本）**：lamp 是 acc 响应块唯一拖累的目标。
  按目标分别加块并**只取 chart-conditioned 一半**（不带 per-axis slope）效果最好——
  **当前最优配置 = `B_resp` = v1 27 + 手工历史 12 + acc 响应块 24 + lamp 响应块(chart-conditioned 13)
  = 76 维：acc 6.501 / cR² +0.434 / lamp 1.227 / QWK 0.710 / BP 119.5，16/18 玩家改善**
  （`response_eval.py` 的 `B_full+lampresp` 行）；
- 新 encoder 一律注册进 `chart_encoder_registry()` 并在同一任务上比较；
- 评价标准唯一：**是否提升对未见 player × chart 首打表现的预测**，不以其与人工
  难度维度的对应性定义价值。

## 4. 目标与指标

| 目标 | 训练形式 | 主指标 | 辅助 |
|---|---|---|---|
| acc | 连续（0-100） | **MAE 与 centered R² 并列为头条**（2026-09-07 用户决定：两者都保留） | R² |
| lamp | 有序 1-9 | ordinal MAE | QWK（quadratic weighted kappa）；regression 式优于 classification 式（3.1 已验证） |
| BP | **log1p(raw BP)** | raw MAE | ratio MAE×1000（BP/notes）作报告视角；MedAE 看长尾 |

- **头条双指标的分工（2026-09-07）**：聚合 MAE 回答"绝对标定"（部署视角），
  centered R² 回答"玩家×谱面 interaction"（研究主问题）。二者缺一不可：
  逐玩家 acc 均值离散度（~10.1）大于模型 MAE（~7.0），nanji 一人占总绝对误差 ~18%，
  故单独引用 MAE 会高估"interaction 进步"；单独引用 centered R² 则丢失绝对精度。
  引用结论时必须同时给出两者。

- NO_PLAY 不作为 lamp 等级；禁止把三目标合成单一分数。

## 5. 必备对照与分析

- H（手工历史统计）、B（chart+history）为固定参照；chart-only（A）作下界；
- **`B_resp` 为当前最优参照**（B + acc 响应块 24 + lamp 响应块 13），已并入
  `compare_nolevel.py` 的常规输出；完整对照与研究见 `response_eval.py` / `PHASE3_6_REPORT.md`；
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
