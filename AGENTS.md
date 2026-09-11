# AGENTS.md — BMS Player–Chart Performance 项目工作指南

> 给未来 agent/session 的最小必读。读完后按需深入，不要凭常识猜测项目约定。

## 项目是什么

研究型项目：**给定玩家截至 T 的游玩历史与一张他没玩过的 BMS 谱面，预测其首打表现
（acc / lamp / BP 三目标分开）**，最终目标是可验证的 player state 表示。
研究工程，不是产品开发；负结果同样是交付物。

## 当前状态一句话（2026-09-11 晚；**Phase 3 已封存**，tag `phase3-complete`，Phase 4 计划见 `PHASE4_READINESS.md`——新方向先查其 §1 负结果清单，勿重复投入）

18 名玩家、协议口径（全 scope，**五表围栏 sl/st/発狂2018/normal☆/overjoy★★**、c_jrank 在列）下基线
（**train 7,001 / test 7,010**，14,011 行；2026-09-11 两次扩围 + 语料库去重后）：
A 11.724 / H 7.619 / **B 6.901**（acc MAE）。
**当前最优 = `B_resp+dev` + MSD 轴（180d 半衰期）：acc **5.801** / cR² +0.463 / lamp 1.102 / BP 109.9**
**（扩围历史：12,725 → 13,446（normal☆）→ 14,011（+overjoy★★）行；normal2/ln 解析器就绪但按用户
决定暂屏蔽；**新旧空间数字不可直接比较**——各空间同配置数字见台账，MSD 增益在最新空间复验
+0.259 p=4e-15）**
（= v1 27 + 手工历史 12 + acc/lamp/BP 响应块 + 11 个 `h_dev_*` + **7 个 MSD 技巧项的响应/偏差块 `m_*`**；
链路（旧空间口径）：B 7.009 → +响应块 6.495 → +180d 遗忘 6.360 → +dev 6.117 → **+MSD 轴 5.893**。
MSD 配对 +0.224（CI 0.164..0.286，p=1.4e-12），打乱对照 6.14~6.18 → 信息非容量；
**裸 MSD 列反而有害（6.250）** —— 人群难度本身不预测成绩，玩家相对化之后才有价值；
**`B_msdonly`（62 特征，MSD 块替换全部结构响应块）5.996 / cR² +0.460 也优于原 BASE** ——
MSD 是更好的条件化基底；**yangtao 首次改善**（19.20→18.57，msdonly 下 17.45）。
复现：`response_decay.py`（遗忘）、`response_dev.py`（dev）、`response_msd.py`（MSD 轴）；
**半衰期已默认 180d**（`history_response.py`；`P3_RESP_HALF_LIFE=uniform` 可复现旧的整档等权，
数字则变为 6.495/6.360 一系——引用历史数字时注意配置）；
`compare_nolevel.py` 的 **`B_full` 行 = 融合最优配置**（从 history_response.parquet 读块），
当前输出 **5.887**，与 `response_msd.py` 精确一致）
次优 `B_full` = 6.583 / +0.430、`B_v2` = 6.796。**个人响应剖面是本阶段唯一有效的表示改进**
（`history_response.py`，11 轴 × resp/slope，按目标分块；**已用"打乱配对"对照排除维度效应**）。
**Cross-player transfer 已验证**（PHASE3_4_TRANSFER.md）：人群预训练+个体 conditioning 的 M2
在 18/18 玩家上胜过 D-local，few-shot 样本效率 3-5×（**这些结果在旧 12,859 行数据集上**，
去重影响量级 ~0.02）。
**Phase 3.5 审阅**（PHASE3_5_REVIEW.md）：B 比平凡基线"玩家均值"（9.997）好 ~3.0（30%）→ 路线健康；
**A 比全局均值（11.43）还差**，不得当下界引用；**C0−B 差距单调收窄 3.42→1.94→1.56→1.25**。
**2026-09-11 已有结论的反转与更正**：① `c0_state_hgb.py` 的 chart 尺度 bug 修复后，"learned state
每个 k 都赢"被推翻（退化为噪声级）；② `perm_space`（排列空间手部几何）胜 v1 但被 v2 支配，勿重复投入；
③ framework paper 的 **felt-time 主张不适用于我们**（样本空间内 NPS 零膨胀）。
所有数字变化见 `EXPERIMENT_LOG.md` 台账。

## 阅读顺序（动代码前必读）

1. `PROTOCOL.md` — 实验协议（样本定义/切分/指标/可信度规范），**改协议必须先改此文件**
2. `EXPERIMENT_LOG.md` — **优化/实验/玩家数据的追加式台账**（看"改过什么、效果如何"先查这里）
3. `PHASE3_AUDIT.md` — 原始存档数据语义（scorelog 的真实含义、所有坑）
4. 最新一期报告（当前：**`PHASE3_6_REPORT.md`** —— 个人响应剖面 + 数据修复 + 五个负结果；
   上一期 `PHASE3_5_REVIEW.md` 的乐观判断已被其更正横幅覆盖）
5. `PHASE3_3_READINESS.md` §7 — 新玩家接入 SOP

## 目录

- `bms_ml/` — Python 包。核心：`parser.py`（BMS 解析）、`phase3/`（当前全部工作）、
  `phase2a/`（旧表征实验）、`output/`（可再生产物，gitignore）
- `bms_ml/phase3/` — 数据管线与实验：
  - **数据**：`players.json`（玩家档案）、`ingest_player.py`（唯一接入入口，fatal/warning 分级）、
    `data.py`（样本构造，含 `SURVIVAL_SCOPE_ONLY` 开关）、`coverage_audit.py`（SL/ST 坐标覆盖）
  - **基线与模型**：`compare_nolevel.py`（H/B/A 基线）、`c_chart_aware.py`（C 阶梯，GPU 自适应）、
    `embed_charts.py`（Phase2A 表征计算，增量安全）
  - **专项实验**：`time_ablation.py`（时间消融）、`lopo_eval.py`（LOPO 冷启动）、
    `transfer_eval.py`（cross-player transfer + few-shot）、`relations.py`（三目标关系统计）
  - **契约**：`chart_repr.py`（特征登记处，特征清单/新 encoder 注册的唯一权威）、
    `common.py`（共享评估原语：Imputer / mae / r2 / centered_r2 / hgb_fit_predict / 难度区域）
  - **已弃用**（用难度表特征，仅留档，**运行会因列缺失而崩**）：`baseline.py`、`baseline31.py`、`c_model.py`
- `PlayerData/beatoraja/<name>/player1/` — beatoraja 原始存档（gitignore，**唯一副本，改动前提醒用户备份**）；
  `PlayerData/LunaticRave2/<id>.db` — LR2 存档（**只有最佳成绩，无时间戳、无逐局**，见下）
- `ref_repo/` — **只读参考项目**（gitignore）：`beatoraja-master`/`lampghost`（schema 语义权威）、
  `Permikon-main`（排列分析）、`BmsReplayViewer`、`AlphaOSU-main`（osu! 推荐器：ALS 矩阵分解 + pass 模型）、
  `osumania_map_analyser-main`（osu!mania 分析器，2026-09-11 加入）。**后者三个可搬的点**：
  ① `js/ett/versions/minaclac-*.wasm` = **MinaCalc 移植**（6 个版本），FFI 只需
  `(keycount, musicRate, scoreGoal, 行掩码 u32[], 行秒 f32[], 行数) → 8 个 MSD 技巧项`，
  键数支持 4–18（4/6/7 官方、其余通用 n-key），`js/ett/calc.js` 有现成 Node 调用路径（本机 node v22）；
  我们的 `corpus/sequences/<sha>.npy` = `(time_sec, lane, type0=普通/1=LN, dur)`，转换约 30 行。
  抽样 3000 谱：99.9% 为 8 列（7K+scratch→通用 n-key 路径）、95.6% 在 Roxy 适用域（LN≤0.18）、
  99.7% 音符数 ≥80 → **全语料可算 MSD**。用途：把 8 个 MSD 当**响应/偏差轴**（人群校准的难度语义，
  比裸结构统计更有意义——是对"更好的轴"而非"更多的轴"的假设检验）。
  **MSD 已落地（2026-09-11，recommend 分支）**：`msd_prep.py`（序列→行掩码 blob）→
  `msd_compute.mjs`（Node 调 0.74.0 WASM，**非 4K 必须 0.74.0**——第一个有 n-key 管线的版本，
  ~1300 谱/s，4562/4562 零失败）→ `msd_finalize.py`（`dataset/msd.parquet`，7 列）。
  覆盖 = 围栏∪已打 5,570 中的 4,562（缺的 1,008 全是围栏候选序列文件缺失；已打谱面仅 0.7% 缺）。
  **注意 Technical 在 n-key 路径恒为 0.18（4K 专属技能项），已丢弃**。
  外部效度：MSD Overall 与表等级 Spearman = insane 0.801 / satellite 0.852 / stella 0.610，
  **全面优于 c_avg_nps**（0.647/0.731/0.435）；最强单项 chordjack +0.883。
  ② `docs/roxy_algorithm.md`：7 条 burst/sustain 应变流 + 分位数/幂均值聚合 + **Ridge 元模型而非树**
  （树对时间戳噪声过敏感——与我们的教训同源）+ 0.5 序数网格校准（**直接适用于我们的 lamp 头**）
  + 估计器 0.4/0.6 融合降方差；③ `js/interlude/`、`js/patterns/` = Interlude RC/LN 键型检测。
- `docs/` — Phase 1/2A 归档文档；`docs/reference/` — framework paper、hastie15a（fast ALS）、recommend

## 环境与命令

- **路径与多系统（重要）**：本项目在多个系统间迁移，同一位置可能有多个盘符别名。
  代码与脚本一律使用相对仓库根的路径；文档/manifest 遇到绝对路径时按下表换算，
  **接入新系统时把新盘符追加进本表，不要改写历史条目**：

  | 位置 | 已知别名 |
  |---|---|
  | 项目根 | 旧系统 `F:\Projects\bms judge` ＝ 本机 `D:\Projects\bms judge` |
  | BMS 语料库 | 旧 `F:\games\BMS` ＝ 本机 `D:\games\BMS` |
  | beatoraja 安装 | 本机 `D:\games\beatoraja`（旧系统无记录） |
  | 旧 venv 的 uv Python | 旧系统 `C:\Users\user\...` ＝ 本机旧 C 盘（现 `E:`），且用户名已变为 `Administrator` |
  | **玩家存档根** | 旧 `玩家资料/<name>/player1` ＝ 本机 `PlayerData/beatoraja/<name>/player1`（2026-09-11 重命名；历史报告不改写，按此表换算） |
  | **只读参考项目** | 旧 `lampghost/`、`beatoraja-master/`、`Permikon-main/` ＝ 本机 `ref_repo/<同名>/`（2026-09-11 收纳） |

  注意：`manifest.jsonl` 的 `path` 字段保存的是旧系统绝对路径（解析时改用 `rel_path`
  或按上表换算）。盘符与用户名都会随系统继续变，**唯一稳定锚点是仓库根的相对位置**。
- `.venv` = Python 3.11 + CUDA torch 2.11+cu128（RTX 4060；下载慢走 127.0.0.1:7897 代理）。
  torch 相关用 `.venv/Scripts/python.exe`；纯分析用系统 `python`（3.13，有 pandas/sklearn/matplotlib）
- 测试：`python -m unittest discover bms_ml/tests`（**40 个**：25 parser/timeline 回归
  + 15 个 phase3 特征登记处、评估原语、HGB 等价性与 LR2 标签口径回归）
- **LR2 存档（2026-09-11 首次真实接入，不再是 stub）**：`lr2_reader.py` + `ingest_player.py --client lr2`。
  一行 = 一张谱的**最佳成绩**（无时间戳、无逐局、无顺序）→ **只能作 player state，不能作首打目标**；
  `acc = (PG×2+GR)/(2×notes)`（对存档 `rate` 验证）；`clear` 只有 **0–5（无 FC/PERFECT）**→ 与
  beatoraja 的 lamp **不可直接混比**；**84% 可经 md5→sha256 关联**语料库；额外提供
  `playcount/clearcount/failcount`（beatoraja 的 scorelog 是破纪录日志，给不出"玩过几次"）。
  名册中 `vsoflan_lr2` = include=false 待决（协议问题：LR2-only 玩家的目标定义）。
  **整轴时间消融已实测（2026-09-11，`P3_TIME_MODE=synthetic[_shuffle]` env 开关进 data.py）**：
  synthetic(保序丢日历) B +1.28 / B_full +0.66 / cR² B 掉 58%、B_full 掉 31%、lamp≈无损；
  shuffle(乱序) 灾难性（B_full 9.890、cR² +0.111 掉 76%、QWK 0.500）；400d 调窗无改善。
  顺序本身是信号主要载体；融合模型（响应/偏差/MSD 轴）对时间损失显著更鲁棒；
  **LR2-only 进当前预测管线的结论被实验加固**（数据入口 = 推荐方向的矩阵补全）
- **客户端必须分开对待（用户 2026-09-11 明确提出，已实测证实并写进 PROTOCOL §1）**：同一玩家两份存档
  共有的 4,576 张谱面上，两客户端 acc 差 **sd 10.16pp**、>5pp 占 29.4%、与 LN 比例相关仅 +0.016 →
  **差异来自判定/记录本身，非计分口径**，且幅度大于模型自身误差。**禁止把 LR2 与 beatoraja 行混进同一
  份 player state 特征集**；`data.py` 已有守卫（client≠beatoraja 直接拒绝构建，已单测）。
  跨客户端 lamp 映射 `lr2_reader.lr2_clear_to_beatoraja`（档位 +2）**仅供对比，不代表可混用**
- **谱面编码器现状（2026-09-11）**：`objective_stats`(27) → `objective_stats_v2`(+14，谱面侧最优)
  → `perm_space`(22，排列空间手部位移几何，**已被 v2 支配**：胜 v1 基线但叠加 v2 无增益，勿重复投入；
  `chart_perm_space.py` 可续跑，`perm_space_eval.py` 可复现)。新编码器一律加进
  `chart_encoder_registry()` 再在同一任务上比
- **玩家侧编码器现状（2026-09-11）**：12 维手工历史 → **+24 维个人响应剖面**（`history_response.py`，
  11 轴 × resp/slope，每玩家每轴一元 OLS，严格因果前缀和）；这是有效改进（`B 7.009 → B_full 6.583`，
  **17/18 玩家改善**，且用"同列数打乱配对"对照排除了维度效应），复现用 `response_eval.py`。
  新增历史特征请沿用同一模式：独立 parquet + `(player, sha256)` 键 + 登记处清单。
  **few-shot 集成已试为净负**（`transfer_eval.py` 的 `P3_USE_RESP`，默认关；训练/评估同口径 + 收缩
  λ=20 仍全面略差，k=1 的"改善"经查是常数干扰效应），**勿重复**；规范 few-shot 曲线在去重数据集上
  已刷新（v1 k=1 23.81 / k=200 11.20，v2 k=1 22.71 / k=200 10.96）。
  **逐轴消融**：信息集中在密度族（`npsstd`/`nps`/`ioi`），11 轴真正有用的只有 3–4 个，
  **不要再扩同类轴**；`h_resp_*` 与 `h_slope_*` 单独都不如合并 → 增益来自跨轴模式的超加性。
  **`h_knn_acc` 的 k 已结案**：无响应剖面时 k 影响明显（k=20 最差），有响应剖面后 k 几乎不影响
  （6.549–6.637）→ 维持 k=20，TODO 关闭
- **推荐工具（recommend 分支，2026-09-11）**：`python bms_ml/phase3/recommend.py --player <名>`
  → `bms_ml/output/phase3/recommend/<名>.{html,csv}`。**已切换到融合最优配置**
  （`chart_repr.BEST_FEATURES`，169 特征 = 结构响应/偏差 + MSD 三块；训练帧从
  history_response.parquet 读块，与 `compare_nolevel.py` 的 `B_full` 同源同配置）。
  对玩家档案外的**全部围栏谱面**给出 acc/lamp/BP 预测并分三组（暂缓 <3.5 / 挑战 3.5–6.0 /
  冲刺 ≥6.0，阈值来自 lamp 回归的实测压缩分布，非名义阶梯）。候选以"未来行"（NaN 目标）
  追加进玩家因果帧，**零重复实现**地复用 `build_history_features` + `build_table`
  （MSD 块同样走 `axes=MSD_AXES` 路径）；唯一例外 `h_knn_acc` 在此重算（内建窗口会被其他
  候选的 NaN acc 污染）。**候选缺 MSD（18.1%，序列未构建的表外文件）→ 特征插补，
  CSV 里有 `has_msd` 列可过滤**。**不确定度**：分位数 HGB + CQR 共形校正（acc 区间覆盖
  0.772，带宽异方差 corr +0.846——高方差玩家得宽区间）+ 通过概率分类器（AUC 0.892，
  校准表近对角），输出 `pred_acc_lo/hi`、`p_pass`、`pred_bp_lo/hi`；见 `uncertainty_eval.py`。
  打分时点 = 玩家最近一次 scorelog 记录（不是墙钟，
  避免 recency 特征出训练分布）。**注意：这不是"练了会变强"的模型** —— 无纵向数据，
  训练价值不可度量。
- **跑批耗时纪律（2026-09-11 实测）**：瓶颈是 LOPO 的 GRU 重训，不是 HGB（单次拟合仅 ~0.4s）。
  `c0_state_hgb.py` ≈13.5 min/seed（1 头）、`c0_fewshot.py` ≈3×（3 头）；`transfer_eval.py` 已把
  恒定的 HGB 拟合移出 k 循环（8× 冗余 → 省 ~1.8 min/run，数值逐点相同）。
  **GRU 加速开关**：c0_fewshot 支持 `--batch`（默认 64 复现全部历史数字；512 实测 **5.1×**，
  但改变 SGD 轨迹、须重新验证，**不得与 batch=64 的历史数字混用**）。
  探索性迭代用 1 seed，只有最终报数才跑 3 seeds。
- **特征清单纪律（2026-09-07 起）**：任何脚本需要 27 维 chart 统计或 12 维历史特征，
  一律 `from chart_repr import OBJECTIVE_STAT_COLS / HISTORY_FEATURES`，**不要在脚本里重新声明**；
  少数派/指标一律 `from common import ...`。此前 5 份拷贝导致 `c_jrank` 加进所有脚本
  却漏了登记处（已修，`test_phase3_registry.py` 会防住复发）
- 数据集重建：`python bms_ml/phase3/data.py`（约 6s；GPU 相关步骤才需要 .venv）
- 基线：`python bms_ml/phase3/compare_nolevel.py`（输出 A/H/B/**B_resp** 四行；B_resp 依赖
  `history_response.parquet`，缺失时该行跳过并提示）；响应剖面的完整对照：`response_eval.py`；
  C 阶梯（GPU，全套 3 seeds ≈2.5min）：
  `.venv/Scripts/python.exe bms_ml/phase3/c_chart_aware.py --variants C0,C1,C2 --seed 0/1/2`
- transfer/few-shot：`python bms_ml/phase3/transfer_eval.py`；LOPO 冷启动：`lopo_eval.py`
- 新玩家 SOP（顺序不能反）：`ingest_player.py add <name> PlayerData/beatoraja/<name>/player1` →
  人工审阅 audit → **再**在 players.json 设 include=true → `data.py` →
  `embed_charts.py`（增量）→ `compare_nolevel.py` → `c_chart_aware.py` → `coverage_audit.py`
  → **在 EXPERIMENT_LOG.md 记一条数据变更**

## 数据语义红线（最容易犯的错）

- **scorelog.db 是唯一必需的玩家文件**（首打事件+时间戳全在其中）；score.db /
  scoredatalog.db 当前管线不读，缺失只损失未来选项（ingest 警告不拒绝）
- **scorelog 是"破纪录日志"不是完整游玩日志**；每谱最早行 = 精确首打（oldscore=0）
- 历史 = 玩家**严格早于目标首打时刻**的首打事件。2026-09-04 修复过泄漏 bug
  （旧代码用 phase cutoff/最后一局做窗口，目标自身结果混入特征）——**不要改回**；
  同理 few-shot/transfer 实验里"屏蔽目标成绩"由该语义结构保证，勿破坏
- course 行过滤：`len(sha256)!=64 或 mode>=100`；清洗：clear==0、ex==0、BP>notes+5
- `mode` 列是游玩选项复合编码（gauge*10000+judge*1000+hispeed*100+option*10+lnmode），
  不是 7K/14K
- **难度表等级不进特征**（只作样本围栏 sl/st/発狂2018 与 coverage 坐标）；特征契约见
  `chart_repr.py`；BP 训练用 log1p(raw)，ratio 只作报告视角
- **judge rank（`c_jrank`）已是特征**（2026-09-05 加入）：判定窗影响 acc，不加会造成
  rank1/rank2 谱面 acc 被系统性高估 5-11pp。已知限制：parser 对未写 #RANK 的谱默认填 2
  （无法区分真 rank2 与缺省）；有效判定窗还受玩家 config 影响（部分玩家无 config 存档）
- `SURVIVAL_SCOPE_ONLY`（data.py）= 排除 FAILED 与 acc<50 的实验开关；当前停机状态
  **False（协议全 scope）**；survival 口径数字见 EXPERIMENT_LOG
- timestamp 缺失玩家用 players.json 的 `time: synthetic`（序数日，非真实时间，不得当真；
  3.4 消融结论：无时间可预测但交互分析降级）
- **跨玩家/时间窗特征不可混口径**：recency 类特征在训练来自密集 scorelog 流、在
  few-shot 下只能来自稀疏首打 prefix——分布不可比，transfer_eval.py 已剔除并注释

## 实验纪律

- 学习型模型 ≥3 seeds 报 mean±std；**单 seed 不作结论**；HGB 固定 random_state
- 每个结果 JSON 记录 `features_used`；per-player 指标必报；centered R² 必报（interaction 主判据）
- H/B 是固定参照，比较必须同行集；结论必须区分"能证明/不能证明"
- **每次优化/实验/玩家数据变更后在 `EXPERIMENT_LOG.md` 追加条目**（日期/变更/效果/commit）
- 方案级变更（协议、样本空间、特征定义）先与用户确认；历史教训见各报告更正横幅
