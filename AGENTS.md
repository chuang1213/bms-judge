# AGENTS.md — BMS Player–Chart Performance 项目工作指南

> 给未来 agent/session 的最小必读。读完后按需深入，不要凭常识猜测项目约定。

## 项目是什么

研究型项目：**给定玩家截至 T 的游玩历史与一张他没玩过的 BMS 谱面，预测其首打表现
（acc / lamp / BP 三目标分开）**，最终目标是可验证的 player state 表示。
研究工程，不是产品开发；负结果同样是交付物。

## 当前状态一句话（2026-09-11）

18 名玩家、协议口径（全 scope，sl/st/発狂2018 围栏、c_jrank 在列）下基线
（train 6,425 / test 6,434）：A 11.97 / H 7.62 / **B 6.99**（acc MAE）；
centered R²：H +0.278 / B +0.387；B_v2（+v2 无阈值谱面统计）当前最优：acc **6.825** / cR² **+0.406**。
**Cross-player transfer 已验证**（PHASE3_4_TRANSFER.md）：人群预训练+个体 conditioning 的 M2
在 18/18 玩家上胜过 D-local，few-shot 样本效率 3-5×。
**Phase 3.5 审阅**（PHASE3_5_REVIEW.md）：B 比平凡基线"玩家均值"（9.997）好 ~3.0（30%）→ 路线健康；
**A 比全局均值（11.43）还差**，不得当下界引用；**C0−B 差距单调收窄 3.42→1.94→1.56→1.25**。
**2026-09-11 修正**：`c0_state_hgb.py` 训练/评估 chart 尺度不一致的 bug（详见 EXPERIMENT_LOG）——
修后其内联 M2 与 `transfer_eval` 官方曲线精确对齐，原"learned state 每个 k 都赢"的结论**被推翻**
（修复后增益退化为噪声级），"C 换掉手工 conditioning"这条路的证据需重估。
所有数字变化见 `EXPERIMENT_LOG.md` 台账。

## 阅读顺序（动代码前必读）

1. `PROTOCOL.md` — 实验协议（样本定义/切分/指标/可信度规范），**改协议必须先改此文件**
2. `EXPERIMENT_LOG.md` — **优化/实验/玩家数据的追加式台账**（看"改过什么、效果如何"先查这里）
3. `PHASE3_AUDIT.md` — 原始存档数据语义（scorelog 的真实含义、所有坑）
4. 最新一期报告（当前：`PHASE3_5_REVIEW.md`，代码审阅 + 方向反思）
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
- `玩家资料/<name>/player1/` — 玩家原始存档（gitignore，**唯一副本，改动前提醒用户备份**）
- `lampghost/`、`beatoraja-master/` — **只读参考项目**（gitignore，schema/语义权威来源）
- `docs/` — Phase 1/2A 归档文档；`docs/reference/` — framework paper

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

  注意：`manifest.jsonl` 的 `path` 字段保存的是旧系统绝对路径（解析时改用 `rel_path`
  或按上表换算）。盘符与用户名都会随系统继续变，**唯一稳定锚点是仓库根的相对位置**。
- `.venv` = Python 3.11 + CUDA torch 2.11+cu128（RTX 4060；下载慢走 127.0.0.1:7897 代理）。
  torch 相关用 `.venv/Scripts/python.exe`；纯分析用系统 `python`（3.13，有 pandas/sklearn/matplotlib）
- 测试：`python -m unittest discover bms_ml/tests`（**36 个**：25 parser/timeline 回归
  + 11 个 phase3 特征登记处、评估原语与 HGB 等价性回归）
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
- 基线：`python bms_ml/phase3/compare_nolevel.py`；C 阶梯（GPU，全套 3 seeds ≈2.5min）：
  `.venv/Scripts/python.exe bms_ml/phase3/c_chart_aware.py --variants C0,C1,C2 --seed 0/1/2`
- transfer/few-shot：`python bms_ml/phase3/transfer_eval.py`；LOPO 冷启动：`lopo_eval.py`
- 新玩家 SOP（顺序不能反）：`ingest_player.py add <name> 玩家资料/<name>/player1` →
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
