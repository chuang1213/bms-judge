# AGENTS.md — BMS Player–Chart Performance 项目工作指南

> 给未来 agent/session 的最小必读。读完后按需深入，不要凭常识猜测项目约定。

## 项目是什么

研究型项目：**给定玩家截至 T 的游玩历史与一张他没玩过的 BMS 谱面，预测其首打表现
（acc / lamp / BP 三目标分开）**，最终目标是可验证的 player state 表示。
研究工程，不是产品开发；负结果同样是交付物。

## 阅读顺序（动代码前必读）

1. `PROTOCOL.md` — 实验协议（样本定义/切分/指标/可信度规范），**改协议必须先改此文件**
2. `PROJECT_ARCHIVE.md` — 全程进展索引与结论
3. `PHASE3_AUDIT.md` — 原始存档数据语义（scorelog 的真实含义、所有坑）
4. 最新一期报告（当前：`PHASE3_4_TIME_ABLATION.md`，含泄漏更正）
5. `PHASE3_3_READINESS.md` §7 — 新玩家接入 SOP

## 目录

- `bms_ml/` — Python 包。核心：`parser.py`（BMS 解析）、`phase3/`（当前全部工作）、
  `phase2a/`（旧表征实验）、`output/`（可再生产物，gitignore）
- `bms_ml/phase3/` — 数据管线与实验：`data.py`（样本构造）、`players.json`（玩家档案）、
  `ingest_player.py`（唯一接入入口）、`compare_nolevel.py`（H/B/A 基线）、
  `c_chart_aware.py`（C 阶梯）、`time_ablation.py`、`coverage_audit.py`、`chart_repr.py`（特征契约）
- `玩家资料/<name>/player1/` — 玩家原始存档（gitignore，**唯一副本，改动前提醒备份**）
- `lampghost/`、`beatoraja-master/` — **只读参考项目**（gitignore，schema/语义权威来源）
- `docs/` — Phase 1/2A 归档文档；`docs/reference/` — framework paper

## 环境与命令

- 双系统机器：旧文档路径 F: → 现 D:，C: → 现 E:
- `.venv` = Python 3.11 + CPU torch（旧 uv venv 已废）。torch 相关用
  `.venv/Scripts/python.exe`；纯分析用系统 `python`（3.13，有 pandas/sklearn/matplotlib）
- 测试：`python -m unittest discover bms_ml/tests`（25 个，parser 回归）
- 数据集重建：`python bms_ml/phase3/data.py`
- 基线：`python bms_ml/phase3/compare_nolevel.py`；C 阶梯：
  `.venv/Scripts/python.exe bms_ml/phase3/c_chart_aware.py --variants C0,C1,C2 --seed 0/1/2`（**必须 3 seeds**）
- 新玩家：`ingest_player.py add <name> 玩家资料/<name>/player1` → 人工审阅 audit →
  **再**在 players.json 设 include=true（add 会覆盖该标记，顺序不能反）→ 重跑 data.py →
  embed_charts.py（增量）→ compare_nolevel → c_chart_aware → coverage_audit

## 数据语义红线（最容易犯的错）

- **scorelog 是"破纪录日志"不是完整游玩日志**；每谱最早行 = 精确首打（oldscore=0）
- 历史 = 玩家**严格早于目标首打时刻**的首打事件。2026-09-04 修复过泄漏 bug
  （旧代码用 phase cutoff/最后一局做窗口，目标自身结果混入特征）——**不要改回**
- course 行过滤：`len(sha256)!=64 或 mode>=100`；清洗：clear==0、ex==0、BP>notes+5
- `mode` 列是游玩选项复合编码（gauge*10000+judge*1000+hispeed*100+option*10+lnmode），
  不是 7K/14K
- **难度表等级不进特征**（只作样本围栏 sl/st/発狂2018 与 coverage 坐标）；特征契约见
  `chart_repr.py`；BP 训练用 log1p(raw)，ratio 只作报告视角
- timestamp 缺失玩家用 players.json 的 `time: synthetic`（序数日，非真实时间，不得当真）

## 实验纪律

- 学习型模型 ≥3 seeds 报 mean±std；**单 seed 不作结论**；HGB 固定 random_state
- 每个结果 JSON 记录 `features_used`；per-player 指标必报；centered R² 必报（interaction 主判据）
- H/B 是固定参照，比较必须同行集；结论必须区分"能证明/不能证明"
- 方案级变更（协议、样本空间、特征定义）先与用户确认；历史教训见各报告更正横幅
