# BMS 难度预测 —— 第一阶段最小闭环

目标：从 BMS 谱面预测难度，第一阶段先建立
「解析 → 中间表示 → 特征 → Dataset → baseline → 训练/验证」的可验证闭环。

## Phase 2A（进行中）

目标与设计：`docs/phase2a/PHASE2A.md`；四轮实验报告见 `docs/phase2a/PHASE2A_REPORT.md` /
`docs/phase2a/PHASE2A_INTERVENTION_REPORT.md` / `docs/phase2a/PHASE2A_TASK_COMPARISON_REPORT.md` /
`docs/phase2a/PHASE2A_POOLED_ONLY_REPORT.md`；进度归档见 `docs/reports/PROJECT_ARCHIVE.md` §11。
Phase 1 封存记录：`docs/phase1/PHASE1.md`。Phase 3（player–chart performance prediction）
审计见 `docs/reports/PHASE3_AUDIT.md`，代码见 `bms_ml/phase3/`。

代码：`bms_ml/phase2a/`（网格表示、T1/T2、统计受控干预、Task A、pooled-only 实验）。

## 项目结构

```
bms_ml/
  parser.py           BMS 文本解析（头命令、通道行、控制流检测、base36/62、编码）
  timeline.py         统一谱面中间表示（note/LN/BPM/STOP 事件、时间轴、元数据）
  features.py         表示 A：统计特征向量；表示 B：note 事件序列；piano-roll 网格
  labels.py           难度表下载/解析（md5/sha256 → level）
  fetch_tables.py     下载难度表（默认走国内镜像 zris.work，可配 7897 代理）
  split.py            同曲分组划分（防同曲差分泄漏）
  dataset.py          PyTorch Dataset / DataLoader
  model.py            最小 MLP baseline
  check_data.py       数据摸底报告
  build_manifest.py   解析全部谱面 → manifest.jsonl + 数据报告 + 隔离报告
  sanity_check.py     随机抽样打印解析统计 + 生成 piano-roll PNG
  train.py            训练入口（无标签时拒绝训练，--smoke-test 只验机制）
  output/             产物目录（manifest、报告、tables、sequences、runs）
```

## 用法

```powershell
# 1) 数据摸底
python -m bms_ml.check_data --data-dir "F:\Projects\bms judge\testbms"

# 2) 下载难度表（国内镜像；可用 --proxy 指定 7897 代理）
python -m bms_ml.fetch_tables --proxy http://127.0.0.1:7897

# 3) 解析全部谱面 → manifest + 报告
python -m bms_ml.build_manifest --data-dir "F:\Projects\bms judge\testbms" `
    --tables-dir bms_ml\output\tables

# 4) sanity：抽样打印 + piano-roll 图
python -m bms_ml.sanity_check --n 6

# 5) 训练 baseline（需 manifest 中有对应表的标签；否则会明确拒绝）
python -m bms_ml.train --table "発狂BMS難易度表" --epochs 40 `
    --out-dir bms_ml\output\runs\run1

# 5b) 无标签时验证训练机制（不构成实验结果）
python -m bms_ml.train --smoke-test --epochs 2

# 6) parser 单元测试（23 个用例，含 bms-js 参考值）
python -m unittest discover bms_ml/tests -v

# 7) dataset audit（数据质量统计 + 难度直方图）
python -m bms_ml.dataset_audit --manifest bms_ml\output\manifest.jsonl

# 8) Satellite 特征分析（严格 song split；mean/线性/MLP/单变量/残差/消融 + 图表）
python -m bms_ml.feature_analysis --manifest output\corpus\manifest.jsonl `
    --out output\corpus\analysis --seed 0
```

### Satellite 特征分析结论（2026-09-02，n=1,845，1,503 songs）

- mean predictor：test MAE 3.382；
- 线性回归：test MAE 1.186（R² 0.849）；
- MLP(26→64→32→1，5 个种子)：test MAE 1.064–1.096（R² 0.859–0.869），
  同一 song split 下稳定优于线性；
- 单变量：密度/规模类最强（peak_nps_1s ρ=0.87、avg_nps 0.83、total_notes 0.78）；
- 消融（线性 test R²）：chord/density 0.779 ≈ 规模 0.734 > lane 0.620 >>
  BPM/timing 0.019 ≈ LN 0.019；
- 残差：误差集中在 sl5-8 与 sl10-12（MAE 1.2–1.7），且随 note 数/密度增大，
  说明高密度谱面的局部结构信息是下一阶段重点。

完整产物在 `output/corpus/analysis/`（split.json / features_schema.json /
normalization.json / results.json / single_feature_report.json /
residual_report.json / 图表）。

## 第二阶段：数据质量验证

### 大型 corpus 管线（F:\games\BMS，171GB / 48,619 谱面）

```powershell
# 1) 流式扫描（只走元数据）→ charts.txt + scan_summary.json
python -m bms_ml.recon_corpus --root "F:\games\BMS" --out output\corpus

# 2) 抽样探格式（新扩展名全量 + 其余按比例）→ probe_report.json
python -m bms_ml.probe_corpus --charts output\corpus\charts.txt --n 1000

# 3) 全量并行解析（多进程、流式写 manifest，默认 8 workers）
python -m bms_ml.build_corpus_manifest --charts output\corpus\charts.txt `
    --out output\corpus --tables-dir output\tables --workers 8

# 4) 体检报告（A-H 指标 / Satellite 分等级 / 极端样本 / 疑似误合并）
python -m bms_ml.corpus_report --manifest output\corpus\manifest.jsonl

# 5) 标准 audit（难度直方图、warnings、组统计）
python -m bms_ml.dataset_audit --manifest output\corpus\manifest.jsonl
```

真实 corpus 暴露并修复的 parser bug：
- `#BPM 0`（或 NaN）导致时间轴除零 → 回落默认 130 + 记录 `zero_bpm` issue；
  负 #BPM 头部取绝对值 + 记录 `negative_bpm`（均有回归测试）。

真实 corpus 中的新格式/异常（按扩展名或结构隔离，不静默丢弃）：
`.pms`(9K) 484、`.bmson` 5、MGQ-LN 22、控制流 252、非 7K 845、无 note 374、
DP(PLAYER≠1) ~11k。极端值只标记不删除（flag: extreme_bpm 1218 / long_chart 520 /
extreme_notes 113 / extreme_stops 206 / extreme_ln 166 / negative_bpm 15）。

song-group 策略（第二版）：**主键 = 基础曲名**（同曲不同 noter 的差分必须同组，
真实数据中 820+ 首歌因此被拆散，已修复）；无标题时回退所在文件夹。
时长/note 数方差大的组进入 `flagged_merges` 人工复核清单
（多数是同一首歌的 gimmick/短版差分，如 FREEDOM DiVE 33s-186s）。

### parser 正确性

时间轴实现对照 bms-js（bemusic/bmspec 的参考实现）逐场景锁定：
- STOP 单位 = `#STOPxx / 48` 拍（bms-js: `stop / 48`；angolmois: "1/192 小节"）。
  注意：BmsReplayViewer 用 `/192` 并按拍计算，STOP 时长偏小 4 倍，属其已知坑。
- STOP 是时间跳变：同 beat 的音符取停前时间，之后的音符取停后时间
  （实现为 `(beat, t_pre, t_post)` 关键点 + 分段线性）。
- 同一 beat 上 BPM 先于 STOP 生效。
- ch03 BPM 为 hex（0x60=96，不是 60）；ch08 引用 #BPMxx（负数取绝对值并标记）。
- 36 进制大小写等价（`zz` == `ZZ`）；仅 `#BASE 62` 时小写 = 36-61。
- 重复通道行按 (measure, channel, fraction) 后行覆盖（bms-js 语义）。
- LN：LNOBJ 结束标记与同 lane 最近潜在头配对（不重复计数）；
  LNTYPE 1 通道 51-59 成对；未闭合头按谱面末尾闭合并记录 warning。

23 个单元测试覆盖以上全部规则（期望值来自 bms-js 实际输出，可人工复算）。

### manifest 字段

每条记录包含：sha256/md5（chart identifier）、path/rel_path、title/artist/genre、
player/player_mode、rank、group_id（song 分组）、quarantine 原因、issues（warning）、
flags、features（A）、note_sequence_path（B）、labels（table+level+value）、
meta（note/LN/时长/BPM/STOP/密度/和弦/jack/scratch 统计）。

标签只记录 `(table, level)`，不生成跨表 universal level。

### song-group split

- `--split-mode song`（默认）：按归一化 (首作者, 基础曲名) 分组，同曲差分同侧；
  基础曲名 = 去掉 [..]/(..) 内容与常见难度后缀。
- `--split-mode chart`：按文件随机划分（对照/调试用）。
- audit 会标记"疑似误合并"（组内时长跨度 >50% 或 note 数差 >5x），供人工复核。

### baseline

train.py 会输出 mean-predictor（预测训练集标签均值）的 val MAE/RMSE 作为基准 A，
与 MLP（基准 B）对比。MLP 若不能稳定击败 mean，应先查数据/特征而非换模型。

### 难度表研究（2026-09 快照）

| 表 | 条目 | hash | 等级 | 说明 |
|---|---|---|---|---|
| Hex 発狂DB | 31,764 | md5 | 0-24 | 规模最大，全量目录，无质量过滤，低等级偏斜 |
| Normal2 | 4,776 | md5(+sha) | ▽1-12+ | 通常范围，内部一致 |
| Satellite | 2,358 | md5+sha256 | sl0-12 | 质量过滤、活跃维护，覆盖 ☆11-★19 |
| Stella | 2,230 | md5+sha256 | st0-12 | 质量过滤，覆盖 ★19 以上 |
| Insane2 | 2,192 | md5(+sha) | ▼0-25 | 新发狂表 |
| Insane1 | 1,035 | md5 | ★1-25 | 旧发狂表 |
| Normal1 | 634 | md5 | ☆1-12 | 旧通常表 |
| LN | 485 | md5 | ◆1-25 | LN 专项 |
| Overjoy | 413 | md5 | ★★0-8 | 超高难 |

关联方式：表内条目以谱面文件 hash 标识（md5 通用，现代表附带 sha256）；
同一 chart 可被多表收录；同一曲目不同差分是独立文件、独立 hash、独立条目。

**第一版主数据集建议：Satellite（sl0-sl12）**——规模大、单一内部一致标尺、
质量过滤、双 hash 可用；Stella 作为高难端的后续补充。
大型合法语料来源：BMS 活动合集包（BOF 等，作者授权免费分发），
按 md5 与难度表匹配，而不是逐曲爬取。

## 关键决定与理由

1. **表示 A + B 并存**：A 是 chart-level 统计特征（26 维，全部来自结构事实，
   不含人工难度公式），供 MLP baseline；B 是 note 事件序列（time/lane/type/duration），
   为以后 1D CNN / RNN / Transformer 保留。当前不用视觉方案。
2. **标签用难度表**（beatoraja/BeMusicSeeker JSON 格式，按 MD5/SHA256 关联谱面），
   不用 `#PLAYLEVEL`；第一版只允许单表内部比较，不做跨表换算。
3. **隔离而非丢弃**：控制流（#RANDOM/#IF 等）、DP（#PLAYER≠1）、MGQ-LN、
   无 note 谱面进入 quarantine 并记录原因；极端值（超长/超多 note/BPM）记录 flag 但不删。
4. **同曲分组划分**：标题去难度后缀 + 首作者名作为 group key，
   同一曲目的所有差分永远在同一侧，并在报告中检查跨 split 同曲泄漏。
5. **单表等级按有序数值回归**（SmoothL1）：这是第一版的简化假设，
   后续可以讨论 ordinal 回归 / 分类等更合适的建模方式。

## 解析范围（v1）

- 普通 note：通道 11-15、18、19（key1-7）、16（scratch）
- LN：#LNTYPE 1 通道 51-59 + #LNOBJ 配对
- BPM：#BPM 头、通道 03（hex）、#BPMxx + 通道 08
- STOP：#STOPxx + 通道 09（单位 1/48 拍）
- 小节长度：#xxx02（小数，默认 4/4）
- 编码：UTF-8 / Shift-JIS / GB18030 自动探测
- 合并语义：同 (measure, channel, fraction) 后行覆盖（bms-js 语义）

## 已知限制

- 只统计 1P 可见通道；BGA/SCROLL/OPTION/地雷不参与难度特征（记录但不使用）
- 同位置 BPM/STOP 的先后按规范处理，但 STOP 后的时间精度为分段插值
- 峰值小节密度用整曲有效 BPM 近似
- 当前 testbms 只有 4 首歌的差分，标签覆盖有限（各表 2~12 张），
  训练结果只用于验证流程，不构成可靠难度模型
