# BMS 难度预测项目 —— 进度归档

> 归档日期：2026-09-04 | 环境：Windows / Python 3.11 + CPU torch（`.venv` 已于 09-03 重建）
> 新 agent 必读：`AGENTS.md`（工作指南）→ `PROTOCOL.md`（实验协议）→ 各期报告。

> **Phase 3.4 完成（2026-09-04）**：Time Ablation（`PHASE3_4_TIME_ABLATION.md`）+ 修复贯穿性
> 历史窗口泄漏 bug（历史 = 严格早于目标首打时刻；此前 3.1-3.3 报告绝对数字偏乐观，已加更正横幅）。
> 当前 8 玩家（chuang/muiclac/tzh/nanji/reiaki/vsoflan/LED/darklord），无泄漏基线：
> A 11.69 / H 7.63 / B 7.02；**C0 的 centered R² 首次转正（+0.08），规模响应假设开始兑现**。
> 时间信息价值 ~1.1 acc MAE；无时间交互信号消失（LR2 判定 B 级：可用但交互降级）。
>
> **Phase 3.1-3.3 摘要**：三目标分离（acc=玩家状态、lamp=谱面、BP=交互）；难度表等级
> 被证明可被客观 kNN 特征完全替代并移出特征（用户决定）；样本空间限定三表并集；
> 协议固化于 `PROTOCOL.md`。详见 `PHASE3_1_REPORT.md` / `PHASE3_2_REPORT.md` /
> `PHASE3_3_READINESS.md`（含 Player Coverage Audit 与新玩家 SOP）。
>
> **Phase 3 起点（2026-09-03）**：研究问题转为 player–chart interaction prediction
> （给定玩家历史与未见谱面，预测其首打表现 score%/lamp/BP）。数据与工程审计：
> `PHASE3_AUDIT.md`；代码：`bms_ml/phase3/`。
>
> **Phase 2A 已完成（2026-09-02）**：目标从"预测 SL"转为"从谱面学习 temporal-spatial chart
> representation"。设计 memo：`docs/phase2a/PHASE2A.md`；四轮小规模实验报告：
> `docs/phase2a/PHASE2A_REPORT.md`（首轮表示 + T1/T2）、`docs/phase2a/PHASE2A_INTERVENTION_REPORT.md`（统计受控干预）、
> `docs/phase2a/PHASE2A_TASK_COMPARISON_REPORT.md`（Task A 对比）、`docs/phase2a/PHASE2A_POOLED_ONLY_REPORT.md`（pooled-only）。
> 代码：`bms_ml/phase2a/`。当前结论与下一步见文末 §11。Phase 1 封存记录：`docs/phase1/PHASE1.md`。

## 1. 项目定位

研究问题：**BMS 谱面难度是否是单一标量？能否用数据驱动的方式学到比现有难度表更丰富的谱面技能需求表示？**

当前路线：结构化统计特征（26 维）→ 已证明能解释 Satellite 难度约 87% 方差；
下一步主线是验证"note sequence 的排列顺序是否携带统计特征之外的信息"。

用户定位：学习型项目。用户亲手写模型（train_step、SequenceCNN），
工程部分由 agent 完成。

## 2. 当前状态（一句话）

Phase 1 已封存（本节以下内容为 Phase 1 归档记录）。
Phase 2A 完成四轮小规模实验：4s×1/60s 网格表示可承载低阶统计之外的排列信息（受控 2-switch 下
S0 恰为 50%、模型可读），但 T1 masked reconstruction 与 Task A（含 pooled-only 变体）的
pretraining 均未让 pooled 表示获得超过 random-init 的结构增益（受控 2-switch probe：
T1 84.6%、Task A per-cell 82.5%、pooled-only 64.6%、random-init 80.4%）。详见 §11。

## 3. 数据

### 原始语料（只读，未改动）

- 路径：`F:\games\BMS`（171.5 GB，2,432,475 文件，3,420 个顶层包目录）
- 谱面文件：48,619（.bme 22,870 / .bms 22,015 / .bml 2,935 / .pms 484 / .bmx 310）
- 压缩包：209 zip + 20 rar + 5 lzh（未解包，内含谱面未统计）
- 测试小集：`testbms`（305 谱面，4 首歌）

### 处理结果（manifest.jsonl，48,619 行）

| 指标 | 数值 |
|---|---|
| 干净谱面（非隔离） | 36,744 |
| 隔离谱面 | 11,875（DP 10,999 / not_7k 845 / no_notes 374 / 控制流 252 / pms 484 / bmson 5 / MGQ-LN 22） |
| 干净歌曲数（song group） | 8,315 |
| 唯一 md5 | 48,290（重复文件仅 329） |
| 有标签（任意表） | 20,739 |
| Satellite 标签（clean 7K） | 1,905（排除 60 个极端 flag 后分析用 1,845） |

### 难度表（output/corpus/analysis 同级的 output/tables/）

| 表 | 条目 | hash | 说明 |
|---|---|---|---|
| Hex 発狂DB | 31,764 | md5 | 最大，无质量过滤，低等级偏斜 |
| Normal2 | 4,776 | md5+sha | 通常范围 |
| Satellite | 2,358 | md5+sha256 | **主数据集**（sl0-12，质量过滤） |
| Stella | 2,230 | md5+sha256 | 高难端 |
| Insane2 / Insane1 / Normal1 / LN / Overjoy | 2,192 / 1,035 / 634 / 485 / 413 | md5 | — |

来源：官方站 + 国内镜像 `https://zris.work/bmstable.htm`（默认，7897 代理可配）。
Turbow 表镜像超时未下载。

## 4. 代码结构（bms_ml/）

| 模块 | 职责 |
|---|---|
| parser.py / timeline.py | BMS 解析 + 统一中间表示（时间轴按 bms-js 参考实现） |
| tests/test_timeline.py | 25 个单元测试（bms-js 参考值 + 回归用例） |
| features.py | 26 维统计特征（A）+ note sequence（B）+ piano-roll |
| dataset.py / split.py / model.py | PyTorch Dataset / song-group split / MLP |
| labels.py / fetch_tables.py | 难度表解析 / 下载（镜像+代理） |
| recon_corpus.py / probe_corpus.py / build_corpus_manifest.py | 扫描 / 抽样探格式 / 全量并行解析 |
| corpus_report.py / dataset_audit.py | 数据体检报告 / audit |
| feature_analysis.py | Satellite 特征分析（mean/线性/MLP/单变量/消融） |
| residual_sequence_analysis.py | 残差 + 序列派生量分析 |
| sequence_dataset.py / sequence_cnn.py / train_sequence_cnn.py | 窗口数据集 / 用户写的 CNN / 训练框架 |
| understand_sequence.py | 序列可视化学习工具 |
| train.py / build_manifest.py / check_data.py / sanity_check.py | 小集（testbms）管线 |

## 5. 阶段进度与结论

### Phase 1：最小闭环（完成）

testbms 305 谱面：解析 → 特征 → Dataset → MLP → 训练/验证。验证了流程通、
同曲泄漏风险（4 首歌全是差分）、标签需用难度表而非 #PLAYLEVEL。

### Phase 2：数据质量与 corpus 建设（完成）

- Parser 修复 5 个真实 bug：STOP 单位（value/48 拍，BRV 的 /192 是坑）、
  头命令正则（#STOP11 被拆坏）、base36 大小写（zz==ZZ）、LNOBJ 重复计数、
  #BPM 0 除零崩溃。全部有回归测试。
- 全量 corpus 管线：48,619 谱面约 8 分钟（8 workers，内存恒定）。
- 分组策略：主键=基础曲名（修掉 noter 字段拆散 820+ 首歌的问题），
  无标题回退所在文件夹；时长/note 方差大的组进 flagged_merges 复核。

### Phase 3：Satellite 特征分析（完成）

数据：1,845 谱面 / 1,503 首歌；严格 song-level 3-way split（1,288/272/285）。

| 模型 | test MAE | R² |
|---|---|---|
| mean | 3.382 | ≈0 |
| 线性回归 | 1.186 | 0.849 |
| MLP（26→64→32→1，5 种子） | 1.064–1.096 | 0.859–0.869 |

单变量最强：peak_nps_1s（ρ=0.87）、avg_nps（0.83）、total_notes（0.78）。
消融：chord/density（R² 0.779）≈ 规模（0.734）> lane（0.620）≫ BPM/LN（≈0）。
残差集中在高密度、高难度谱面（sl5-8 与 sl9-12）。

### Phase 3.5：残差 + 序列聚合统计（完成，负结果）

从 note sequence 提取 17 个聚合量（密度变化/间隔结构/lane 移动/chord 形状），
与 |residual| 做偏相关（控制 sl）——全部 ≤0.14；26+5 特征实验反而更差
（1.115 vs 1.059）。**结论：序列的简单聚合统计不携带超出 26 特征的独立信息。**

### Phase 4：Sequence CNN 首轮（进行中，负结果）

窗口：512 event，非重叠，同谱面窗口同 split；
train/val/test 窗口 4,534 / 966 / 990。

| 模型 | chart MAE | R² | 说明 |
|---|---|---|---|
| A mean | 3.382 | ≈0 | |
| B 26 特征 MLP | 1.068 | 0.864 | |
| **B_window 对照** | **1.081** | 0.858 | 同窗口同标签，证明信息可提取 |
| C Sequence CNN（用户实现） | 2.624 | 0.323 | 窗口级 MAE 2.73 |
| D 26 特征 + CNN | 1.135 | 0.861 | 无互补 |

关键判断：B_window（1.08）证明窗口里有可提取信息，C（2.62）失败在模型归纳偏置
（kernel-3 conv + GAP 难以从原始事件流恢复全局密度），不是数据/标签噪声。
C 训练曲线 100 epoch 从 3.47 缓慢降到 2.86，仍在学习但平台期远高于统计模型。
**待办：核对用户 sequence_cnn.py 的结构（GAP 维度、transpose、padding、激活）后
再决定是否修改模型设计。**

### Phase 4a：输入表示实验（delta_t，完成）

同一 CNN 只改输入列：

| 表示 | chart MAE | R² | 窗口时长辅助任务 MAE / R² |
|---|---|---|---|
| [time, lane, type, duration] | 2.621 | 0.325 | 5.48 / 0.39 |
| [delta_t, lane, type, duration] | **1.088** | **0.866** | **0.10 / 0.999** |

结论：absolute timestamp 是主要表示问题；delta_t 使 CNN 追平统计特征（1.09 ≈ 1.08）。
产物：analysis/rep_experiment/。

### Phase 4b：感受野实验（完成）

固定 delta_t 表示，2×k3 / 4×k3 / 2×k5：

| 模型 | RF | 参数量 | chart MAE | R² |
|---|---|---|---|---|
| A 2×k3 | 5 | 18,785 | 1.076 | 0.867 |
| B 4×k3 | 9 | 43,489 | 1.088 | 0.870 |
| C 2×k5 | 9 | 31,073 | **1.023** | **0.883** |

加深无收益；第一层加宽核（k5）有小而一致的收益（总体 MAE 首次低于 26 特征 MLP）。
产物：analysis/rf_experiment/。

### Phase 4c：Chord-grouping 表示（完成，负结果）

同 timestamp 的 note 合并为一个时间事件（10 维：delta_t + 8 lane 存在位 + has_ln）：

| 表示 | chart MAE | R² | sl0-2 | sl3-5 | sl6-8 | sl9-12 |
|---|---|---|---|---|---|---|
| 旧（note 事件） | 1.064 | 0.876 | 0.844 | 0.901 | 0.854 | 1.720 |
| 新（时间点事件） | 1.097 | 0.840 | **0.615** | **0.866** | 1.224 | 1.964 |

总体略差；低难度带改善、中高难度带变差。混淆因素：时间点数量减半导致窗口数减半
（train 4,534→1,860）且每窗口覆盖约两倍时长；编码丢失 chord 内 note 数/时长细节。
按约定不继续加复杂度。产物：analysis/chord_grouping/。

### Phase 4d：Lane one-hot（完成，无差异）

只把 lane 整数换成 8 位 one-hot（delta_t/type/duration 归一化统计与基线完全一致）：
MAE 1.0505（整数） vs 1.0523（one-hot）——无实质差异。产物：analysis/lane_onehot/。

### Phase 4e：Lane shuffle 破坏测试（完成，关键否定）

整谱打乱 lane（保留 delta_t/type/duration 与 lane 频率，固定种子可复现）：
MAE 1.0582（原始） vs 1.0524（shuffled）——打乱 lane 后性能不变，
说明当前 CNN 的预测力几乎全部来自时间/density 结构，lane 排列没有贡献可提取信号。
产物：analysis/lane_shuffle/。阶段封存见 docs/phase1/PHASE1.md。

## 6. 关键决策记录

1. 时间轴以 bms-js 参考实现为准（STOP /48、停前/停后双时间点、BPM 先于 STOP）。
2. 标签用难度表 hash 匹配，不合并跨表 scale；Satellite 为第一主数据集。
3. 隔离而非删除（DP/控制流/MGQ/PMS/bmson/无 note）；极端值只标记。
4. song-group 按基础曲名分组；flagged_merges 人工复核。
5. 序列表示保持 N×4（time/lane/type/duration）不修改；窗口 z-score 在 train 拟合。
6. 已知坑（踩过）：BRV 的 STOP /192 偏小 4 倍；重复通道行应按 fraction 覆盖
   而非拼接；base36 大小写等价；#BPM 0 除零。

## 7. 产物清单

- `bms_ml/output/corpus/`：scan_summary / charts.txt / probe_report /
  manifest.jsonl（58.7MB）/ quarantine_report / dedupe_report / audit_report /
  corpus_report / sequences/*.npy（36,744 个干净谱面的 note sequence）
- `bms_ml/output/corpus/analysis/`：split.json / features_schema.json /
  normalization.json / results.json / single_feature_report.json /
  residual_report.json / baseline_predictions.json / sequence_quantities.json /
  residual_sequence_report.json / 图表（sl_histogram、26 特征图、pred/residual、
  ablation、sequence_learn/）
- `bms_ml/output/corpus/analysis/seq_model/`：results.json / window_preprocessing.json /
  cnn_checkpoint.pt（seed 0 的 C 模型）
- `bms_ml/output/tables/`：10 张难度表 JSON

## 8. 复现命令

> 路径别名：下述 `F:\` 为旧系统写法，本机对应 `D:\`（项目根 `F:\Projects\bms judge`
> ＝ `D:\Projects\bms judge`，语料库 `F:\games\BMS` ＝ `D:\games\BMS`）。
> 完整对照表见 `AGENTS.md`「路径与多系统」；接入新系统时在该表追加新别名，勿改写历史条目。

```powershell
cd "F:\Projects\bms judge"
.venv\Scripts\python.exe -m bms_ml.recon_corpus --root "F:\games\BMS"
.venv\Scripts\python.exe -m bms_ml.probe_corpus --n 1000
.venv\Scripts\python.exe -m bms_ml.build_corpus_manifest --workers 8
.venv\Scripts\python.exe -m bms_ml.corpus_report
.venv\Scripts\python.exe -m bms_ml.feature_analysis --seed 0
.venv\Scripts\python.exe -m bms_ml.residual_sequence_analysis
.venv\Scripts\python.exe -m bms_ml.train_sequence_cnn
.venv\Scripts\python.exe -m unittest discover bms_ml/tests
```

## 9. 用户学习任务状态

- `train_step.py`：已写函数体，但顺序有误——`optimizer.zero_grad()` 必须在
  `loss.backward()` 之前（当前顺序会清掉刚算好的梯度，参数永远不更新）。留待修正。
- `sequence_cnn.py`：已实现（51 行），SequenceCNN 跑通实验。待核对结构细节。
- `understand_sequence.py`：已完成并理解 N×4 含义。

## 10. 下一步候选（未定，按用户节奏）

> 注：以下 Phase 1 遗留候选已被 Phase 2A 取代，见 §11。

1. 核对/修正 SequenceCNN 结构（GAP 轴、transpose、感受野），重跑 C/D；
2. 若结构无误，讨论最小设计改动（如把与前/后 event 的时间差加入每行输入，
   需用户同意修改数据定义）；
3. 可解释性实验（用已保存的 cnn_checkpoint.pt）——但需等 C 先能匹配 B_window。

---

## 11. Phase 2A 进度（2026-09-02）

### 研究问题

能不能从谱面本身（不依赖人工 skill 维度、不把 SL 当目标）学习出比标量难度更丰富的
temporal-spatial chart representation？核心工作假设：固定时长窗口内的 note 排列本身
含有可学习、可复用的结构信息。

### Representation v1

- 4s 窗口 × 1/60s cell（T=240）× 8 lane（0-6 keys 有序 + scratch 平面），通道 = onset 计数 +
  LN hold 标志；window-relative 时间；不含 BPM/STOP/SCROLL（side channel 预留）。
- 代码：`bms_ml/phase2a/grid_data.py`；设计：`docs/phase2a/PHASE2A.md`。

### 四轮实验与关键结果

| 轮次 | 内容 | 关键结果 |
|---|---|---|
| R1 | 网格表示 + T1 masked reconstruction / T2 next-window + 成绩单 R1-R4 | T1 结构可学（onset F1 0.120 vs random 0.053）；T2 退化为预测空；SL 线性探针增益很小；R4 显示 pretrained 对局部 lane shuffle 的敏感度远高于 random-init（L2 1.40 vs 0.003） |
| R2 | 统计受控 2-switch 干预 + audit + frozen probe | 理论统计最大差全部为 0（S0=50%）；排列确实改变（hamming 0.75）；pretrained 85.2% vs random 83.0% vs S1 80.1%——pretraining 增益仅 ~2pp |
| R3 | Task A 几何关系 pretext（per-cell head） | 任务可学，但 frozen probe 无增益（81.0% ≈ random 81.3%）；**发现并修复上一轮 mask leak（遮挡掩码写反），干净 per-cell 结果 82.5% vs random 81.3%** |
| R4 | Pooled-only Task A（head 只能读 pooled + 查询位置） | 可学（0.533 vs random 0.333）但低于 stats baseline（0.588）；受控 2-switch probe **64.6% < random 80.4%**——训练主动把 pooled 推向统计捷径，结构敏感度不升反降 |

### 核心结论

1. **输入表示本身承载排列信息**：网格 + 线性探针就能读出受控空间排列变化（random-init 已 80%+），
   功劳主要在表示设计而非 pretraining。
2. **"任务可学"不等于"表示变好"**：T1 与 Task A（per-cell 与 pooled-only 两种形态）都没有让
   pooled 表示获得超过 random-init 的结构增益；pooled-only 反而显著下降。
3. 已排除的路径：per-cell 任务头（encoder 偷懒）、pooled-only 几何预测（64 维过度压缩 + 统计捷径）、
   简单变换判别（输入像素可读，不构成结构理解）。

### 下一方向（未定，按用户节奏）

- A. 换"窗口级、必须全局推理"的监督：head 只能用 pooled，target 为两段之间的相对顺序/相对变换，
  使"统计捷径"没有落点；
- B. 回到表示粒度问题（更长窗口 / 多尺度），64 维 pooled 对 per-note 几何确实太挤；
- C. 接受"网格承载排列信息"，转向"哪些排列信息对 skill demand 有预测力"（需 replay/行为标签，Phase 2B）。

### 实验纪律与教训

- 每个 intervention 必须做统计 audit（理论不变量最大差 = 0）并带 statistics-only baseline；
- mask 类任务必须有输入级 leak 断言（本轮借此抓到 build_task_data 掩码写反的 bug）；
- frozen probe 必须同时对比 random-init 与 stats baseline，避免把"输入可读"误当"结构理解"。
