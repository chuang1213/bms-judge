# Phase 3 Baseline v0 报告：首打表现预测

> 日期：2026-09-03 | 分支：phase3 | 代码：`bms_ml/phase3/{data.py, baseline.py}`
> 产物：`bms_ml/output/phase3/{dataset/, baseline_results.json, baseline_mae.png, baseline_scatter.png}`（不入库）

## 0. 一句话结论

**"玩家历史 → 未见谱面首打表现"的映射成立**：仅用简单历史统计（H）就把首打 acc 预测从 trivial 的 MAE 12.85 降到 **9.35**（R² 0.32，时间外推测试），首打失败预测 AUC 0.84。且出现一个关键的信号分离：**acc 等级几乎完全由玩家状态主导（chart 特征无增益），而"是否会 FAIL"主要由谱面特征主导**——这直接决定了 Baseline C（history encoder）的设计重点。

## 1. 实验设置

- 样本：4 玩家（chuang/muiclac/tzh/nanji）× 目标限定 sl/st/発狂三表并集 × manifest 特征可用；首打 clear=NO_PLAY 排除；
- 切分：**严格时间外推**——每玩家取其首打时间线 q50/q75 为 T_train/T_test；train 目标 = (T50, T75] 的首打，特征只用 ≤T50 的历史；test 目标 = T75 之后的首打，特征用 ≤T75 的全部历史（真实部署语义）。train 1,983 / test 1,986，每玩家每相位 ~480-510；
- 标签：主 = 首打 acc（EX×50/notes，0-100）；辅 = 首打是否 FAILED（lamp==1）；
- 特征：
  - chart 侧（A）：26 维统计特征 + 表来源 one-hot + 归一化等级；
  - 玩家侧（H）：截至 cutoff 的 11 个滚动统计（历史首打 acc 均值/std/最近10次均值、同表同等级历史均值、BP 均值、fail/fc 率、活跃度、距上次游玩天数等）；
  - B = A + H；
- 模型：Ridge 与 HistGradientBoosting（默认参数，无调参——本轮只看结构性的比较）。

## 2. 主结果：首打 acc（test，时间外推）

| 模型 | test MAE | test R² |
|---|---|---|
| M0 全局均值 | 12.85 | -0.06 |
| M1 玩家历史均值 | 12.66 | — |
| M2 (表,等级) 均值 | 14.03 | — |
| M3 玩家偏移+等级偏移 可加 | 12.54 | 0.124 |
| A chart-only（HGB） | 13.88 | -0.12 |
| **H history-only（HGB）** | **9.35** | **0.318** |
| B chart+history（HGB） | 9.55 | 0.327 |

## 3. 关键发现

### 3.1 玩家历史统计携带了几乎全部 acc 可预测性

- H（9.35）显著优于一切 chart 侧或可加模型；**B（9.55）没有优于 H**——加入 26 维 chart 特征 + 表等级后 acc 预测无增益；
- chart-only（A，13.88）甚至不如"玩家历史均值"（12.66）：**不看玩家、只看谱面，无法预测特定玩家的首打 acc**；
- M2（表等级均值 14.03）比全局均值还差：跨玩家平均的等级均值在时间外推下不稳定（四玩家水平差异 + 时间漂移）。

### 3.2 信号分离：acc 看玩家，FAIL 看谱面

首打 FAILED 预测 AUC：

| 特征 | AUC |
|---|---|
| A chart-only | **0.821** |
| H history-only | 0.725 |
| B chart+history | **0.840** |

acc 的"水平"由玩家当前状态决定（历史特征赢）；但"这张谱会不会把他打崩"主要由谱面难度结构决定（chart 特征赢，历史特征只补 0.02 AUC）。**这回答了审计报告 §10 的风险 1**：模型不是只会"玩家强不强"——fail 预测明确依赖谱面信息；也不是只会"谱面难不难"——acc 水平必须靠玩家历史。

### 3.3 同谱不同人（414 张谱 / 892 行，测试期内≥2人首打）

| 模型 | MAE |
|---|---|
| M1 玩家均值 | 12.19 |
| A chart-only | 14.37 |
| H history-only | **9.14** |
| B chart+history | 9.48 |

与总体一致的排序——同一张谱给不同玩家的 acc 差异，主要被各自的历史统计解释。

### 3.4 逐玩家

| 模型 | chuang | muiclac | tzh | nanji |
|---|---|---|---|---|
| M1 玩家均值 | 10.20 | 6.68 | 11.14 | 22.42 |
| H history-only | 6.37 | 4.79 | 7.41 | 18.60 |
| B chart+history | 6.64 | 5.21 | 8.26 | 17.89 |

三人 MAE 5-8 已可用；**nanji（17.9-18.6）是明显短板**——16 个月的高密度档案、acc 分布双峰（常打远超自身水平的谱，BP p75=1270）、且其"近期统计"的时间窗口语义与其他人不同。后续需要针对性的 temporal 特征（趋势/波动率）或分段建模。

## 4. 结果能证明 / 不能证明

- 能证明：玩家历史中的简单统计已携带超过一切谱面边际信息的首打 acc 可预测性（时间外推下成立）；首打失败风险主要由谱面侧决定；chart 特征与历史特征的信息是互补维度（acc vs fail）而非同一维度。
- 不能证明：简单统计就是玩家历史的全部信息（这正是 Baseline C 的问题——序列 encoder 是否能从原始历史事件中抽出滚动统计之外的东西）；不能证明"练这张谱能涨分"（selection bias 未处理）；绝对数字依赖默认参数，未调参。

## 5. 下一步（按优先级）

1. **Baseline C：history encoder**——把玩家首打事件序列（chart 等级/特征 → 结果 → 相对时间）喂给小 encoder，对比 H 的手工统计。判据：C 是否 > H（9.35），重点观察 nanji 是否被修复；
2. **lamp/BP 独立头**：fail AUC 0.84 说明 lamp 方向信息量充足，值得单独建模（有序回归）；BP 右偏，考虑 log 变换；
3. **特征归因**：HGB 的 feature importance / permutation，确认 h_acc_last10、h_level_acc、level_norm 的贡献结构；
4. **nanji 专项**：temporal 特征（斜率、波动率、近7天/30天分层）；
5. 数据管线已固化（`data.py`），后续 C 的样本直接复用 `samples.parquet` + `firstplays.parquet`。
