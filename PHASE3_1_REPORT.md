# Phase 3.1 报告：三目标表现预测（Acc / Lamp / BP）

> 日期：2026-09-03 | 分支：phase3 | 代码：`bms_ml/phase3/{relations.py, baseline31.py, c_model.py}`
> 产物：`bms_ml/output/phase3/{target_relations.*, baseline31_results.json, c_model_*.json, phase31_summary.png}`（不入库）
> 数据定义与 v0 完全一致（4 玩家、三表并集、严格时间外推、首打事件、NO_PLAY/ex==0 排除、course 排除、manifest.notes 归一）。
> 数据修正：新增两条清洗规则——首打 ex==0（含 1 例 minbp=INT32_MAX 哨兵）视为未实际游玩，排除（共 3 行）。

## 0. 一句话结论

三个目标都可预测，且呈现清晰的**信息梯度**：acc 由玩家历史主导（chart 无增益）、lamp 由谱面主导（history 补充）、BP 需要两者（唯一 B 显著最优的目标）；同 acc 的玩家在 lamp/BP 上仍有巨大差异，**acc 不足以描述 player performance** 得到直接证实。C（GRU history encoder）v1 在 acc 上输给手工统计（11.96 vs 9.34）、在 lamp 上赢 H 但输 B；多任务共享头无明确收益。

## 1. 三目标的统计关系（relations.py，n=7,939 首打）

| 相关（整体） | Spearman | Pearson |
|---|---|---|
| acc ↔ BP | -0.834 | -0.889（清洗后；哨兵值曾使其假性归零） |
| acc ↔ lamp | +0.747 | +0.636 |
| BP ↔ lamp | -0.866 | -0.491 |

相关性高 ≠ 信息结构相同。**条件分析（同 acc 桶内的 lamp/BP 离散度）**：

| acc 桶 | n | lamp p10→p90 | BP p10→p90 | fail 率 |
|---|---|---|---|---|
| 75-80 | 1,258 | 1 → 7 | 18 → 129 | 14% |
| 80-85 | 846 | 4 → 7 | 12 → 80 | 3% |
| 85-90 | 415 | 6 → 7 | 6 → 60 | 1% |
| 90-95 | 121 | 7 → 8 | 1 → 24 | 0% |

acc 80-85 桶内：lamp 跨 EASY→EXHARD、BP 相差 6.7 倍；分玩家看同一 acc 桶，画像完全不同——chuang（n=374，HARD 为主，BP p50=36）、muiclac（n=22，BP p50=13、零 FAILED——精准型）、nanji（n=242，含 LASSIST 与 FAILED——不稳定型）、tzh（n=208，FAILED 13 例）。**同 acc、不同生存状态，直接证明 acc 不充分。**

## 2. Baseline 矩阵（test，时间外推，A=chart-only / H=history-stats / B=A+H）

### acc（MAE，越低越好）
| A | H | B |
|---|---|---|
| 13.78 | **9.34**（R² 0.32） | 9.67 |

### lamp（有序 1-9；regression 与 classification 比较）
| | A | H | B |
|---|---|---|---|
| regression ord MAE | 1.761 | 1.936 | **1.469** |
| regression QWK | 0.509 | 0.304 | **0.627** |
| classification ord MAE | 1.925 | 2.015 | 1.505 |

→ **回归式优于分类式**（三种特征集一致）；lamp 分玩家（B）：chuang 1.30 / muiclac 1.48 / **nanji 1.34** / tzh 1.76——nanji 的 lamp 是他最好预测的目标。

### BP（双轨 × 双变换；每格同时报告 raw MAE 与 ratio MAE×1000）
| 训练轨道 | A | H | B |
|---|---|---|---|
| raw（raw 目标） | 405 / 140 | 217 / 79 | 245 / 86 |
| **raw（log1p 目标）** | 255 / 87 | 203 / 70 | **182 / 63** |
| ratio（raw 目标） | 403 / 139 | 208 / 74 | 241 / 85 |
| ratio（log1p 目标） | 371 / 128 | 205 / 73 | 237 / 83 |

（格式：raw MAE [BP 单位] / ratio MAE×1000 [misses/note]）

**BP 结论**：① log1p 变换在 raw 轨上收益巨大（B：245→182，中位绝对误差 107→40）；② **训练用 log1p(raw BP) 全面最优**——它在两个尺度上都赢过 ratio 轨道直接训练（63.4 vs 83.1）；notes 归一化更适合作为**报告与跨谱面比较的视角**而非训练目标（raw 轨模型转换出的 ratio 预测甚至比 ratio 轨自己更准）；③ 依据：log1p(raw) 偏度 0.15（近正态），而 log1p(ratio) 偏度仍 2.01——ratio 跨越三个数量级，条件化不佳；④ 低 BP 区间离散性：raw BP ≤10 有 355 行（整数值主导），但 ratio <1% 的 713 行有 642 个不同取值（notes 千级时分辨率 ~0.001），**ratio 在低 BP 区并不粗糙**；⑤ 同表同等级的跨玩家离散度：raw ±77.9（中位 ~100 的 78%）→ ratio ±0.017（中位 ~0.04 的 43%）——归一化确实提升可比性约一倍；⑥ 难度越大 BP 中位数并不上升（各等级 p50 都在 ~100 附近 / ratio ~4-5%）——**强 selection bias：玩家只打自己能活下来的谱**，BP 的"难度解释力"被选择性尝试吃掉了。

### 分玩家（acc MAE / lamp ord MAE / BP raw MAE，B 模型）
| | chuang | muiclac | tzh | nanji |
|---|---|---|---|---|
| acc | 6.58 | 5.38 | 8.67 | 17.85 |
| lamp | 1.30 | 1.48 | 1.76 | 1.34 |
| BP | 44.9 | 61.8 | 101.2 | 510.3 |

nanji：acc 与 BP 的短板（BP 510 由超长谱 BP 数千的长尾主导），但 lamp 反而最好预测——他的问题不在"能不能过"而在"非 fail 局的稳定性与超长谱崩盘"。

## 3. 信息分工（Phase 3.1 核心问题）

| 目标 | 主导信息 | 证据 |
|---|---|---|
| acc | **player state** | H(9.34) < B(9.67) < A(13.78)：chart 特征零增益甚至负增益 |
| lamp | **chart 难度**，history 补充 | A(1.761) > H(1.936)，B(1.469) 显著最优 |
| BP | **两者交互** | 唯一 B 显著最优的目标（182 vs H 203 / A 255） |

v0 发现的分离在 lamp/BP 上不仅存在，而且构成了一个梯度：**acc → lamp → BP，信息权重从"玩家现在什么状态"逐渐转向"谱面是什么结构、两者如何交互"**。三个目标不是冗余的刻度，而是三个不同的信息探针。fail-AUC（v0）与这里的 lamp/BP 结果相互印证。

## 4. C：history encoder（GRU，最近 100 次首打事件 × 9 维，val 早停）

| 目标 | C independent | C shared（3 头联合） | 最优手工模型 |
|---|---|---|---|
| acc MAE | 11.96 | 13.53 | **9.34 (H)** |
| lamp ord MAE / QWK | 1.765 / 0.519 | **1.674 / 0.563** | 1.469 / 0.627 (B) |
| BP raw MAE | 221.3 | 260.2 | **182.5 (B)** |

- **C v1 没有超过手工统计**（acc 输 H 2.6 MAE；lamp 赢 H 输 B；BP 输两者）。序列输入只有 9 维事件特征 + 1,982 训练样本，小 GRU 学不出超过 11 个精心设计滚动统计的东西，属于诚实的 v1 负结果；
- **多任务（问题 5）**：shared 相对 independent——lamp 改善（1.674 vs 1.765）、acc 与 BP 恶化（13.5 vs 12.0；260 vs 221）。**当前规模下无明确多任务收益**，不强行共享；
- nanji：C independent 的 acc（16.43）略优于 H（18.50）——序列信息对他的近期波动略有帮助，但不足以修复；
- 训练稳定性记录：BP 头最初用 z-score + MSE 导致 expm1 外推爆炸（raw MAE 万级）；改为有界目标（log1p 原尺度）+ Huber + 输出钳制到训练值域 + 按时间 val 早停后正常。此坑记录在案。

## 5. 七个问题的答案

1. **可预测程度**：acc MAE 9.34（R² 0.32）/ lamp ord MAE 1.47（QWK 0.63，regression 式）/ BP raw MAE 182（MedAE 40）。三者都可预测，BP 必须用 log1p 训练；
2. **分工**：acc=player-dominated，lamp=chart-dominated，BP=interaction——见 §3 表；
3. **互补增益**：B 在 lamp、BP 上显著超过 max(A,H)；在 acc 上无增益；
4. **三目标信息不同**：高相关但条件分布与特征响应结构不同（§1/§3）；
5. **multi-task**：无明确收益（lamp 小赢、acc/BP 输），v1 不采用共享头；
6. **nanji**：仍是 acc/BP 的主要 failure case（17.9 / 510），但 lamp 是他最好预测的目标；C 略改善其 acc；
7. **acc 不充分**：同 acc 桶内 lamp p10-p90 跨 3 级、BP 相差 6.7 倍、同 acc 分玩家画像不同（§1），证实需要多维 player state。

## 6. 下一步候选

1. C 的输入瓶颈明确：事件特征只有 9 维标量。把 **chart 侧特征**（26 维/表等级）注入事件序列（每个历史事件带上其谱面特征）是 C v2 最直接的加强，检验"序列 + 更丰富事件"能否真正超过手工统计；
2. lamp 换 CORAL/累积概率式 ordinal 头，验证 QWK 还能涨多少；
3. nanji 专项：时间分层特征（近 7/30 天分层、acc 斜率/波动率）与超长谱单独分桶；
4. BP 的 selection bias 建模（玩家"敢打"本身就是信号——尝试验证集上"是否出现在目标集"的选择模型）。
