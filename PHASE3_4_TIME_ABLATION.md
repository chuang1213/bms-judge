# Phase 3.4 报告：Time Ablation / Timestamp Necessity Study

> 日期：2026-09-04 | 分支：phase3 | 独立 ablation，不修改 Phase 3 主结果
> 代码：`bms_ml/phase3/time_ablation.py`；结果：`bms_ml/output/phase3/time_ablation_results.json`
> 数据：6 玩家（chuang/muiclac/tzh/nanji + 新增 reiaki/vsoflan），样本空间 = sl/st/発狂2018 三表并集
> （本实验同时是泄漏修复后的首次完整重跑，见 §2 更正说明）

## 1. Research Question

> 玩家能力建模到底有多依赖时间信息？LR2 没有可靠游玩时间但仍是未来重要数据源——在写任何
> LR2 专用架构之前，需要知道 timestamp 是 player state 的必要条件，还是可选增强。

## 2. 实验设置（含一次重要更正）

- 样本：6 玩家 × 三表并集目标，train 3,036 / test 3,040，严格时间外推切分（q50/q75 逐玩家）；
- **泄漏修复（本实验触发的更正）**：写作过程中发现 3.1 以来的历史特征窗口以 phase cutoff /
  玩家最后一局为界，导致 (a) 目标谱面自身的结果进入 h_knn_acc（k=20 含自身）与 h_acc_mean，
  (b) test 行包含整个测试期战果，(c) h_days_since_active / h_plays_last30d 窗口无上界、
  把 cutoff 之后的未来活动计入。**修复**：每个样本的历史 = 该玩家严格早于目标首打时刻的
  首打事件（"走向这张谱面那一刻"的状态）；时间特征以目标时刻为界。此前 3.1/3.2/3.3 报告的
  绝对数字（尤其 centered R² +0.125/+0.361）偏乐观，相关报告已加更正横幅；
- 变体（同一行集/目标/切分/imputer/HGB 预算，3 seeds；历史成员资格 = 目标首打之前的事件，
  对所有变体一致——被消融的是"历史内部何时发生"的信息）：
  - **H_time**：现行 H（12 特征，含 recency/日历项）
  - **H_no_time**：纯集合统计（8 特征；无绝对日期/recency/时间窗/时序位置）
  - **H_order_only**：H_no_time + h_acc_last10（只保留游玩顺序——timestamp 的弱化形式，如实声明）
  - **H_masked_25/50/75/100**：随机隐藏每玩家 X% 的 scorelog 日期后重算 4 个时间特征；
    100% 掩蔽 ≡ H_no_time 作为机制 sanity anchor（实测完全一致 ✓）

## 3. 总体结果

| 变体 | acc MAE | centered R² | lamp MAE | lamp QWK | BP raw MAE | BP ratio×1000 |
|---|---|---|---|---|---|---|
| **H_time** | **8.17** | **+0.159** | **1.982** | **0.318** | **162.2** | **57.2** |
| H_order_only | 9.07 | -0.009 | 2.041 | 0.269 | 173.6 | 61.5 |
| H_no_time | 9.28 | -0.027 | 2.070 | 0.282 | 171.6 | 60.5 |
| H_masked_25 | 8.80 | +0.015 | 2.015 | 0.271 | 164.1 | 57.8 |
| H_masked_50 | 8.66 | +0.076 | 2.009 | 0.276 | 164.2 | 57.5 |
| H_masked_75 | 8.64 | +0.071 | 1.955 | 0.343 | 163.2 | 56.9 |
| H_masked_100 | 9.28 | -0.027 | 2.070 | 0.282 | 171.6 | 60.5 |

- **时间信息价值 ≈ 1.1 acc MAE（8.17 vs 9.28，相对 ~12%）**，lamp/BP 同向（BP 162 vs 172）；
- **centered R² 是最大的差距所在**：H_time +0.159，H_no_time -0.027——去掉时间后
  player×chart interaction 信号**消失**，模型退化为"玩家强度 + 噪声"；
- masked 曲线合理：25-75% 遮蔽落在两者之间（8.6-8.8），100% 与 H_no_time 精确重合；
- H_order_only（9.07）≈ H_no_time（9.28）：**只保留游玩顺序几乎无增益——价值在日历语义
  （距今多少天、近期活动强度），不在先后顺序本身**；
- 参照：chart-only A = 12.26。**H_no_time 仍显著优于 chart-only**——history-only 的
  player state 成立。

## 4. 难度区间分解（acc MAE / centered R²）

| 区域 | n | H_time | H_no_time |
|---|---|---|---|
| SL | 750 | 7.28 (-0.479) | 8.61 (-1.174) |
| ST0-3 | 1,060 | 7.23 (+0.016) | 8.30 (-0.225) |
| ST4-7 | 413 | 9.96 (-0.237) | 10.41 (-0.227) |
| ST8+ | 23 | 20.06 (-4.27) | 18.68 (-3.94) |
| 発狂★ | 794 | 9.00 (-0.063) | 10.37 (-0.317) |

时间信息**在所有区域都有贡献**（SL/ST0-3/発狂 约 1.1-1.4 MAE），不是只在高难/稀疏区；
ST8+ 仅 23 样本，无统计意义。centered R² 在 H_time 下以 ST0-3 (+0.016) 和発狂 (-0.063)
最接近可用，去掉时间后全面转负。

## 5. 逐玩家与成长诊断

| 玩家 | drift (Spearman acc~time) | H_time MAE | H_no_time MAE | 时间收益 |
|---|---|---|---|---|
| nanji | 0.078 | 17.77 | 21.71 | **+3.94** |
| vsoflan | 0.238 | 7.20 | 8.90 | +1.69 |
| reiaki | 0.453 | 4.94 | 5.27 | +0.33 |
| tzh | 0.392 | 6.77 | 6.83 | +0.06 |
| chuang | 0.409 | 6.95 | 6.95 | 0.00 |
| muiclac | 0.426 | 5.50 | 5.25 | -0.24 |

- **时间收益与 drift 无关**（nanji drift 最低却收益最大）——时间信息的价值不是"追踪成长"，
  而更像**当前活动状态**（h_plays_last30d / h_days_since_active 充当"是否在密集练习期"的
  指示器），对状态波动大的玩家（nanji 双峰）最重要；
- 单特征比较（泄漏修复后）：h_acc_last10（9.11）优于 h_acc_mean（10.37）——近期 form 比
  长期均值更准；但在全特征集合中 order-only 的 last10 增益消失（H_order_only ≈ H_no_time），
  说明日历上下文携带了顺序无法替代的信息；
- fixed-chart 跨玩家子集（762 张）：H_time 8.05 / BP 164.7 vs H_no_time 9.14 / 177.5——
  交互验证同向。

## 6. LR2 结论分级（按 brief 框架）

**总体判定 = B：timestamp 有价值，但不是架构前提。**
`H_no_time`（9.28）损失 ~12% 精度但仍远超 chart-only（12.26）——history-only 的
player state 成立，LR2 数据可以直接以 `time: synthetic` 进入统一管线（已有机制），
其预测能力对"这个玩家面对新谱面大概什么表现"是可用的。

**但有一个重要限制 = 交互研究当前需要时间**：player×chart interaction（centered R²，
本项目下一阶段的核心研究对象）在 H_no_time 下消失。因此 LR2 数据可用于训练与粗预测，
但在"同谱不同人为什么不同"的交互分析中只能作辅助，不能作为主要证据来源——除非未来
found 无时间特征下新的条件化方案（如纯 kNN 条件化的更细粒度版本）。

## 7. Limitations

1. 时间特征只有 4 个标量（recency/activity/span），未测试更丰富的时间编码（趋势/斜率/
   分段）——本实验测量的是"当前 H 的时间依赖"，不是"时间信息的上限"；
2. H_no_time 与 H_time 的差距中可能有一部分来自 HGB 对日历特征的过拟合性利用
   （masked 曲线显示 25% 遮蔽时掉得比 50/75 多，提示部分价值来自少量关键日期）；
3. ST8+ 区域样本不足（23），该区域结论缺失；
4. mask 以"行级随机"模拟 timestamp 缺失——真实 LR2 是整段缺失（时间轴完全断裂），
   两者对 recency 特征的破坏模式不同；
5. drift 只测了 acc 维度；lamp/BP 的时间依赖可能不同。

## 8. 对未来 player-state 架构的建议

1. **数据结构维持"timestamp available / missing 二态"**：`time: real|synthetic` 机制保留，
   synthetic 只保证顺序与协议兼容，绝不当真实时间用（PROTOCOL.md §2）；
2. player state 内部分两层：**set statistics（无时间，可移植到 LR2）+ temporal layer
   （recency/activity，仅 real-time 玩家）**——H_no_time 特征集即第一层的候选定义；
3. 模型应显式声明其历史特征是否使用时间（`features_used.uses_timestamp`），
   使 beatoraja/LR2 来源的样本可区分评估；
4. 交互（centered R²）研究暂以 real-time 玩家为主，LR2 玩家标记为 interaction-secondary。

## 9. 结论（回答 brief 的四问）

1. **timestamp 贡献多少？** acc ~1.1 MAE（12%），BP/lamp 同向但幅度较小；centered R²
   +0.159 → -0.027（交互信号消失）——对预测是增强，对交互分析是必要条件；
2. **贡献的是 current ability 还是 growth？** 当前证据指向 **current activity state**
   （收益与 drift 无关、与活动强度特征绑定），不是成长轨迹追踪；
3. **LR2 history-only 是否足以进入同一 pipeline？** 是（判定 B）——以 `time: synthetic`
   进入，参与训练与预测；但其样本在交互分析中降级为辅助；
4. **下一批玩家数据优先扩充什么？** 不变（coverage audit 结论）：ST4-12 与高库重叠玩家；
   另加一条：**timestamp 缺失的玩家无需排除，但需标记**，两类玩家的贡献维度不同。


## 10. 泄漏修复后的 C 阶梯重跑（同协议，6 玩家，3 seeds mean±std）

泄漏修复改变了所有模型的相对位置，必须完整重报（参照：A chart-only = 12.26）：

| 模型 | acc MAE | centered R² | lamp MAE | lamp QWK | BP raw MAE |
|---|---|---|---|---|---|
| H（=H_time） | 8.17 | +0.159 | 1.982 | 0.318 | 162.2 |
| B | 7.18 | +0.297 | 1.432 | 0.640 | 131.7 |
| **C0（outcome-only GRU）** | 8.88±0.35 | -0.07±0.04 | **1.44±0.02** | **0.64±0.01** | 156.8±2.8 |
| C1（+26D 事件统计） | 11.84±1.05 | -0.36±0.16 | 1.74±0.02 | 0.51±0.01 | 183.1±7.3 |
| C2（+Phase2A 表征） | 11.91±0.67 | -0.34±0.06 | 1.69±0.04 | 0.54±0.01 | 211.1±9.2 |

三个新事实：

1. **C0 在 lamp 上首次追平最强手工模型**（1.44±0.02 vs B 1.43，QWK 0.64 vs 0.64），
   acc 也逼近 H（8.88 vs 8.17）——无泄漏数据上，纯 outcome 序列编码器已成为
   lamp 方向与手工统计并列的一级基线；
2. **C1/C2 恶化于 C0**（事件级 chart 统计与表征在 3k 样本上引入过拟合）——与泄漏期
   阶梯结论相反，再次说明该规模下结论对数据扰动敏感，一切以多 seed + 无泄漏为准；
3. 泄漏对历史各报告的影响方向：H/B 的绝对 MAE 偏乐观、H 的 centered R² 被抬高
   （+0.361 → +0.16）；"层级结构"结论在修复后仍成立且更干净（B 的 acc 优势变得明确：
   7.18 vs 8.17，此前泄漏掩盖了 chart 特征对 acc 的真实互补性）。
