# Phase 2A 候选 pretext task 比较 + Task A 小规模实验

> 系列报告：R1 首轮 → PHASE2A_REPORT.md ｜ R2 受控干预 → PHASE2A_INTERVENTION_REPORT.md ｜
> R4 pooled-only → PHASE2A_POOLED_ONLY_REPORT.md

> 生成时间：2026-09-02 07:27:46 | 运行 615.9s

## 1. 三个候选任务
### A. 局部几何关系预测（本轮实现）
- 输入：窗口网格 + 被遮挡 span/随机 cell（与 T1 相同）；
- target：被遮挡真实 note 与最近可见前序 note 的客观几何关系：
  lane distance（0/1/2/3+）、相对移动方向（-1/0/+1）、相对时间桶、是否 chord（simultaneity）——全部自动生成；
- 学到什么：必须对具体 lane 位置与时间关系作出承诺（不是'有没有 note'）；
- 最容易的 shortcut：类别先验 / 从 span 边界复制 / 密度→lane 距离分布；
- 支持'理解结构'的结果：test 准确率显著高于 stats baseline 与 random-init，且 interior 样本仍有收益；
- 只说明'任务可预测'的结果：准确率高但 frozen probe 无增益（decoder 型学习，表示未受益）。

### B. 时间结构重排
- 输入：一个窗口切成 K 段并打乱；target：正确顺序（或 pairwise order）；
- 学到什么：段间 temporal 连续性 / 乐句结构；
- 退化风险：很多谱面的段顺序本身不唯一（低 ceiling）；模型可能只学'边界 gap 更平滑'的局部统计；
- 判定标准：与 A 相同，但先要排除 ambiguity（需要额外分析正确顺序是否可恢复）。

### C. 结构一致性 / matching
- 输入：两个窗口，其中一个来自另一个的已知程序化变换；target：匹配/关系；
- 学到什么：跨窗口内容相似性；
- 风险：上一轮已证明变换判别在 random features 上就可读（83%），且我们还没有正确的不变性集合，容易把'输入像素可读'误当成'结构理解'。

## 2. 推荐：A，理由
1. A 直接修复 T1 的已知失败模式：T1 损失被空 cell 主导（模型靠预测空拿低 loss，recall 仅 0.066）；A 的监督只落在真实 note 上，无法用'预测空'偷分；
2. target 是几何关系（lane distance / 方向 / 相对时间 / 同时性），迫使模型对位置作出承诺，且全部自动生成；
3. B 有 order-ambiguity 低 ceiling 风险，C 已被证明在随机特征上可读——A 是三者中'任务难、信号干净、可判别'平衡最好的。

## 3. 小规模实验结果
> **更正（2026-09-02 补充）**：本报告与 Pooled-only 实验联调时发现，当时 build_task_data 的
> 遮挡掩码写反（`masks != 0` 保留了被遮挡 cell、清空了未遮挡 cell），导致上一轮 per-cell Task A
> 的模型在训练/评估时能看到被遮挡内容（mask leak）。修复（`masks == 0`）后的干净数字：
> trained acc = {'lane_dist': 0.5047, 'direction': 0.6049, 'dt_prev': 0.5998, 'simult': 0.7917}
> （平均 0.625，仅比 stats baseline 0.588 高 0.037）；2-switch frozen probe：
> per-cell taskA 0.8250 / T1 0.8459 / random-init 0.8125。结论方向不变（pretraining 增益很小），
> 但本报告正文中的 Task A 准确率数字（0.776 等）已被污染，请以 Pooled-only 报告 §5 为准。

- 数据：train 3911 窗口 / 68012 focal notes；test 2470 窗口 / 42081 focal notes
- 任务准确率（test，4 heads 均值）：
  - statistics-only baseline：{'lane_dist': 0.4271, 'direction': 0.4797, 'dt_prev': 0.6555, 'simult': 0.7891}
  - random-init head：{'lane_dist': 0.1662, 'direction': 0.0264, 'dt_prev': 0.1973, 'simult': 0.2626}
  - Task A trained：{'lane_dist': 0.5047, 'direction': 0.6049, 'dt_prev': 0.5998, 'simult': 0.7917}
  - 分层（interior/edge/random）：{'span_interior': None, 'span_edge': {'lane_dist': 0.4294, 'direction': 0.5352, 'dt_prev': 0.5496, 'simult': 0.771, 'n': 29176}, 'random_cell': {'lane_dist': 0.675, 'direction': 0.7623, 'dt_prev': 0.7133, 'simult': 0.8384, 'n': 12905}}

## 4. 与 T1 的比较（frozen pooled 表示的排列敏感度）
- Task A pretrained：0.825
- T1 pretrained：0.8459
- random-init：0.8125
- S0（保持的统计）：0.5；S1（+空间聚合）：0.7937

## 5. 结果能证明什么 / 不能证明什么
- 能证明：如果 Task A 的 frozen probe 明显高于 T1 与 random-init，说明该任务确实给 pooled 表示
  施加了比 T1 更强的结构压力；任务准确率高于 stats baseline 说明监督信号不能只用低阶统计解出。
- 不能证明：几何关系预测 ≠ skill demand；也不能把 probe 增益直接解释为'学会了 stair/jack'。

## 6. 失败最可能的原因（如果失败）
- 类别先验主导（多数 focal 的 lane distance=1/2，方向=0）：训练陷入 prior，需先平衡采样；
- span 边界复制：模型只在边界样本上得分，interior 无增益；
- 头部吃 local features，pooled 表示仍然懒惰：需要把任务头改为只用 pooled 或加 pooled-only 探针；
- 样本太少 / epoch 太少，任务未收敛。

## 7. 是否值得全量
**结论（修复 mask leak 后）：不值得进入全量。**

- 干净数字下 per-cell Task A 的 frozen 2-switch probe 为 82.5%（random-init 81.3%、T1 84.6%），
  增益仅 ~1.2pp，且任务准确率只比 stats baseline 高 0.037——"任务可学"没有转化为"表示变好"；
- 随后的 Pooled-only 实验（见 PHASE2A_POOLED_ONLY_REPORT.md）进一步证明：即使 head 只能读 pooled，
  pretraining 反而让排列敏感度下降（64.6% < random 80.4%）；
- 因此本路线（Task A 系 pretext）不值得继续调参；下一方向见 PROJECT_ARCHIVE.md §11。
