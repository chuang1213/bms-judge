# Phase 2A 统计受控干预实验报告（小规模）

> 系列报告：R1 首轮 → PHASE2A_REPORT.md ｜ R3 Task A 对比 →
> PHASE2A_TASK_COMPARISON_REPORT.md ｜ R4 pooled-only → PHASE2A_POOLED_ONLY_REPORT.md

> 生成时间：2026-09-02 06:28:24 | 运行 408.7s

## 1. intervention 的精确定义
- 窗口内 tap（type==0，key lanes 0-6）随机 2-switch：`(t1,l1),(t2,l2) -> (t1,l2),(t2,l1)`
- 每窗口交换次数 ≈ swaps_factor × tap 数（本实验 swaps_factor=10.0）
- LN（type==1）与 scratch（lane==7）不参与交换（理由：避免 hold 通道与 scratch 统计被改变）
- 不是简单随机 permutation：2-switch 是保持 chord size 与 per-lane totals 同时成立的最小组原语

## 2. 理论上保持不变的统计量
- note 总数；每时间位置 note 数（chord size 分布）；每条 lane 总 note 数；
  时间顺序；onset/duration；局部 density（任何时间分桶）；NPS

## 3. 实际 audit 结果
- 样本：80 首歌 / 546 谱面 / 6505 窗口（全部来自 pretrain test，未见歌曲）
- 交换数均值 382.27；hamming（tap lane 改变占比）均值 0.7512，中位 0.8133；no-op 窗口占比 0.0506
- 全局置换相似窗口占比 0.0347（接近 0 说明确实破坏了局部结构，而非整体重贴标签）
- 理论不变量的最大绝对差：{'note_count': 0.0, 'per_lane': 0.0, 'chord_hist': 0.0, 'density_profile': 0.0, 'nps': 0.0}
- 被改变的空间聚合量（confound 记录）：{'adjacent_frac_delta': 0.10187504667159294, 'max_run_delta': 1.2328977709454265, 'span_mean_delta': 0.7331990895043637, 'span_std_delta': 0.3375917004572935}
- audit 判定：PASS

## 4. 小规模 pretrained vs random-init
- pretrained（Phase 2A T1-hold）：test acc 0.8523 (0.8451–0.857，chance 0.5)
- random-init：test acc 0.8295 (0.8167–0.842，chance 0.5)

## 5. statistics-only baseline
- S0（只含理论上被保持的统计：note count / per-lane / chord-size hist / density profile / NPS）：test acc 0.5
- S1（S0 + 简单空间聚合：相邻过渡率 / 同 lane 最长 run / chord span）：test acc 0.8013（这些聚合量本身属于'排列摘要'，预期 > chance，用于说明 intervention 确实改变了排列）

## 6. 结果能证明什么
- 表示（网格 + 冻结编码器 + 线性探针）对"统计受控的空间排列变化"敏感：
  pretrained 85.2% / random-init 83.0% / S1 80.1%，全部远高于 chance 50%。
- 统计控制成立：S0（只含被保持的边际统计）恰好 50.0%，说明原始/变体在低阶统计上确实不可区分。

## 7. 结果不能证明什么
- 区分原始/变体 ≠ 学到 skill demand；≠ 原谱比变体更难；≠ Framework 轴正确。
- S1 > chance 也不构成问题：它说明我们确实改变了排列，只是被人工聚合量捕获。
- **不能把"pretrained 能区分"归因于 pretraining**：random-init（83.0%）与 S1 简单聚合（80.1%）已接近 pretrained（85.2%），
  且 pretrained 与 random-init 的种子区间有重叠（0.845–0.857 vs 0.817–0.842）。
  这说明"能看到排列"的能力主要来自网格输入表示本身，而不是 T1 自监督目标教会了编码器额外的东西。

## 8. 是否值得进入全量实验
**不建议马上全量跑同一个二分类 probe。** 理由：
1. audit 干净（S0 = 50.0%，理论统计全部保持）说明 intervention 设计本身成立、可以复用；
2. 但 probe 的边际信息已经耗尽：pretrained 相对 random-init 只有 ~2pp 且区间重叠，
   全量只会以更高置信度重复同一个结论——"表示对排列敏感，但敏感度主要来自输入表示而非 pretraining"；
3. 真正有区分度的下一步是改造 pretraining 目标，而不是把 probe 放大。

建议的下一步（三选一，按优先级）：
- A. 把"结构敏感度"写入训练：例如在 T1 上增加一个辅助判别头（原始 vs 2-switch 变体），
  看 encoder 能否在统计受控条件下显著超过 random-init（这是对"pretraining 学到结构"的直接检验，
  但注意不要把本 probe 的测试集泄漏进训练）；
- B. 用更难、更接近"人类可命名结构"的探针（例如给定两个局部片段，判断是否同一 pattern 平移/置换），
  看 pretrained 是否在 random-init 无法线性读出的任务上拉开差距；
- C. 接受"网格本身承载排列信息"这个结论，把问题转向"哪些排列信息对 skill demand 有预测力"
  （需要 replay 或更细的行为标签，属 Phase 2B 范畴）。

**结论边界（保留）**：本实验只证明表示"看到了"排列变化；不证明该变化对应 skill demand，
也不证明 pretraining 是看到它的原因。
