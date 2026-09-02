# Phase 2A Pooled-only Task A 小规模实验报告

> 系列报告：R1 首轮 → PHASE2A_REPORT.md ｜ R2 受控干预 → PHASE2A_INTERVENTION_REPORT.md ｜
> R3 Task A 对比 → PHASE2A_TASK_COMPARISON_REPORT.md

> 生成时间：2026-09-02 07:17:12 | 运行 636.7s

## 0. 设计
input window → encoder → pooled latent → geometry prediction heads
head 输入 = concat(pooled, 查询位置 (t/T, l/7))；无 per-cell feature map 旁路。
监督定义、数据表示、encoder、数据切分与上一轮 per-cell Task A 完全一致。

## 1. 数据
- train 3911 窗口 / 68012 focal；test 2470 窗口 / 42081 focal；1 seed

## 2. sanity：pooled-only 下 Task A 是否可学
- statistics-only baseline：{'lane_dist': 0.4271, 'direction': 0.4797, 'dt_prev': 0.6555, 'simult': 0.7891}
- random-init pooled-only head：{'lane_dist': 0.2881, 'direction': 0.3062, 'dt_prev': 0.0, 'simult': 0.7374}
- pooled-only trained：{'lane_dist': 0.3775, 'direction': 0.561, 'dt_prev': 0.4041, 'simult': 0.7885}
- best val mean acc：0.5339（未崩溃）
- mask leak 检查：训练首 batch 断言通过（被遮挡 cell 在输入中确实为 0）。

## 3. 表示级测试 1：统计受控 2-switch probe
- taskA_pretrained：{'acc_mean': 0.6458, 'acc_min': 0.625, 'acc_max': 0.6875, 'n_test': 160}
- T1_pretrained：{'acc_mean': 0.8459, 'acc_min': 0.8313, 'acc_max': 0.8688, 'n_test': 160}
- random_init：{'acc_mean': 0.8042, 'acc_min': 0.7937, 'acc_max': 0.8125, 'n_test': 160}
- S0：{'acc': 0.5}
- S1：{'acc': 0.7937}

## 4. 表示级测试 2：几何关系 frozen linear probe（pooled → mean lane_dist/direction/dt/simult）
- pooled_only_pretrained：
  - lane_dist: MAE 0.4856 R² 0.1539（mean baseline MAE 0.5637）
  - direction: MAE 0.3699 R² 0.0736（mean baseline MAE 0.3903）
  - dt_prev: MAE 0.5439 R² 0.0825（mean baseline MAE 0.6007）
  - simult: MAE 0.1941 R² 0.5443（mean baseline MAE 0.337）
- random_init：
  - lane_dist: MAE 0.5015 R² 0.1273（mean baseline MAE 0.5637）
  - direction: MAE 0.3551 R² 0.1138（mean baseline MAE 0.3903）
  - dt_prev: MAE 0.5459 R² 0.0589（mean baseline MAE 0.6007）
  - simult: MAE 0.2159 R² 0.4528（mean baseline MAE 0.337）

## 5. 与上一轮 per-cell Task A 的对照
- **更正（重要）**：上一轮 per-cell Task A 的数据存在 mask leak（build_task_data 的遮挡掩码写反：
  `masks != 0` 保留了被遮挡 cell 的原值、清空了未遮挡 cell）。本轮 sanity 断言将其抓出并修复为
  `masks == 0`。因此上一轮"Task A 大幅超过 stats baseline"的数字不可信。
- 修复后干净的 per-cell Task A 对照（同一切分，重跑）：
  - trained acc：{'lane_dist': 0.5047, 'direction': 0.6049, 'dt_prev': 0.5998, 'simult': 0.7917}
    （平均 0.625，仅比 stats baseline 0.588 高 0.037）
  - per-cell 2-switch probe：taskA 0.8250 / T1 0.8459 / random 0.8125
  → 即使修复 leak，per-cell Task A 相对 random-init 的表示增益也只有 ~1.2pp，结论与之前一致。

## 6. 结论（由数字决定）
**核心判据不成立：pooled-only 没有建立"任务压力 → representation"这条链。**

1. **sanity 通过但任务远未学透**：pooled-only 训练后可学（val 0.534，test 0.533 vs random-init 0.333），
   但**低于 statistics-only baseline（0.588）**——64 维 pooled + 查询位置无法超过逐 note 的可见上下文统计。
2. **表示级测试 1（受控 2-switch）**：pooled-only pretrained **64.6% < random-init 80.4% < T1 84.6%**。
   pooled-only 训练不仅没有增强排列敏感度，反而显著削弱了它（比随机初始化还差 16pp）。
3. **表示级测试 2（几何 frozen probe）**：只有 simult（本身最统计可解）有明显增益（R² 0.54 vs 0.45），
   lane_dist（0.15 vs 0.13）与 dt_prev（0.08 vs 0.06）增益微小，direction 反而更差（0.07 vs 0.11）。

一句话：**pooled-only 的压力存在，但模型选择用"统计捷径 + 查询先验"应付它，而不是把结构写进 pooled 表示；
结果是表示对结构的敏感度不升反降。**

## 7. 失败时的三个候选解释
- **pooled 过度压缩（有支持）**：test 0.533 < stats baseline 0.588，说明 64 维全局向量 + 位置查询
  不足以恢复 per-note 几何；方向 head 的 frozen probe 甚至比 random 更差。
- **统计捷径（有支持）**：simult head 收敛到与 stats baseline 相同水平（0.789 ≈ 0.789），
  且 2-switch 探针从 80.4% 掉到 64.6%，说明训练把 pooled 特征推向"密度/查询先验"型编码，
  主动舍弃了排列信息。
- **4s 粒度（未验证，可能性较低）**：target 全部在窗口内定义，不需要更长上下文；
  本轮没有证据支持它是主因，但不能排除。

**建议：停止 Task A 路线的继续调参**（增加容量/epoch 只会强化统计捷径）。若要继续 representation 研究，
应先换"窗口级、必须全局推理"的监督（例如给定两段，预测相对顺序/相对变换，head 只能用 pooled），
或先回到表示粒度问题（更长窗口/多尺度）再做任务设计。
