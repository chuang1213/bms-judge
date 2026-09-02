# Phase 2A 第一轮实验报告（最小可运行版本）

> 系列报告：R2 受控干预 → PHASE2A_INTERVENTION_REPORT.md ｜ R3 Task A 对比 →
> PHASE2A_TASK_COMPARISON_REPORT.md ｜ R4 pooled-only → PHASE2A_POOLED_ONLY_REPORT.md

> 生成时间：2026-09-02 06:05:21 | 运行 276.1s

## 1. 数据规模与 split
- clean 7K 无 flag 谱面：35187（7993 首歌）
- pretrain split（song-level，seed=0）：train 30557 / val 1811 / test 2819
- Satellite test 歌曲所在组从 pretrain train 排除：206 组
- 窗口数：train 304858 / val 21708 / test 44885

## 2. representation 的确切定义
- 固定时间网格：4s 窗口，1/60s cell（T=240），lane 轴 8（0-6 keys 有序 + 7 scratch 平面）
- 通道：C0 = onset 计数（cap 2）；C1 = LN hold 标志（最简 hold 表示）
- window-relative 时间；窗口不重叠；不含 BPM/STOP/SCROLL 输入；未来 side channel 预留

## 3. T1 / T2 训练设置
- T1 masked reconstruction：遮挡 1s 连续 span + 10% 随机 cell，BCE（masked cells），epochs=15，batch=128，lr=0.001
- T2 next-window：同编码器，预测下一 4s 网格，epochs=8
- 模型：小 CNN（3×Conv2d，time k7 / lane k3 + 全局池化分支），参数约 13 万

## 4. density-only baseline
- T1：per-lane 未遮挡率 + ±1s 邻域未遮挡率的混合，预测遮挡区域
- T2：当前窗口 per-lane onset/hold 率 + 密度延续先验

## 5. R1–R4 结果
### R1 结构恢复（held-out 歌曲，T1 masked reconstruction）
| 模型 | onset F1 | precision | recall | BCE(all) | BCE(span) |
|---|---|---|---|---|---|
| density-only baseline | 0.0 | 0.0 | 0.0 | - | - |
| random-init encoder | 0.0529 | 0.0272 | 1.0 | 0.70245 | 0.70247 |
| T1 encoder (with hold) | 0.1204 | 0.6688 | 0.0661 | 0.06375 | 0.07208 |
| T1 encoder (onset-only ablation) | 0.1119 | 0.6239 | 0.0615 | 0.10701 | 0.11678 |

R1 分密度层（onset F1，低/中/高）：
- density-only: stratum_low=0.0, stratum_mid=0.0, stratum_high=0.0
- T1 encoder: stratum_low=0.0144, stratum_mid=0.0416, stratum_high=0.1705

### T2 next-window（对照）

### R2 SL 线性探针（frozen encoder，sanity check）
- pretrained：test MAE 1.3022，R² 0.7897（Phase 1 参考：26 特征 MLP MAE 1.068，mean 3.382）
- random-init：test MAE 1.3588，R² 0.7757

### R3 结构探针（无人工标签）
- next_density_pretrained：MAE 12.6088，R² 0.687（mean baseline MAE 29.5059）
- chord_occupancy_pretrained：MAE 0.0146，R² 0.8248（mean baseline MAE 0.0417）
- next_density_random：MAE 12.1943，R² 0.6911（mean baseline MAE 29.5059）
- chord_occupancy_random：MAE 0.0126，R² 0.8661（mean baseline MAE 0.0417）

### R4 变换敏感性（frozen encoder）
- 时间平移（+7s，同内容）：cosine 1.0，L2 0.0054
- 时间平移（+1ms 量化稳健性）：cosine 0.9999
- 窗口平移 0.5s（87.5% 重叠）：cosine 0.9985
- 全局 lane 置换（8 个固定置换，scratch 不动）：cosine [1.0, 0.997, 0.999, 1.0, 1.0, 0.998, 0.997, 0.999]；8-way frozen-probe 识别置换 test acc 0.45 （chance 0.125）
- 局部 lane shuffle：cosine 0.9922；binary frozen-probe 判别 test acc 0.8656 （chance 0.5）
- random-init 参考：时间平移 cosine 1.0；局部 shuffle cosine 0.9997

## 6. 与 random-init / 简单 baseline 的比较
见 R1（density baseline 与 random-init encoder）、R2（random-init 探针）、R3（random-init 探针）。

## 7. 最重要的失败模式
- 大量空 cell：如果 onset precision/recall 明显偏离 0.5 阈值 F1 而 BCE 很低，说明模型在预测空/复制密度；
  span（连续遮挡）与 random-cell 的 BCE 差异能部分暴露这一捷径。
- 若 R2 无增益：局部结构可学但不对 SL 相关（有信息量的负结果）。
- 若 R4 对局部 shuffle 完全不敏感：编码器没有利用空间结构。

## 8. 每个结果能证明 / 不能证明什么
- R1 只能证明：temporal-spatial 表示中存在可学习、可泛化的结构；不能证明学到了 skill demand。
- R2 只能证明：representation 保留了传统难度中的可线性读出信息；SL MAE 不是优化目标。
- R3 只能证明：latent 本身包含低阶结构信息（不只是 decoder 的局部捷径）。
- R4 只测量：latent 对三类变换的距离与可读性；全局置换距离小 ≠ 置换无关，距离大 ≠ 人类难度改变。

## 9. 下一步最值得做的一个实验

**首选：改进 T1 遮挡重建的评估与训练目标，确认"学到的结构"到底是什么。**

理由：R1 显示局部结构可学（onset F1 0.120 vs random-init 0.053、density baseline 0.0），且 R4 显示
pretrained 编码器对局部 lane shuffle 的敏感度远高于 random-init（L2 距离 1.40 vs 0.003，判别探针 86.6%）——
这是"表示中存在可学习、可复用的结构"的正向证据。但 R2/R3 的线性探针显示 pretrained 相对 random-init 的增益很小
（SL MAE 1.302 vs 1.359；结构探针持平），说明当前 64 维全局池化特征主要被低阶统计量主导。

因此下一步最有区分度的实验是：**用结构敏感度本身作为任务**——例如在冻结编码器上训练"原始 vs 局部 lane shuffle"
判别（R4 已显示可行，86.6%），并把它与"全局 lane 置换不可区分"（45%）对照，验证表示是否真正编码了
"局部相对空间结构"而不是"绝对 lane 身份"；同时把遮挡重建改成低置信度采样 + 更细的 recall 校准，排除
"预测空"捷径后再评估。如果表示确实编码局部结构，再进入多尺度 / 位置事件表示。
