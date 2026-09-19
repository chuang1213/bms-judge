# PHASE4_READINESS.md — Phase 4 开发准备

> 写于 2026-09-11 晚，Phase 3 封存时点（tag `phase3-complete`，`recommend` 分支已并回 `phase3`）。
> 给下一个 session 的入口：本文 + `AGENTS.md`（工作指南）→ `PROTOCOL.md`（协议）→
> `EXPERIMENT_LOG.md`（台账）。所有数字的口径与配置见台账对应条目。

## 0. Phase 3 冻结快照

| 项 | 状态 |
|---|---|
| 数据 | 18 名玩家（beatoraja）、**五表围栏**（sl/st/発狂2018/normal☆/overjoy★★）、**14,011 行**（train 7,001 / test 7,010）、c_jrank 在列；语料库 36,464 序列 |
| 最优模型 | **`B_full`：acc 5.801 / cR² +0.463 / lamp 1.102（QWK 0.750）/ BP 109.9** |
| 特征 | 单一事实来源 `chart_repr.BEST_FEATURES`（169 维）= 27 客观 + 12 手工历史 + 三目标响应/偏差块（**180d 半衰期已设为默认**）+ MSD 轴响应/偏差块 |
| 日常基线 | `compare_nolevel.py` 的 `B_full` 行，与 `response_msd.py` 逐位一致 |
| 增益链路（旧空间口径） | B 7.009 → +响应块 6.495 → +180d 遗忘 6.360 → +dev 6.117 → +MSD 轴 5.893；五表空间 5.801 |
| 测试 | **52 全绿**（`python -m unittest discover bms_ml/tests`） |
| 推荐工具 | `recommend.py --serve`：拖入 scorelog.db → 本地打分 → 分组/通过概率（校准过）/80% 区间（CQR，宽度随玩家方差 corr +0.846）/表×等级格子筛选；**overjoy 推荐侧屏蔽**；新玩家零样本可用 |
| 迁移 | 人群预训练+个体 conditioning 在 18/18 玩家胜 D-local，few-shot 效率 3-5×（旧空间口径） |

**口径警告**：跨围栏版本（12,859 → 12,725 → 13,446 → 14,011）的数字不可直接比较；
引用历史数字必须带台账里的配置说明（半衰期、特征集、围栏版本）。

## 1. 已关闭的问题（负结果与红线，勿重复投入）

| 问题 | 结论 | 证据 |
|---|---|---|
| 难度表等级作特征 | **红线**：被客观 kNN 完全替代，禁入特征 | PHASE3_1 / PROTOCOL §1 |
| perm_space（排列空间手部几何） | 被 v2 支配，叠加无增益 | EXPERIMENT_LOG 2026-09-11 |
| framework paper 的 felt-time | 不适用（样本空间内 NPS 零膨胀） | EXPERIMENT_LOG |
| learned state（C0）"每个 k 都赢" | chart 尺度 bug 修复后退化为噪声级 | c0_state_hgb 修复记录 |
| 表外谱计入历史（HISTORY_USES_OFFTABLE） | **边际，未采纳**（B_full −0.077 且 BP 变差） | cmp_offtable.json |
| 条件基底异质性 → 混合集成 | 无收益（噪声级） | _blend_test |
| lamp isotonic 后校准 | **阴性**（校正噪声 > 收益） | EXPERIMENT_LOG |
| 多变量 ridge 响应（11 轴联合） | **阴性**（不如单变量族） | response_ridge.json |
| LR2 与 beatoraja 混池 | **禁止**（标签尺度差 sd 10.16pp > 模型自身误差；data.py 有守卫） | PROTOCOL §1 |
| 裸 MSD 难度列 | **有害**（6.250）；MSD-only 条件化基底反而更优（5.996）→ 玩家相对化是前提 | response_msd.py |
| dev 扩展到其余 20 个统计 | acc 不显著（p=0.17），lamp/BP 改善；仅作 lamp/BP 增强器 | response_devfull.py |

## 2. Phase 4 候选方向（按价值排序）

### A. 数据扩容 —— 真正的天花板
所有建模方向在 2026-09-11 已扫过一轮；样本空间是唯一没动的根本变量。
- **normal2 / ln 表准入**：解析器就绪（`data.py` 的 `admit` flag，一条命令 A/B）。
  注意点：normal2 等级带 +/- 后缀（已剥离入库）；**ln 的 MSD 用 note-starts 计算，
  对 LN 谱只具指示性**——准入 ln 前先验证其 MSD 轴与 LN 谱首打的关系；
- **新玩家 / ST4-12 中高段玩家**：接档 SOP 见 `PHASE3_3_READINESS.md`（ingest → players.json
  → data.py 四步）；few-shot 区间（k=5..50）的结论需要真实短历史玩家复验。

### B. few-shot / 新玩家 conditioning（部署关键区间）
- 现状：响应/dev 块在 k=5..50 **有害**（`transfer_results_resp.json`：k=5 +3.01）；
  MSD few-shot profile 已测（`transfer_results_msdresp.json`）；现有 `RESP_LAMBDA=20` 收缩只补回一半；
- 入口：`transfer_eval.py`（k 网格，18 玩家 holdout）；
- 思路：按 k 门控特征块（k 小时整体关闭响应块）/ 按有效样本数自适应收缩 / 人群先验+个体残差两段式。

### C. 推荐方向：矩阵补全（ALS）
- 契合点：**LR2 存档没有时间戳与首打语义，但这正是 ALS 分数/通过模型的输入形式**
  （`ref_repo/AlphaOSU-main`：fast ALS + pass 模型 + 推理，配 `recommend.pdf`）；
  `lr2_reader.py` 已就绪（6,439 行最佳成绩、84.2% 可关联语料库）；
- beatoraja 侧同样可组 (player × chart) 最佳成绩矩阵；
- 产出形态：**排序质量对比**（NDCG@k / Recall@k，与 B_full 的预测排序比），不是 MAE；
- 这是"预测"转"推荐"的最短路径，也是 LR2 数据的正确入口。

### D. 不确定度与高方差玩家
- 现状：CQR 80% 区间已上线（覆盖 0.772、宽度随玩家方差 corr +0.846）；
  **不存在失效玩家**——corr(逐玩家 MAE, 玩家内 std) = +0.882，MAE/std 中位 0.45；
- 方向：按方差分层的推荐阈值（高方差玩家用区间而非点值做分组）；分层评估进日常基线。

### E. 表示深水区（低优先）
- LN 谱面的专用表示、note-sequence encoder——phase2a 的教训在案
  （pretraining 增益未超 random-init），投入前先设计受控对照。

## 3. 建议的第一步

1. **继续 beatoraja 主线** → normal2 admit A/B（五分钟、一条 flag、按既有台账格式记录）；
2. **转推荐方向** → 先跑 ALS baseline：beatoraja 矩阵 + LR2 矩阵合并分解，NDCG@10 对比 B_full 排序；
3. **收数据** → 按 SOP 接 1-2 名新玩家，重跑 `compare_nolevel.py` 验证 B_full 稳定性。

## 4. 入场检查清单

- [x] 52 测试全绿；`B_full` 参照行与实验逐位一致
- [x] `EXPERIMENT_LOG.md` / `PROTOCOL.md` / `AGENTS.md` / `PROJECT_ARCHIVE.md` 与代码同步
- [x] `recommend` 分支并回 `phase3`，tag `phase3-complete`
- [x] 散落诊断脚本归档至 `bms_ml/output/phase3/audit/`（索引见其中 README）
- [x] 推荐工具运行手册：`python bms_ml/phase3/recommend.py --serve`（模型缓存，首次 ~18s / 后续 ~5s）
