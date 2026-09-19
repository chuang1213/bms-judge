# PHASE4B_CLOSING_REPORT.md — Phase 4B 封存报告

> 2026-09-12。本报告封存 Phase 4B / cold_chart。
> 协议：`PHASE4_PROTOCOL.md` §11；计划：`PHASE4B_PLAN.md`；调查：`PHASE4B_REFERENCES.md`；
> 特征实验：`PHASE4B_FEATURES_REPORT.md`；M3 负结果：`PHASE4_M3_REPORT.md`。

## 0. 一句话结论

**Phase 4B 封存。** 主特征集定格为 **`obj + msd`（33 列）**，beatoraja 冷谱面难度
**MAE ≈ 5.9**（同一批 38,042 行上，稳定优于 26 统计基线约 **+0.81**），
玩家×谱面交互门通过。三件收尾实验证明：**特征集已到 100% 覆盖、调参不是瓶颈、v2 没有边际价值**。
**当前数据下的成绩预测天花板约在 MAE 5.9。下一步最有价值的是补数据，不是换模型。**

## 1. 最终特征集

**`obj + msd` = 26 个客观谱面统计 + 7 个 MinaCalc MSD 轴。**

| 来源 | 列数 | 覆盖 | 文件 |
|---|---|---|---|
| 客观统计 `cs_*`（= `OBJECTIVE_STAT_COLS` 去掉 `c_jrank`） | 26 | **100%** | `cross_section.parquet` |
| MinaCalc MSD（**打过上限补丁**） | 7 | **12,290 / 12,292 = 99.98%** | `msd_cap100.parquet` |

- **不用难度表等级**（`c_jrank` 始终排除），**不用时间信息**，**不混 LR2/beatoraja**。
- 缺失值保持 **NaN**，从不填 0 或均值。
- `msd_cap40.parquet` 作为对照保留，**与 cap100 分文件、永不混列**。

### 1.1 收尾实验 A：MSD 补到 100%（完成）

原缺口是 **98 张谱没有 note sequence**。原因不是解析失败，而是
`build_corpus_manifest.py` **只为"干净"（无 quarantine 标签）的谱面保存 sequence**，
所以所有被隔离的谱面都没有——尽管它们的源文件能正常解析、也有玩家在上面有成绩。

- 98 张的来源：`control_flow` 69、`not_single_play` 24、`mgq_ln` 5
- **成功补齐 98/98**（只新增、不覆盖既有 sequence）
- MSD 覆盖：**12,190 → 12,290 / 12,292 = 99.98%**
- MSD 成本：7,038 张新谱用 **4.9 秒**；全量 12,292 张重算用 **8.9 秒**

**剩余 2 张无法计算的谱（明确列出）：**

| sha256 | 原因 |
|---|---|
| `5f98e6da85ef…` | **只有 1 个音符**（keycount 8，1 行）→ MinaCalc 返回 0 |
| `ae3af0457ee6…` | **只有 1 个音符**（keycount 8，1 行）→ MinaCalc 返回 0 |

这两张是退化谱面（单音符），MinaCalc 主动拒绝，**不是我们的 bug**。它们保留 NaN。

### 1.2 收尾实验 B：GBDT 嵌套 CV 调参（完成）

网格 24 组 × 3 内折，**只在 train/validation 上调参**，测试集每个 seed 只碰一次。

| | 未调参 | 调参后 | 增益 |
|---|---|---|---|
| beatoraja（38,176 行） | 5.9575 ± 0.1368 | 5.9396 ± 0.1464 | **+0.018** |
| LR2（2 玩家，仅记录） | 9.9721 ± 0.3489 | 9.9950 ± 0.3075 | −0.023 |

- 增益 **0.018 < seed 波动 0.137**；LR2 上甚至变差。
- 选择的配置高度一致：`max_depth=6`（比默认 3 深），`lr=0.04`，`max_iter=200`。
- **判定：调参不是瓶颈。正式接受未调参的 ~5.96。**

### 1.3 收尾实验 C：v2 同分布边际检查（完成）

**不把 v2 放进 common 定义**（那会把共同行从 38k 压到 10k 并换成更容易的子集 ——
这正是之前踩到的坑）。改为 `obj+msd` 与 `obj+msd+v2` **在 v2 可用的同一批行上**
（beatoraja 10,080 行 = 26.4%）比较。

| 模型 | 基线 | +v2 | 增益 | 逐 seed | 判定 |
|---|---|---|---|---|---|
| **GBDT（主模型类）** | 6.1525 | 6.1524 | **+0.0001** | +0.041 / −0.074 / +0.033 | **关闭 v2** |
| Ridge | 6.4559 | 6.2456 | +0.2103 | 全部为正 | keep |

- **主模型类 GBDT 上 v2 的增益是 0.0001，且 3 个 seed 方向不一致。**
- Ridge 上的"稳定增益"说明 v2 与线性模型有交互，但 **GBDT 已经吃掉了这部分信息**。
- 交互门：GBDT 的玩家内 MAE 变化 −0.015（不显著）→ 同样关闭。
- **判定：正式关闭 v2。** 按用户决定，可作为附录记录，不进主线。

## 2. 两道门的最终结果（beatoraja，共同层 38,042 行 / 16 玩家）

主特征集 `obj+msd`，GBDT：

| 门 | 指标 | 结果 |
|---|---|---|
| **第一道门** 冷谱面难度 | 谱面级 MAE | **5.980**（参考基线 content_ridge 6.793，**增益 +0.813，稳定**） |
| **第二道门** 玩家×谱面交互 | 玩家内 MAE | **5.736 → 5.170**（交互模型，增益 +0.566） |
| | 玩家内 Spearman | 0.543 → 0.616（+0.073） |
| | 置换检验 | 打乱内容↔玩家配对后变差（回到非交互水平） |

**两道门通过，且未被判为 `difficulty_only`（即交互是真的，不是只做了难度预测）。**

LR2（2 名玩家）数字同向但**不作统计判定**。

## 3. 负结果登记（全部封存，不再投入）

| 方向 | 结论 | 证据 |
|---|---|---|
| **M3 矩阵补全 / 协同过滤** | **失败**。带偏差的 ALS 没有稳定超过门基线，玩家内排序未改善，测试集作弊调参也不算通过 | `PHASE4_M3_REPORT.md` |
| **perm / panchira 排列空间** | **与客观统计重复**。单独用几乎等于基线（增益 +0.012，不稳定）；加到 `obj+msd` 上反而变差 0.014 | §2.1 特征实验 |
| **v2 客观统计 v2** | **无边际价值**。GBDT 上增益 0.0001，seed 方向不一致 | §1.3 |
| **trills[21]** | **未移植**。按用户决定，等主结果后再定；主结果显示边际收益已快速衰减 | `PHASE4B_FEATURES_REPORT.md` §5 |
| **framework 七轴** | **不实现**。6–8 天工作量、触碰"禁止手工技能轴"红线、且**本项目没有 character 真值无法验证实现正确性**；felt-time 已被 Phase 3 证不适用 | `PHASE4B_REFERENCES.md` §C |
| **note-sequence encoder** | **本阶段不做**。属独立高风险路线；若做须另立 Phase 4C 并设定门禁 | 用户决定 3 |

## 4. 当前数据下的预测上限

**约 MAE 5.9**（beatoraja 冷谱面难度，共同层）。

**这个数字在成绩表上是什么意思**：预测"某人在一张他从没打过的谱上能打多少分"
（满分 100），平均差 **5.9 个百分点**。作为对照，只会背"这人平均水平"的模型差 6.6–6.8 个百分点。

三条独立证据表明这不是调参或特征工程的问题：

1. **调参**：嵌套 CV 只带来 0.018（< seed 波动）→ 不是瓶颈。
2. **特征**：唯一的稳定新信息是 MSD；perm 与统计重复、v2 无边际价值。
   特征覆盖已达 99.98%，没有"还没算出来"的东西了。
3. **模型类**：Ridge 全线不如 GBDT，但把 Ridge 换成 GBDT 的那部分收益
   （约 1pp）在 Phase 4B 早期就已经拿到；再往上没有新的模型类被证明有效。

## 5. 下一步建议（按价值排序）

**核心判断：下一步最有价值的是补数据，不是换模型。**

| 优先级 | 方向 | 为什么 | 代价 |
|---|---|---|---|
| **1** | **更多玩家数据** | 当前 beatoraja 只有 16 名有 `score.db` 的玩家（steve / taffy / buzhang 缺 score.db）。玩家数是矩阵类方法的硬约束；冷玩家与交互门都直接受益 | 接档 SOP 已存在（`PHASE3_3_READINESS.md`） |
| **2** | **更多 LR2 存档** | 现在只有 2 个 LR2 玩家（且 82% 观测不重叠）。LR2 的分数尺度不同但**没有时间戳限制问题**，是最容易扩的人群 | 收集存档即可 |
| **3** | **重复游玩 / 干预记录** | 唯一能打开"重打会不会提高"这个被明确列为非目标的问题的数据。需要 beatoraja `scoredatalog.db` 或更完整的 per-play 日志 | 需要玩家配合采集 |
| **4** | **产品化预测器** | 现在 `obj+msd` 在 99.98% 的谱面上可用、单谱成本毫秒级（MSD 1ms + 26 统计已有）。**可以在不追精度的情况下先做"这张新谱对你大概多少分"的只读预测器** | 小；但注意协议 §11.6：不是推荐器 |
| **5** | Phase 4C learned representation | 只有在 1–4 都做完、或明确要赌表示学习时才做。**门禁：必须稳定超过 `obj+msd` 的 5.9**，否则停止，不进入 M5 | 高 |

**不建议做**：
- 继续堆模型/特征（已证明是瓶颈之外的东西）；
- 启动 M5 推荐器——推荐器应建立在稳定预测模型之后，而当前目标明确是成绩预测，不是推荐器；
- 直接上 note-sequence 深度学习（Phase 2A 的预训练未超随机初始化的教训在案）。

## 6. 封存清单

**最终特征集与数据**

| 文件 | 内容 |
|---|---|
| `output/phase4/dataset/msd_cap100.parquet` | **主用** MSD，12,290 行 × 7 轴 + `msd_cap` 溯源列 |
| `output/phase4/dataset/msd_cap40.parquet` | 原版对照，12,290 行（永不与 cap100 混列） |
| `output/phase4/dataset/cross_section.parquet` | 47,090 行横截面 + 26 个 `cs_*` |
| `output/phase4/dataset/perm_space_full.parquet` | 12,290 行 × 22 列（负结果，留档） |
| `output/phase4/dataset/chart_stats_v2_full.parquet` | 12,292 行 × 14 列（负结果，留档） |

**报告与结果**

| 文件 | 内容 |
|---|---|
| `PHASE4_PROTOCOL.md` | 协议（§11 为 Phase 4B） |
| `PHASE4B_REFERENCES.md` | 三个参考项目调查 |
| `PHASE4B_PLAN.md` | 实验计划 |
| `PHASE4B_FEATURES_REPORT.md` | 特征族实验矩阵 |
| `PHASE4B_CLOSING_REPORT.md` | 本文件 |
| `PHASE4_M3_REPORT.md` | M3 负结果 |

**代码与结果 JSON**

| 文件 | 内容 |
|---|---|
| `bms_ml/phase4b/build_content_features.py` | 特征构建（sat / seq / msd / msd-patched / perm / v2） |
| `bms_ml/phase4b/run_phase4b.py` | Phase 4B 两道门（原始） |
| `bms_ml/phase4b/run_phase4b_features.py` | 特征族实验矩阵 |
| `bms_ml/phase4b/run_phase4b_tuning.py` | 收尾实验 B |
| `bms_ml/phase4b/run_phase4b_v2check.py` | 收尾实验 C |
| `output/phase4/features_build.json` | 构建报告（覆盖率、饱和度、失败清单） |
| `output/phase4/phase4b_features.json` | 实验矩阵结果 |
| `output/phase4/phase4b_tuning.json` | 调参结果 |
| `output/phase4/phase4b_v2_marginal.json` | v2 边际结果 |
| `bms_ml/tests/test_phase4b.py`、`test_phase4b_features.py` | 回归测试 |

## 7. 已知不确定 / 未验证

- **MSD 的 LN 处理不准确**：`msd_prep.py` 把 LN 起点当 tap、忽略终点（Etterna 的 RC 取向）。
  本轮**没有为 LN 谱单独评估**，最终特征集对 LN 谱的有效性未验证。
- **GBDT 的"调参无增益"只在 24 组网格内验证**，不排除网格外有更好的配置；
  但增益量级（0.018）远小于 seed 波动，继续扩大网格的收益期望很低。
- **v2 的负面结论在 GBDT 上成立、在 Ridge 上不成立**（Ridge 上 v2 稳定有效）。
  这不是矛盾，而是说明 v2 的信息与线性模型互补、而已被 GBDT 吸收。
- **LR2 全程只有 2 名玩家**，所有 LR2 数字都不构成统计结论。
- **谱面效应里混着"这张谱有多流行"**（打的人多 → 均值更可靠），本轮未拆开。
