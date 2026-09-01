# PHASE 2A —— Chart Representation v1 设计 memo

> 日期：2026-09-02 | 状态：设计待审（尚未实现）
> 前序：`PHASE1.md`（Phase 1 封存）、`PROJECT_ARCHIVE.md`、`handoff.md`
> 本 memo 只回答问题与设计，不包含实现代码。

> **状态更新（2026-09-02 晚）**：本设计已实现并完成四轮小规模实验，报告：
> R1 首轮（网格 + T1/T2）→ `PHASE2A_REPORT.md`
> R2 统计受控干预 → `PHASE2A_INTERVENTION_REPORT.md`
> R3 Task A 对比 → `PHASE2A_TASK_COMPARISON_REPORT.md`
> R4 pooled-only → `PHASE2A_POOLED_ONLY_REPORT.md`
> 当前状态：4s×1/60s 网格可承载排列信息，但 T1 / Task A（含 pooled-only）的 pretraining
> 均未让 pooled 表示获得超过 random-init 的结构增益；进度归档见 `PROJECT_ARCHIVE.md` §11。

## 0. 一句话

研究问题：**能不能从谱面本身（不依赖人工 skill 维度、不把 SL 当目标）学习出比传统标量难度更丰富的 chart representation？**

本 memo 决定：用什么底层表示（Representation v1）、用什么自监督任务、第一轮实验怎么做、以及如何用成绩单区分"学到了结构"与"只是重建得好"。

---

## 1. 对当前研究问题的理解

Phase 1 已经确立的事实（不再重复研究）：

- 26 个 chart-level 统计特征在严格 song split 下 R²≈0.88（test MAE ≈1.0–1.07）；
- 序列 CNN 在 delta_t 表示下追平统计特征（1.02–1.09），但没有超过；
- 简单序列聚合统计不携带超出 26 特征的独立信息（偏相关 ≤0.14）；
- lane shuffle 对 SL 回归无可检测影响——但这只是"当前模型不敏感"，不能推出"lane 排列没有信息"；
- Phase 1 最大未知：**在密度之外，note 排列顺序里到底有没有当前模型没拿到、但真实存在的结构信息**。

Phase 2 把这个问题换一个问法：**谱面局部结构本身是否含有可学习、可复用的结构信息**——无论 SL 是否编码了它。

工作假设：玩家理解谱面 = 观察一个连续时间窗口内、不同空间位置上即将到来的 note。因此"固定时长窗口内的 temporal-spatial note arrangement"比 measure、比整谱统计更接近玩家的输入。

重要边界（本阶段始终生效）：

- SL、Framework 七轴、人类口语维度都是**外部观察工具**，不是 representation 的定义；
- 本阶段目标不是"预测 SL 更准"，而是"让模型先自己学结构，再拿外部工具做 sanity check"；
- "重建得好"≠"学到了对 skill demand 有意义的表示"——必须靠成绩单来证伪。

---

## 2. Representation v1 候选方案

三个候选：

**A. 事件序列（Phase 1 的 N×4 延续）**
每行 `(window-relative time, lane, type, duration)`。
优点：无损、紧凑、可直接复用现有序列。缺点：和弦被拆成多行（同时性隐含在 delta_t=0）；固定 event 数窗口覆盖可变时长；lane 只是整数 id，模型必须自己学会"相邻 lane 在空间上相邻"。

**B. 固定时间网格（推荐）**
窗口切成等长时间 cell（v1：1/60s），每个 cell 有 lane 轴（7 key + scratch 平面），值为 onset 计数 ∈ {0,1,2+}。
优点：同时性显式（同一 cell = 同时）；时间轴天然 window-relative；lane 轴保持空间顺序，卷积/注意力可直接学到相邻关系；固定大小、实现简单；接近"玩家看到的局部"（2D 滚动视野）。缺点：量化有损（16.7ms cell 可能拆开极快 jack）；网格稀疏（大部分 cell 为空）。

**C. 位置事件（chord-grouped position sequence）**
把时间点聚成 position（16.7ms 窗口内 = 同一位置，与 Framework 的 chord window 一致），每个 position = `(t, chord mask, per-lane duration)`。
优点：同时性显式且无损、紧凑、Pos/s 分解自然（NPS = Pos/s × avg chord size，Framework §3.5 的洞察）。缺点：变长序列需要序列模型；实现与评估比网格复杂。

**推荐 B（网格）作为 v1 主方案**，理由：

1. 时间与空间都编码成坐标轴，"相对空间关系"（相邻、距离、左右手）成为模型结构的一部分，而不是隐式任务；
2. 同时性显式，mask 类任务可以直接问"这一时刻这一 lane 有没有 note"；
3. 固定大小，第一轮实验最简单，最能排除"模型容量/表示工程"的干扰（Phase 1 的教训：只把 absolute time 换成 delta_t，CNN 就从 2.62 降到 1.09——表示工程先于模型工程）；
4. 与 osu! mania 的"ordered lanes + hold"结构同构，不锁死 BMS。

C 作为 v1.1 的对照/替代（若网格量化被证明丢失关键结构）；A 作为无损原始数据保留（网格由它派生，可随时重建）。

### 2.1 设计决策表

| 设计问题 | v1 决定 | 理由 / 备选 |
|---|---|---|
| 窗口长度 n | **4s 主尺度** | 覆盖局部 pattern（0.2–2s）与短乐句；中位谱面 13 NPS 下约 50–90 个 onset（p50–p90 密度），信息量足够且网格很小（240×8）。多尺度 1s/4s/16s 留到 v1.5 |
| 时间表示 | **window-relative**（窗口内 0..n） | 平移不变；Phase 1 已证明 absolute time 是表示问题。chart-relative 位置（前/中/后段）作为可选 side channel 保留 |
| note 字段 | onset 时间、lane 位置、onset 类型（tap / LN-start） | LN 本体、keysound、velocity 等暂不进入（见 §2.3） |
| lane 表示 | **有序 lane 轴（K=7）+ 独立 scratch 平面**；绝对身份（scratch 标志、左右手分组）作为可加通道 | 相对空间关系由空间轴自然支持；scratch 的机械特殊性由平面区分；两者可独立消融 |
| 同时 note | 同一 cell = 同时；per-cell per-lane onset 计数 {0,1,2+} | cell 与 Framework 的 16.7ms chord window 对齐；2+ 保留极快重复（jack）信息 |
| 窗口 overlap | **不重叠**（stride = n） | 评估干净；重叠只作为 pretrain 数据增广旋钮 |
| 前后 context | v1 窗口自包含，无跨窗口输入 | "预测下一窗口"是任务设计而非输入设计；全局 context embedding 留到 v2 |
| osu! 兼容 | 底层 = `(time × ordered lanes)` 网格，K 可配；scratch = "特殊 lane 标记"而非硬编码 | mania 4K/7K 直接映射；SV/scroll gimmick 与 BMS BPM/SCROLL 走同一条"时间扭曲 side channel"（§2.3） |

### 2.2 一个窗口长什么样（v1 tensor）

```
[T=240, L=8, C=2]   # 4s，1/60s，7 keys + scratch
C0: onset（tap 或 LN-start）计数 ∈ {0,1,2+}
C1: 预留（v1 全 0；LN hold / dense 标记的扩展位）
```

### 2.3 暂时排除的信息 + 重新加入路径

**原则：暂时不加入 ≠ 从设计上删除。** 被排除的信息都以"旁路 side channel"记录，且不改主表示结构。

| 排除项 | 为什么先排除 | 重新加入路径 |
|---|---|---|
| BPM 数值 / BPM change | 位置已由 timeline 解成秒；喂 BPM 数值会让模型偷学"gimmick 检测"而非结构 | 每 position 局部 BPM 列 + chart 级 BPM 剖面；Framework 的 felt-time 修正作为时间轴校正 |
| STOP | 干净谱面中很稀疏（p99=2），v1 忽略 | 时间轴 gap 事件 / 每 position "前有 STOP" 标志 |
| SCROLL/SPEED | 显示层 gimmick，改变"读谱速度"而非 note 布局 | 每 lane 每 time 的 scroll 因子通道（对应 osu! SV） |
| LN 本体（hold duration） | 按约定暂不进入（见下方"设计张力"） | 每 lane hold 通道（C1 预留位），零架构改动 |
| keysound / 音量 / BGA / OPTION / 地雷 / 隐形 note | 作者风格层，非通用结构 | 每 note 附加属性列 / chart 级元数据 |
| 2P/DP、PMS、控制流 | corpus 已隔离；v1 只研究 SP 7K | 换 K 与平面数即可扩展 |
| RANDOM/MIRROR 选项 | 玩家看到的有效谱面 = 基础谱 + permutation | §5 的"全局 lane 置换探针"就是这条路的起点 |

**设计张力（需要你拍板）：** LN 在"BPM、STOP、LN"清单里被归为 gimmick。严格说 LN 不是 gimmick 而是 note 类型——hold 是 osu! mania 也有的一般机制，且约 1/3 的干净谱面含 LN。v1 按你的指示只把 LN-start 当 onset（LN 谱会变成稀疏 tap 谱，属刻意简化）；但请知道：把 hold 通道打开是零架构改动的一步，建议放在 v1.1 与 cell size 一起做，而不是永远排除。

### 2.4 多尺度（v1.5，非 v1）

- 阅读/运动规划视界约 0.2–3s，乐句约 4–16s，整曲 60–200s；不存在单一正确尺度。
- v1 只做 n=4s 单尺度，先回答"局部结构是否可学"。
- v1.5：1s（微 pattern：jack/trill/stair 片段）/ 4s（主）/ 16s（乐句与段落）三尺度独立编码 + 融合，或对 4s 网格做时间下采样构成金字塔。是否值得做，取决于 v1 成绩单。

---

## 3. 自监督任务候选（比较 2–3 种）

### T1 遮挡重建（masked reconstruction）—— 推荐主任务

- 做法：对窗口网格挖掉一段连续时间跨度（约 1s）+ 随机 10% cell，用剩余部分预测被挖 cell 的 onset（多类：空 / 单 onset / 2+）。
- **可能学到**：局部节奏连续性、和弦形状、lane 相邻/楼梯结构、典型密度剖面。
- **学不到**：与局部统计无关的全局语义（疲劳、段落结构、作者风格）；重建好 ≠ 对 skill demand 有意义（必须靠 §5 成绩单证伪）。
- 主要陷阱：大多数 cell 是空——模型可能靠"预测空"拿高准确率。对策：结构式遮挡 + 密度分层评估 + 边缘基线（每 lane 空率、密度延续）。

### T2 下一窗口预测

- 做法：给前 4s 窗口（可加少量历史），预测接下来 4s 的 onset 网格（或更粗的下窗口密度剖面）。
- **可能学到**：窗口间转移——乐句/段落变化、重复结构、密度轮廓。
- **学不到**：本地不可预测的谱面；并且会被"同曲差分共享乐句"的 corpus 偏差污染（必须 song split）；很容易退化成"密度延续"基线。作为次要任务/探针，不作为主任务。

### T3 对比 / 变换判别（留到 v2，v1 先作为探针）

- 实例判别（SimCLR 式）：正样本 = 同一窗口的增广，负样本 = 其他窗口。**风险**：增广集合决定不变性，而我们现在不知道正确的不变集——绝不能预设"对局部 lane shuffle 不变"。
- 变换判别：给定窗口 + 一种程序化变换（局部 lane shuffle、时间扭曲、和弦拆分），预测是哪种变换。这直接测试编码器对特定结构属性的敏感度；v1 先作为冻结编码器探针（§5-R4），作为训练目标留到 v2。

**为什么 v1 选 T1：** 直接检验"窗口内结构可学"（本阶段核心问题），无标签、便宜、可解释；T2 作对照回答"结构是否延伸到窗口之间"；T3 的结论完全取决于还没定下的增广设计，v2 再做。

---

## 4. 第一轮最小实验

### 4.1 数据与划分

- 数据：**35,187 张 clean 无 flag 7K 谱面**（沿用 Phase 1 的 quarantine/flag 过滤），不做人工挑选；
- 全部作为无标签 pretrain 数据（自监督红利：不再局限于 1,845 张 Satellite 标签谱）；
- song-level split（复用 `split.py` 的分组逻辑，扩展到全部约 8.3k 首歌）：约 90/5/5；
- **隔离约束**：Phase 1 `split.json` 中的 Satellite test/val 歌曲必须从 pretrain 训练集排除，保证 R2 探针是真正的"未见谱面"；
- 窗口：n=4s、cell=1/60s、stride=4s，从现有 sequence npy（N×4）派生，不存储大文件（按谱面流式构建）；
- 量化审计：报告同 cell 多 onset 占比、同 lane 跨 cell 拆分占比（量化质量底线）。

### 4.2 模型

- 小 CNN（2D：time × lane），参数量与 Phase 1 同量级（约 10–50K）；
- 第一层时间核 k≈7、lane 核 k≈3（可覆盖约 100ms 内相邻 lane 的局部 pattern）；
- 不用 Transformer、不做超参搜索；3 seeds。

### 4.3 训练目标

- T1 遮挡重建（主）：结构式遮挡 + 随机 cell，交叉熵/BCE；
- T2 下一窗口（对照）：同编码器、同数据，预测下窗口 onset（或密度剖面）。

### 4.4 成绩单（R1–R4）

| 成绩单项 | 具体做法 | 它能证明 | 它不能证明 |
|---|---|---|---|
| R1 结构恢复 | 遮挡重建在 held-out 歌曲上的 cell 级指标，按密度分层；对照边缘基线 | 局部结构可学习、可泛化到未见歌曲 | 重建好 ≠ 对 skill demand 有意义 |
| R2 SL sanity | 冻结编码器 + 线性探针 → Satellite SL（1,845 子集），对比 Phase 1（B=1.068 / B_window=1.081）与随机初始化编码器 | representation 携带与 SL 相关、可线性读出的信息；pretrain 相对 random-init 的增益 | SL 准 ≠ 学到 skill demand；SL 本身有约 1 级噪声 |
| R3 结构探针 | 无标签探针：下窗口密度预测、遮挡 span 的 chord size 分布预测 | 学到可泛化的时序/结构特征 | 探针任务本身是人为聚合 |
| R4 稳定性/敏感性 | 冻结编码器：时间平移（窗口整体平移）→ 表示距离应≈0；**全局 lane 置换**（固定映射、scratch 不动）→ 测量距离；**局部 lane shuffle**（Phase 1 式）→ 测量距离与探针变化 | 区分"表示对无关变化稳定"与"对结构破坏敏感" | 置换距离小 ≠ 置换无关；距离只是测量，需结合下游判断 |

### 4.5 成功 / 失败判读

- **成功信号**：R1 明显优于边缘基线；R2 线性探针明显优于 random-init（且不是仅靠密度）；R4 时间平移稳定、局部 shuffle 敏感。
- **失败信号**：R1 ≈ 边缘基线（先换 C 位置事件表示再下结论）；R2 无增益（结构可学但不对 SL 相关——也是有价值的负结果：说明局部结构不携带 SL 需要的信息）；R4 对什么都敏感/不敏感（编码器没学会区分结构变化）。
- 所有结论都按密度分层并对照 random-init，避免"能重建=有用"和"SL 没涨=白干"两种错误推论。

---

## 5. 最大风险与 confound

1. **重建捷径**（预测空 / 复制邻居）→ 结构式遮挡 + 边缘基线 + 密度分层；
2. **SL 标签噪声**（Phase 1 MAE≈1 的地板）→ SL 只是 sanity check；
3. **同曲差分共享乐句** → 一切都在 song-level split 内；pretrain 与 R2 探针的 split 必须协调（§4.1）；
4. **16.7ms 量化** → 量化审计 + cell size 消融（v1.1，候选 1/30、1/60、1/120）；
5. **BPM/STOP 被排除但 corpus 里有** → v1 过滤 flag 谱；残余影响小（stop p99=2）；felt-time 作为 side channel；
6. **重叠窗口 inflate 指标** → 评估用非重叠；
7. **网格 ≠ 玩家真实看到的渲染** → v1 不声称验证"视觉输入是否必要"（那是未来 A/B）；
8. **不变性先验污染** → v1 不把任何 lane 不变性写进训练目标，只测量；
9. **LN 谱被稀疏化** → v1 记录 LN 分层结果，v1.1 打开 hold 通道后再复查；
10. **pretrain 数据包含 R2 test 歌曲** → 已用隔离约束处理（§4.1），但报告中仍需分别给出 seen/unseen 的结果。

---

## 6. 建议暂时不做的事

- 不做 SL regression 主目标（只做 R2 探针）；
- 不定义 / 不输出 skill ontology（jack / trill / stair / chord / scratch / stream / LN 都不作为输出维度）；
- 不复刻 Framework 的七个 scalar（只借用输入设计原则：chord window、position 分解、lane 相邻、scratch 特殊、felt-time 思想）；
- 不把 BPM / STOP / SCROLL / LN hold 加入 v1 输入（side channel 设计保留）；
- 不用 replay；
- 不依赖人工挑选或修改谱面（变换全部程序化，只用于 R4 探针）；
- 不直接支持 osu!（只是底层表示不锁死 BMS）；
- 不用复杂模型、不做大规模超参搜索；
- 不做多尺度（v1.5）；
- 不把任何"SL 或 Framework 指标预测得好"当作 representation 有效性的定义。

---

## 7. 与 Phase 1 未知的承接

如果 R1 显示结构可学、且 R4 显示网格编码器对局部 lane shuffle **敏感**（而 Phase 1 的事件 CNN 不敏感），这将是第一个正面证据：Phase 1 的 lane shuffle 负结果更可能是"模型/表示局限"而不是"lane 排列没有信息"。反过来，如果 R4 显示完全不敏感，则"当前模型类不利用空间结构"这一结论得到强化。两种情况都有研究价值——但注意 R2 不参与这个判断（SL 是否编码空间信息是另一个问题）。
