# PHASE4B_REFERENCES.md — 三个参考项目调查（先调查，不写模型）

> 2026-09-12。本文件只做调查与复用判断，**不改任何代码**。
> 服务对象：`PHASE4_PROTOCOL.md` §11 的 Phase 4B / cold_chart，两道门（冷谱面难度、玩家×谱面交互）。
> 配套：`PHASE4B_PLAN.md`（实验计划）。

## 0. 三个项目分别"到底能给我们什么"

| 项目 | 一句话 | 对我们的价值 | 我的结论 |
|---|---|---|---|
| **MSD**（`osumania_map_analyser-main/`） | 把谱面喂给 Etterna 的 MinaCalc，输出 7 轴技能数值 | **可直接扩到几乎全部谱面，成本极低** | **立刻扩覆盖**，这是最高性价比的一步 |
| **Permikon / panchira**（`Permikon-main/`） | 枚举 5040 种键位排列，算手部移动几何；另有 anchors[7] + trills[21] | Phase 3 已经用 Python 重实现并**超出了**它的核心 | **不重写**；只补 2 个小字段（评估是否值得） |
| **Framework paper**（`docs/reference/framework paper.pdf`） | 七轴"谱面性格"框架（chord/stream/scratch/soft/ln/stair/distraction）+ felt-time | 复杂、需自建、且是人工定义的技能轴 | **不实现**；只保留为外部基线/解释候选 |

**一句话总结**：这一轮真正值得做的只有 MSD 扩覆盖；panchira 已经吃到了；framework paper 的成本远高于收益，且碰到红线。

---

## A. MSD — `ref_repo/osumania_map_analyser-main/`

### A.1 精确可复用路径

真实路径含空格，注意引用：

```
ref_repo/osumania_map_analyser-main/
  ManiaMapAnalyser by Leo_Black/js/ett/
    calc.js                    # 调用 WASM 的胶水层：buildRows 等
    constants.js               # 技能集名称、版本表
    index.js                   # 对外入口
    versions/
      minaclac-74.0.wasm       # ← 我们正在用的版本
      minaclac-74.0.js         # Emscripten glue
      minaclac-68.0-unofficial.wasm / 70.0 / 72.0 / 72.3 / 75.0
  tools/patch-minaclac-msd-cap.mjs   # 把 WASM 里 f32.const 40.0 改成 100.0
  LICENSE                      # MIT (c) 2026 Leo_Black
```

- **输入格式**：MinaCalc 的 FFI 只要 `(keycount, rate, score_goal, u32[] rowMasks, f32[] rowSeconds, nRows, out)`。
  **没有 BPM、没有 timing point、没有 LN 终点**——所以谱面必须先被压成"同一时刻的所有 lane"位掩码行。
- **输出字段**：`Overall, Stream, Jumpstream, Handstream, Stamina, JackSpeed, Chordjack, Technical` 共 8 个 f32。
  我们的 `msd_finalize.py` 丢掉 `Technical`：在 n-key 路径上它是常数 0.18（4K 专用），零信息。
  最终 7 轴：`msd_overall / stream / jumpstream / handstream / stamina / jackspeed / chordjack`。
- **依赖**：Node.js（跑 ESM + WASM）。**不需要 Rust、不需要编译**。
- **运行成本（实测）**：**约 1 ms/谱**，2932 张谱 2.0 秒（≈1500 谱/秒，实测 stdout `2500 done (1512.4/s)`）。
  可缓存：中间产物 `charts.bin` → `msd.jsonl` → `msd.parquet` 三段都可复用。
- **许可证**：MIT。可自由使用/修改/再分发，保留版权声明即可。

### A.2 `tools/patch-minaclac-msd-cap.mjs` 是什么，我们要不要用

它把 MinaCalc 内部的技能值上限从 **40.0 拔到 100.0**（等长 5 字节替换，`43 00 00 20 42` → `43 00 00 c8 42`），
理由是超高难谱（高速连打/高密度）的技能值会被 40.0 截平。

- **对我们的影响**：BMS 侧的 sl/st 高段谱确实会撞到 40。**实测确认（2026-09-12）**：
  - 旧范围 5,497 张谱里 **11 张**撞顶（0.20%），集中在 **stella 10 / 11 / 12** 与 **overjoy 6 / 8**；
  - 50 音以上的高段谱里 `handstream` 撞顶最多（10 张）；
  - `spearman(msd_overall, level)`：全集 0.0758，去掉撞顶后 0.0754 → **截平没有扭曲整体排序关系**
    （这个相关系数本来就低，因为难度表等级量的不是同一个东西），
    但它**确实让最难的那几张谱在特征上变成同一个值**。
  - **用户决定（2026-09-12）**：撞顶集中在高段 → **打补丁并全量重算**。
- **它改的是 `ref_repo` 里的 WASM 原文件**。`ref_repo/` 是对照参考仓，**严禁改写**。
  实现方式：把 WASM 复制到 `output/phase4/features_build/ett_patched/` 再打补丁，
  `ref_repo` 保持原样（`build_content_features.py::make_patched_wasm`）。
- **patched / unpatched 分成两个文件，永不混列**：
  `msd_cap40.parquet`（stock，40）与 `msd_cap100.parquet`（patched，100）。
  两者都带 `msd_cap` 溯源列；**主结果用 patched，cap40 作对照**。
- **实测补丁效果**（12,192 张谱，两者逐行比）：

  | 轴 | 改变的谱数 | 最大涨幅 |
  |---|---|---|
  | `msd_overall` | 11 | **+7.60**（41.48 → 48.48） |
  | `msd_handstream` | 7 | +7.90 |
  | `msd_jumpstream` | 4 | +7.56 |
  | `msd_chordjack` | 4 | +7.75 |
  | `msd_stream` | 3 | +3.45 |
  | `msd_stamina` | 1 | +7.28 |

  只有 11 张谱受影响，但**每张都涨 3–8 个 MSD 点**——不补丁会让这批高段谱被压成与
  真实难度不符的值。

### A.3 本项目里已有的重复实现（先复用，别重写）

| 文件 | 作用 | 判断 |
|---|---|---|
| `bms_ml/phase3/msd_prep.py` | note sequence → `charts.bin`（行掩码位打包） | **可复用，只需换输入谱面清单** |
| `bms_ml/phase3/msd_compute.mjs` | `charts.bin` → MinaCalc → `msd.jsonl` | **原样复用** |
| `bms_ml/phase3/msd_finalize.py` | `msd.jsonl` → `msd.parquet` + 外部效度检查 | 可复用，但要加"上限堆积检查" |
| `bms_ml/output/phase3/dataset/msd.parquet` | 5,497 行 × 8 列 | 覆盖不足，见下 |

**这三段就是完整管线，不需要重写。问题不在管线，在输入的谱面清单。**

### A.4 为什么只有 5,497 行（审计结论，已实测）

`msd_prep.py` 的 `chart_set()` 取的是 **Phase 3 的围栏（sl/st/insane 表）∪ 有首打记录的谱**，
那是为 Phase 3 的首打模型准备的范围，**不是 Phase 4B 的范围**。

实测数字：

| 量 | 值 |
|---|---|
| Phase 3 构建目标（表 ∪ 已打） | 6,509 |
| `msd.parquet` 实际行数 | 5,497 |
| 目标里有、但没算出来 | **1,012**（缺 sequence / 键数<4 / 行数<2） |
| **Phase 4B 真正需要的谱面**（cross_section 里 `mode_kind=='sp'` 的唯一 sha256） | **12,292** |
| 其中已有 MSD | 5,254 |
| **需要补算** | **7,038** |
| 在 Phase 4B 谱面里、磁盘上有 note sequence 的 | 12,194 / 12,292（**99.2%**） |

**结论：MSD 覆盖不足是"范围选错"，不是"算不动"。补 7,038 张谱按实测吞吐约 5 秒。**

> 注意：Phase 4B 的谱面宇宙是 cross_section 里的谱（12,292），**不是**整个 7 键语料（36,974）。
> 冷谱面的候选只能来自 cross_section，所以 12,292 就是 100% 覆盖的分母。

### A.5 接入 Phase 4B 的方式

- **作为特征输入**：是。7 轴作为一组特征族 `MSD`，与 26 客观统计拼成 `content+msd`。
- **作为外部基线**：也是。`msd_overall` 单独作基线，用来回答"MSD 是不是已经吃掉了内容的全部信号"。
- **作为数据质检**：顺带。MSD 与难度表等级的单调性可以交叉验证我们的 acc 解析是否正确（不引入等级作特征）。

### A.6 风险和边界

1. **缺 MSD 的谱必须保持缺失（NaN），不许填 0 或均值**——填 0 会让"高难谱"看起来像"没信息"，
   填均值会引入一个假的中等难度。缺失值由 GBDT/Ridge 自己处理（Ridge 需要显式处理，见计划）。
2. **不能拿 MSD 当"最终谱面表示"**。它是外部程序算的、面向 osu!mania/Etterna 的 4K 生态校准的，
   BMS 7K+scratch 只是它 n-key 路径的副产物。它可以是输入，不是结论。
3. **7 轴之间有强相关**（overall ≈ 若干轴的组合），拼进线性模型会共线。需要看方差膨胀或直接上 GBDT。
4. **`Technical` 必须继续丢**（n-key 上是常数）。
5. **不能改 `ref_repo/` 里的 WASM**（对照仓，且在 `.gitignore` 语境外）。要打补丁就复制副本。
6. **LN 处理**：`msd_prep.py` 把 LN 起点当 tap、忽略 LN 终点。这是 Etterna 的 RC 取向，
   对 LN 谱不准确。**这一点要在报告里标注**，不能默认 MSD 对 LN 谱有效。

---

## B. Permikon / Permidex — `ref_repo/Permikon-main/`

**澄清**：`Permikon-main` 就是 Permidex 相关项目——README 明确写它是
"wraps the [Permidex](https://permidex.app) UI and panchira analysis engine in a native desktop app via Tauri 2"。
即：**Permidex = UI/网站**，**panchira = 分析引擎**（`panchira_cli/`），**Permikon = 把两者包成桌面 app**。

### B.1 精确可复用路径

```
ref_repo/Permikon-main/
  README.md                      # 项目定位（见上）
  panchira_cli/                  # ← 分析引擎（Rust），核心在这
    Cargo.toml / Cargo.lock
    src/evaluator.rs             # ★ 核心算法：5040 排列的几何指标
    src/parser.rs                # BMS/BME 解析
    src/bmson.rs                 # BMSON 解析
    src/lib.rs                   # 对外 API
    src/main.rs                  # CLI 入口
    src/db.rs                    # SQLite 缓存
  public/ranker.js               # 前端：把各项归一化后加权排序
```

**panchira 的输出（`PermRow`，`evaluator.rs:41-50`）**：

| 字段 | 含义（读源码确认） |
|---|---|
| `smooth` | 关键位置序列对"位置序号"做**加权线性回归的残差标准差**（越平滑 → 手移动越线性） |
| `tight` | 每个位置**同时按键的跨度**（max lane − min lane）的加权平均 |
| `base` | 加权平均"variability" = \|相邻位置平均 lane 变化\| + 该位置跨度 |
| `spike` | variability 的**加权 p90**（分位数视图） |
| `anchors[7]` | 每个 key lane 的**加权按键占用**（哪个键被按得最多） |
| `trills[21]` | 21 个 lane 对 (a,b) 的**加权交替次数**（a<b，7 选 2） |

**一个关键细节（`evaluate_all`，`evaluator.rs:289-307`）**：
panchira 最后对 `smooth/tight/base/spike` 做**单谱内 min-max 归一化到 [0,1]**。
这是给它的排序 UI 用的，**会摧毁跨谱可比性**——同一个 1.0 在不同谱里含义不同。
作为预测特征这是致命的，**不能照搬**。

- **输入格式**：`.bms/.bme/.bmson` 文件（它自己解析）；或经 `src/db.rs` 的 SQLite 缓存。
- **依赖**：Rust（编译）+ Tauri（桌面壳）。引擎本身只需 Rust。
- **运行成本**：5040 排列 × 每谱位置数；它自己走 Rust 原生循环。
- **许可证**：**没有 LICENSE 文件**（`ref_repo/Permikon-main/` 下确认无 LICENSE，README/Cargo.toml 无 license 字段）。
  这意味着**默认保留全部权利**。→ **不允许把它的 Rust 源码/WASM 直接搬进我们仓库**。
  Phase 3 的做法是**按算法描述用 Python 重新实现**（算法/公式本身不受版权保护），
  这是正确的规避方式，应继续保持。

### B.2 本项目里已有的重复实现：`bms_ml/phase3/chart_perm_space.py`

**结论：Phase 3 的 `chart_perm_space.py` 已经覆盖并超出了 panchira 的核心。**

| panchira 字段 | Phase 3 `perm_space` | 判断 |
|---|---|---|
| `smooth` | `ps_smooth_{mean,std,min,max,base_pct}`（加权回归残差，公式一致） | **已覆盖，且扩展成 5040 排列上的分布** |
| `tight` | `ps_tight_*`（加权跨度，公式一致） | **已覆盖** |
| `base` | `ps_base_*`（\|Δavg\| + span，公式一致） | **已覆盖** |
| `spike` | **被主动替换为 `ps_spread_*`** | 见下 |
| `anchors[7]` | **没有** | 缺口（但见下） |
| `trills[21]` | **没有** | 缺口（但见下） |
| scratch / LN | panchira 丢弃；Phase 3 保留 `ps_scratch_pos_frac` / `ps_scratch_key_frac` | **Phase 3 更全** |

Phase 3 的三处**已声明**的偏离（`chart_perm_space.py:23-41`，我核对过是有意的、合理的）：

1. **不做单谱 min-max 归一化** —— 正是为了保住跨谱可比性（panchira 的做法对预测有害，已确认）。
2. **`spike` 换成 `spread`** —— `spike` 需要对每个排列排序（O(5040·P log P)），
   且它基本是 `base` 的分位数视图；`spread`（位置本身的方差）更便宜且是不同量。
3. **保留两个 scratch 特征** —— scratch 被锚定在排列群外，但它是机械上不同的输入，不该丢。

**所以：panchira 的"核心"（smooth/tight/base + 5040 枚举）我们已经有，而且更好。
不需要重写 5040 排列。** Phase 3 用的是 NumPy 分块向量化（`_CHUNK`），不是逐排列 Python 循环。

**实测成本**：`chart_perm_stats()` 平均 **0.09 秒/谱**（中位 0.08，最大 0.40，60 张谱实测）。
现有 `chart_perm_space.parquet` 只有 **4,262 行**（又是 Phase 3 范围）。
补齐到 12,194 张 ≈ **12 分钟单进程**。

### B.3 接入 Phase 4B 的方式

- **作为特征输入**：是。22 列 `ps_*` 作特征族 `panchira/perm`。
- **作为外部基线**：可以。单独用 `ps_*` 跑一次冷谱面，看手部几何单独有多少信息。
- **作为解释/可视化**：可以（`base_pct` 回答"这段谱的写法在它的 5040 种排列里算不算变态"）。

### B.4 风险和边界

1. **许可证**：无 LICENSE → **禁止搬运 Rust 源码**。只允许"按公式用 Python 实现"（Phase 3 已遵守）。
2. **严禁照搬单谱归一化**——会摧毁跨谱可比性。
3. **`anchors[7]` / `trills[21]` 的取舍**：
   - `anchors[7]`：每键占用。但我们的 26 客观统计里**已经有 `c_lane1..c_lane7` + `c_lane0_scratch`**（每 lane 音符数）。
     `anchors` 是"加权按键次数"，和 lane 计数高度共线，**边际信息可能很低**。
   - `trills[21]`：21 个 lane 对的交替次数。这个**确实是新信息**（lane 对交互），
     但 21 个特征里有大量稀疏组合，且会与 `v2_*`、`jack_count` 部分重叠。
   - **判断：不急着移植**。先做一次"边际信息"检查（见计划 §4），只有当 perm 特征族
     明显有信号、且 `trills` 被证明不与现有特征共线时才补。
4. **`spike` 不要为了"和 panchira 一致"而加回来**：它的成本（每排列排序）与收益（≈base 的分位数）不成比例。
5. **位置序号假设**：Phase 3 把 lane 1..7 当 1-D 空间坐标（4 为中心），scratch 锚定在外。
   这是**物理布局假设**，对 7K+scratch 的 BMS 成立，但**不适用 DP**。DP 谱必须在 `mode_kind` 过滤掉（已经是）。

---

## C. Framework paper — `docs/reference/framework paper.pdf`

### C.1 它是什么

45 页技术报告，提出 **BMS Chart Character Framework**：

- **七轴并行归属模型**：`chord / stream / scratch / soft / ln / stair / distraction`。
  一个事件可以同时计入多个轴，**各轴之和不约束为 1**（"parallel ownership"，相对的是旧的"互斥划分"模型）。
- **felt-time 1 秒分桶**：对抗 BPM trick 造成的 NPS 虚高（论文例子 8,000 NPS → 449 NPS）。
- **双阈值方案**：家族相对百分位 `p33/p67` 与绝对地板 `0.03/0.08` 取 max，防止稀疏轴误报红。
- 校准语料：**8,555 张 BMS 谱（SP 6,703 / DP 1,852）**；阈值为 SP/DP 分别标定。
- 它还有一层是"标签 + 雷达图 + 密度条"的**可视化 UI**（三个 layer）。

**重要**：论文自己强调的是 **"character（是哪一种难）"而不是 difficulty**，
而且它 §2.5.5 明确说**用等级单调性去验证一个 character 轴是错的**。
所以它**本来就不是为"预测成绩"设计的**，把它当特征需要重新论证。

### C.2 精确可复用路径

- 论文本体：`docs/reference/framework paper.pdf`（45 页，可用 pypdf 抽文字）。
- **本项目里没有它的任何实现。** 我检查过 `bms_ml/phase3/` 的特征注册表：
  `OBJECTIVE_STAT_COLS`(27) / `OBJECTIVE_V2_COLS`(14) / `OBJECTIVE_PERM_COLS`(22) 里
  **没有任何一列是论文七轴的实现**。`v2_*` 有 IOI/NPS/同时押/熵/手平衡，
  是"客观统计 v2"，**不是**论文的 chord_stair_distraction 公式。
- 论文提到的实现文件（如 `_detect_stair_chains_v3`、`scripts/_distraction_overlap_probe_full.py`）
  **不在本仓库**，只在论文正文里被引用。

### C.3 实现成本估算（只估，不做）

我抽读了 chord / stair / distraction 三轴的公式正文。**每个轴都不是一个公式，而是一条流水线**：

| 轴 | 论文里的核心机制 | 粗估 |
|---|---|---|
| `chord` | 16.67ms 窗口内 ≥3 lane 判定 + 按 size class 的**负担加权 Shannon 熵** + "单形状负载比例"修正（v2.1） | 1 天 |
| `stream` | 密度/连续性流水线 + 与 BPM-trick 的交互 | 0.5–1 天 |
| `scratch` | 3 个隐藏子指标 + 长划（LS）子指标 + SP 专用策略 | 1 天 |
| `soft` | 累积 log² 负担 + 基 BPM 判定 + felt frame + 与 soflan/visual_gimmick 标签的责任划分 | 1–1.5 天 |
| `ln` | LN 保持长度、hold floor（1 frame ≈ 16.67ms）等 | 0.5 天 |
| `stair` | v3 链检测：±1 邻接 + **24 tick 回溯匹配** + 重设计史（v2→v3 修了一个 ρ=−0.495 的反向 bug） | 1 天 |
| `distraction` | 250ms 分桶 + 4 步流水线 + 三个子模式（transition / impossible / adjacent）+ 乘积形式 + 按 chart_seconds 归一 | 1–1.5 天 |
| 校准 | 阈值要在**我们的语料**上重新标定（p33/p67 与地板是拟合出来的，不是常数） | 0.5–1 天 |

**合计 ≈ 6–8 个工作日**，而且每一条都需要对照论文的验证章节自查（否则实现出来的可能是"看起来像"但方向错的轴）。

### C.4 接入 Phase 4B 的方式（按用户指令，只能是这三种）

- **外部基线**：作为"一个 BMS 社区框架能预测到什么程度"的对照。
- **事后解释**：模型学到的东西能不能用七轴语言描述。
- **候选特征**：与数据驱动表示做对比。

**不能**当作最终 BMS representation。

### C.5 风险和边界（为什么我不建议做）

1. **红线直接冲突**：七轴是**人工定义的技能轴**。用户明确禁止"不手工定义速度/耐力/LN 等人工技能轴"，
   并要求参考项目"只能作为基线、候选输入或解释"。
   七轴作为**候选输入**在字面上合规，但它正是协议最警惕的那类东西。
2. **felt-time 在 Phase 3 已测过且不适用**（PHASE4_READINESS §1：样本空间内 NPS 零膨胀）。
   **不要重复踩坑。** 论文的 felt-time 主张对本样本无效。
3. **阈值是校准出来的，不是定义**：p33/p67 与 0.03/0.08 地板来自它自己的 8,555 谱。
   用到我们语料上必须重标定，否则会系统性偏。重标定又要一批带真值的谱——而我们的真值只有成绩。
4. **成本/收益倒挂**：6–8 天工作量，换来的是一组人工轴。
   而 Phase 4B 现有基线（26 统计 GBDT）已经到 5.81，MSD 扩覆盖只要 5 秒。
   如果 MSD+统计涨不动了，那才说明"需要新的信息轴"，但那时更该做的是**从 note sequence 学**（协议 §11.5 第 2 层），
   而不是手写七条公式。
5. **它 §2.5.5 自己的警告**：character 轴不该用等级单调性验证。我们没有 character 的真值，
   所以**无法在本项目里验证七轴实现是否正确**——这是最根本的问题。做出来也无法证明做对了。

**结论：本轮不做 framework 七轴。** 保留为"如果 GBDT 基线饱和了再考虑的候选"，并在报告里登记这个决定。

---

## D. 阶段 3 已有实现的复用审计（总表）

| 资产 | 行/列 | 覆盖 Phase 4B 需求 | 复用动作 |
|---|---|---|---|
| `output/phase3/dataset/msd.parquet` | 5,497 × 8 | 12,292 中只有 5,254 | **扩到 12,194**（换输入清单重跑，≈秒级） |
| `output/phase3/dataset/chart_perm_space.parquet` | 4,262 × 23 | 4,262 / 12,194 | **扩到 12,194**（≈12 分钟单进程） |
| `output/corpus/manifest.jsonl` 的 `features[26]` | 36,974 谱 | 已在 cross_section 里作 `cs_*` | 直接用，**已有告警：`features_schema.json` 27 名 vs 26 维，禁止按它索引** |
| `phase3/chart_stats_v2.py` | 14 列 `v2_*` | 4,262 行 | **未进 cross_section**；是第 4 个候选特征族（便宜：一次解析） |
| `phase3/msd_prep.py` / `msd_compute.mjs` / `msd_finalize.py` | 管线 | — | **原样复用**，只改输入清单 + 加上限检查 |
| `phase3/chart_perm_space.py` | 22 列 | — | **原样复用**，只改输入清单 |
| `phase3/response_*` / `history_*` | 玩家响应块 | 不适用 | Phase 4B 是冷谱面，玩家侧无历史可用；**不要碰** |
| `phase4b/run_phase4b.py` | 两道门 | — | 本轮扩展的落点 |

---

## E. 一句话决策清单

| 参考项目 | 复用哪部分 | 不做什么 |
|---|---|---|
| MSD | `msd_prep/compute/finalize` 三段原样，换输入清单，扩到 12,194 | 不改 `ref_repo` 的 WASM；不填 0/均值；不当最终表示 |
| Permikon | **不搬运**（无许可证）；核心已被 Python 实现覆盖 | 不重写 5040 排列；不照搬单谱归一化；暂不移植 anchors/trills |
| Framework | **只引用**，作为候选基线与解释语言 | 不实现七轴（6–8 天 + 红线 + 无法验证）；不重测 felt-time |
