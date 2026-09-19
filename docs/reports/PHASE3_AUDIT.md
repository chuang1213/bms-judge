# Phase 3 数据与工程审计报告

> 日期：2026-09-03 | 分支：phase3 | 状态：审计完成（含新增玩家 tzh / 南极），待实施 baseline
> 数据源：`玩家资料/`（4 个玩家的 beatoraja 原生存档）| schema 参考：`lampghost/`（只读）+ `beatoraja-master/`（只读）
> 审计脚本：`bms_ml/phase3/audit_scorelogs.py`；数字报告：`bms_ml/output/phase3/audit/scorelog_audit.json`；图：`bms_ml/output/phase3/audit/plots/`

## 0. 一句话结论

**数据足够支撑 Phase 3 第一阶段**：scorelog 的"首打事件"完整且带时间戳，chart ID 可与我们已有的解析语料稳定对齐（sha256 直连 + md5 桥接），sl/st/発狂三张表覆盖了目标首打的相当比例，四玩家两两重叠 1,500+ 可充分支撑 proof-of-concept。但 scorelog 是**"刷新记录日志"而非完整游玩日志**（已从 beatoraja 源码 + 数据双重验证），这个语义必须写进所有后续数据构造代码。

## 1. 数据源

| 玩家 | 目录 | score.db | scorelog.db | scoredatalog.db | 备注 |
|---|---|---|---|---|---|
| chuang | `玩家资料/chuang/player1` | 2,690 行 | 9,110 行 | 2,409 行 | 用户本人 |
| muiclac | `玩家资料/muiclac/player1` | 6,194 行 | 20,064 行 | 4,881 行 | 数据量最大，另有 replay/ 目录 |
| tzh | `玩家资料/tzh/player1` | 3,291 行 | 8,240 行 | 2,710 行 | 2021-07 起，5 年跨度 |
| 南极（nanji） | `玩家资料/南极/player1` | 3,062 行 | 6,606 行 | 2,992 行 | 2025-05 起，16 个月但密度极高 |
| buzhang（局长） | `玩家资料/局长/player2` | 202 行 | 302 行 | 197 行 | **已确认舍弃**（数据量过小） |

`玩家资料/steve/` 为空；`回放收集/批次1/` 只有 8 个 .brd/.lr2rep（无 scorelog）。**v0 使用 chuang / muiclac / tzh / 南极 四人**。

所有库均为 SQLite。schema 与 LampGhost `internal/entity/beatoraja.go` 及 beatoraja `PlayDataAccessor.java` 完全一致。

## 2. 各表语义（已对照 beatoraja 源码逐一验证）

### score.db → `score` 表：每谱面最佳记录

主键实际是 (sha256, lnmode)。字段：判定明细（epg/lpg/egr/lgr/egd/lgd/ebd/lbd/epr/lpr/ems/lms）、notes、combo、minbp、playcount（**真实游玩次数计数器，最可靠的"玩过几次"来源**）、clearcount、trophy、option/seed/random、date（最后一次游玩时间）、state、scorehash、avgjudge。

### scorelog.db → `scorelog` 表：**记录刷新日志（不是完整游玩日志）**

beatoraja `PlayDataAccessor.updateScore()`：只有当本次游玩使 **clear / EX score / minbp / combo 任一项刷新历史最佳** 时才写入一行。因此：

- **每张谱的第一局必定在**（首局必然刷新全部指标；实测 muiclac 全部 6,184 张谱首行 oldscore=0，无一例外）；
- 中间"什么都没刷新"的局被跳过（这解释了 log 行数 ≈ 74% × playcount 总和；推测主要是中途 ESC 退出的局）；
- **`score` 字段 = 该局之后的历史最佳 EX score，不一定是当局实际 EX**（当局没刷 EX 分时等于旧最佳；实测 13,880/13,880 行 oldscore == 组内前序累计最大值，语义零例外）；
- `oldscore/oldminbp/oldclear` = 该局之前的最佳值 → 可以精确重建任意时点的 best 轨迹；
- **首打标签完全可靠**：首行 score 即当局实际 EX、clear 即当局 lamp、minbp 即当局 BP。

### scoredatalog.db → `scoredatalog` 表：每局判定明细（但覆盖不完整）

新版本 beatoraja（~2023 起）每次游玩都写一整行判定明细 + option/seed/random。**但实测覆盖有缺口**：muiclac 逐月 scorelog:datalog 行数比从 18% 到 94% 波动（部分月份 datalog 明显少于 scorelog，而 scorelog 只是子集），且 playcount=5 的谱面 datalog 只有 1 行。结论：**datalog 不能当完整游玩历史用**，只作部分局面的判定级补充（含 BP 构成、random option 等）。缺口原因未查清（版本切换/多机使用），不影响 v0。

### score.db → `player` 表：不使用

字段单位与 updatePlayerData 代码对不上（playcount 累计 2.1M/13.9M，更接近"每天按键数"；playtime 疑似毫秒）。不影响本任务，v0 不用它。

## 3. mode 列的真实含义（重要，易踩坑）

`mode` **不是** 7K/14K 之类的游玩模式，而是选项复合值。两条写入路径：

- **普通游玩**（`MusicResult` → `writeScoreData`）：`mode = containsUndefinedLongNote ? lnmode : 0`，lnmode: 1=LN, 2=CN, 3=HCN。所以 0=普通谱游玩，1/2/3=LN 谱的 LN 化游玩；
- **段位/course 游玩**（`CourseResult` → 另一 `writeScoreData(models[])`）：`mode = (ln?lnmode:0) + option*10 + hispeed*100 + judge*1000 + gauge*10000`，其中 **sha256 = 各谱面 sha256 字符串拼接**（128/192/256 字符），notes = 全 course 总和。

由此，观测到的 mode 分布解码：0（97% 的行，普通谱）、1/2（LN/CN 谱）、100/1000（罕见）、10000/10010/10002/10020（**course 行**——它们的 sha256 长度 256/320、0 张命中我们的语料库、notes 中位数 7k~11k，全部吻合）。

**过滤规则：`len(sha256)==64` 且 `mode<100` = 真实单谱游玩事件。**（429/20064 行 muiclac course 行被此规则排除）

## 4. Chart ID 对齐（问题 #4/#6 的答案）

- beatoraja 的 sha256 = 谱面文件原始字节的 SHA-256。验证：Satellite 表中 1,907 个 md5 能匹配到我们 manifest 的条目，其 sha256 与 manifest 逐字节一致（1,907/1,907）；
- 玩家图谱面与 manifest（48,290 唯一 sha256，含解析特征/note sequence）对齐：**union 7,240 张中 4,960 张（68.5%）有完整解析特征**（chuang 93%、muiclac 65%、buzhang 90%）；
- **発狂表只提供 md5**（无 sha256 字段），经 manifest 的 md5→sha256 桥接 100% 可转（1,035/1,035）；
- 三名玩家的 scorelog 处于同一 sha256 ID 空间（beatoraja 全局哈希），可直接合并；
- ⚠️ 注意 manifest 的 `path` 是旧系统路径（F: 盘），按 F→D、C→E 换算后文件实测存在；rel_path/md5/sha256 与路径无关。

## 5. 四玩家 overlap（问题 #7 的答案）

唯一谱面数（过滤 course 行后）：chuang 2,555 / muiclac 5,986 / tzh 2,784 / nanji 2,915。

| 两两共同 | chuang | muiclac | tzh | nanji |
|---|---|---|---|---|
| chuang | — | 1,559 | 1,502 | 1,604 |
| muiclac | 1,559 | — | 1,543 | 1,608 |
| tzh | 1,502 | 1,543 | — | 1,660 |
| nanji | 1,604 | 1,608 | 1,660 | — |

- 任意三人共同：1,194~1,262；**四人共同 1,054 张**（"同谱不同人"评估的主力桥梁）；四人并集 8,622 张；
- buzhang（197 张）确认舍弃后不参与以上数字。

## 6. 难度表覆盖（sl / st / 2018発狂）

| 表 | 条目 | ID | chuang | muiclac | tzh | nanji |
|---|---|---|---|---|---|---|
| Satellite（sl0-12） | 2,358 | md5+sha256 | 461 | 718 | 682 | 551 |
| Stella（st） | 2,230 | md5+sha256 | 615 | 479 | 404 | 674 |
| 発狂BMS2018（★1-25） | 1,035 | md5（桥接） | **1,035** | 936 | 989 | 967 |

- 発狂表 = `output/tables/insane_data.json`（發狂BMS難易度表现行 body，2018 大改版后的版本，★1-★25+??）。**已确认用现行 body**；
- chuang 首打谱面覆盖了**全部** 1,035 张発狂条目（重度表玩家），muiclac 91%；
- 三表并集覆盖首打目标：chuang 80%、muiclac 27%（玩大量无表谱）、buzhang 99%。**muiclac 的表覆盖低是 selection bias 的主要来源，建模时须注意**。

## 7. "截止 T 预测首打"样本可行性（问题 #5 的答案）

首打事件定义：某 (sha256) 在 scorelog 中的最早一行（过滤 course 行后）。按时间 50%/70%/80% 分位取 T：

| 玩家 | 时间跨度 | T@50% | history | targets（表内） | T@70% targets（表内） | T@80% targets（表内） |
|---|---|---|---|---|---|---|
| chuang | 2022-12→2026-08 | 2024-02-18 | 1,278 | 1,277（1,020） | 767 (598) | 511 (394) |
| muiclac | 2021-11→2026-09 | 2024-07-08 | 2,993 | 2,993（802） | 1,796 (654) | 1,197 (462) |
| tzh | 2021-07→2026-09 | 2025-10-17 | 1,392 | 1,392（1,187） | 835 (714) | 557 (451) |
| nanji | 2025-05→2026-09 | 2025-10-08 | 1,458 | 1,457（1,038） | 875 (674) | 583 (453) |

**训练集估算（50% cutoff，目标限定表内）**：1,020 + 802 + 1,187 + 1,038 ≈ **4,050 个带表等级的首打样本**（历史 7,121 条首打记录可构造玩家侧特征）。多个 cutoff 可做时间外推验证（train T1、test T2>T1）。

## 8. 首打标签分布（决定训练顺序）

| 玩家 | FAILED | EASY/LASSIST/ASSIST | NORMAL | HARD | EXHARD | FC | PERFECT | NO_PLAY | BP p25/p50/p75 |
|---|---|---|---|---|---|---|---|---|---|
| chuang | 815 | 259 | 94 | 702 | 449 | 33 | 11 | 192 | 28/65/154 |
| muiclac | 2315 | 1255 | 517 | 1133 | 596 | 80 | 4 | 86 | 34/83/163 |
| tzh | 1430 | 566 | 142 | 462 | 74 | 3 | 0 | 106 | 74/128/245 |
| nanji | 1392 | 709 | 109 | 395 | 193 | 3 | 1 | 113 | 75/176/1270 |

- 首打 FAILED 率 32%~51%，lamp 各级都有样本，**lamp（有序分类）和 score%（连续，acc=EX×50/notes）都可直接训练**；
- BP（minbp）右偏严重但信息量大（含 0 = 无 miss；nanji p75=1270 提示其常打超长谱/高难谱），可作为第二目标；
- 首打 clear==NO_PLAY 的行（chuang 192 / muiclac 86 / tzh 106 / nanji 113）语义可疑（疑似中止局/practice），v0 排除；
- 建议 **v0 主标签 = 首打 score%（acc）**（连续、样本利用最充分、跨玩家可比），lamp 作有序分类辅助头，BP 备选。

## 9. 异常与坑清单（问题 #9 的答案）

1. **scorelog 是刷新日志**（§2）——任何"把 scorelog 当逐局历史"的代码都是错的；旧能力的重建必须经 oldscore/oldminbp/oldclear 链；
2. **score 字段 = 局后最佳**，非当局值；只有刷新 EX 的行 score=当局实际 EX；
3. **course 行混入**：sha256 长度 128/192/256/320、mode≥10000，两处都要过滤；
4. scoredatalog 覆盖有缺口（§2），不能当完整历史；
5. `score.notes` 是"最后一局时谱面的 notes"，可能与首打时文件不一致（实测 muiclac 1 行 acc=710%）；**计算 acc 一律用 manifest 的 meta.total_notes**（sha256 精确对应文件本体）；
6. `score.minbp` 与判定明细不完全一致（beatoraja 分字段独立保存各指标的最佳值，minbp 来自可能不同的一局）；minbp 语义 = bad+poor+**空 poor（ems+lms）**（实测 78-88% 精确相等，其余为跨局混合）；
7. 首打 clear=NO_PLAY 行语义可疑（§8），v0 排除；
8. random option 只有 scoredatalog 覆盖的局可观测（chuang ~24% 的 datalog 局、buzhang ~65% 使用非 normal option）→ **多数游玩的有效谱面排列未知**（R-RANDOM/S-RANDOM 下玩家看到的排列 ≠ 原谱）。v0 接受此噪声并记录；敏感性检查可排除 datalog 中观测到 random 的谱面；
9. manifest 的 `path` 为旧盘符路径（F:→D:、C:→E 转换后存在），勿直接使用；
10. 玩家覆盖极度不均（muiclac 5,986 vs tzh 2,784 等），所有 per-player 统计必须分开报告；nanji 档案只有 16 个月但密度极高，其"近期统计"特征的时间窗口语义与其他玩家不同，建模时注意。

## 10. v0 预测样本构造规范（建议）

```text
样本 = (player u, cutoff T, target chart c)
  c 满足：u 在 T 前从未游玩过 c（scorelog 无该 sha256 行，course 行除外）
  c 的首打发生在 (T, T_end] 内
特征：
  chart 侧：manifest 解析特征（26 维统计 / note sequence / Phase2A 网格——可替换模块）
            + 难度表等级（sl/st/★ 任一可得则用，one-hot 表来源）
  player 侧：u 截至 T 的历史（scorelog + score.playcount + 表等级），temporal aggregation 待对比
标签（首打行）：
  主：acc = first_play_exscore * 50 / manifest.meta.total_notes（连续）
  辅：lamp（clear 1-9，有序）、BP（minbp，整数）
排除：course 行、首打 clear=0 行、manifest 无该 sha256 的 target（v0）、len(sha256)!=64
split：严格时间切分（train T1 / test T2>T1），绝不随机打散；同表等级不构成泄漏，
       但 chart 特征必须来自谱面文件本身（与标签产生无关）
```

## 11. 最小 baseline 建议

- **A. Chart-only**：表等级（sl/st/★ 编码）+ manifest 26 维统计特征 → 线性/GBDT/小 MLP 预测首打 acc；
- **B. Chart + player 轻量统计**：加玩家截至 T 的滚动统计（近期 acc 均值/斜率、各表等级段的历史首打 acc、lamp 分布、playcount、距上次游玩天数）；
- **C. History encoder + chart representation**：历史事件序列（chart 特征 + 结果 + 相对时间）→ 小 encoder；chart 侧先用 26 维 + 表等级，Phase2A 网格作为 v1 消融。
- **判断标准**：C − B − A 的严格递增即 Phase 3 实质性进展；同时用四人共同谱面（1,054 张，"同谱不同人"）做分层评估，检验模型学到的是交互而非"玩家强/谱面难"两个边际量。

## 12. 已确认的决策（2026-09-03）

1. **発狂表用现行 body**（=2018 大改版后，★1-25）；
2. **v0 舍弃表外谱面**：训练/评估目标限定 sl/st/★ 三表并集内的谱面；
3. **buzhang 舍弃**：v0 使用 chuang / muiclac / tzh / 南极 四人；
4. LampGhost 与 beatoraja-master 只作只读参考，直接解析原始 SQLite；
5. 首打 clear=NO_PLAY 行 v0 排除；random option 噪声 v0 接受并记录。
