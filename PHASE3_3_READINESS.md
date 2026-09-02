# Phase 3.3 Readiness Report

> 日期：2026-09-03 | 分支：phase3 | 定位：**站稳脚跟**——新玩家数据到来前，把数据入口、
> 实验协议、表征接口固化，不再堆模型。
> 新增：`coverage_audit.py`、`ingest_player.py`、`players.json`、`chart_repr.py`、`PROTOCOL.md`

## 1. 当前已经稳定的部分

- **数据管线**：`raw save → ingest_player.py 校验 → players.json → data.py → samples.parquet`
  全链路可复现（HGB 重跑字节级一致）。样本定义、清洗规则（course 行 / NO_PLAY / ex==0 /
  BP>notes+5）、时间切分（q50/q75 逐玩家）全部固化在 `PROTOCOL.md`；
- **无表依赖**：难度表等级已从特征与样本筛选中完全移除，`h_knn_acc`（26 维客观统计空间
  k 近邻历史均值）替代 `h_level_acc`，交互信号无损恢复（centered R² +0.125 vs +0.150）；
  难度表降级为外部参照；
- **基线格局**（2,856/2,858 样本，全客观特征）：H 主导 acc（10.03，centered R² +0.361
  为全项目最强交互证据），B 主导 lamp（1.614/QWK 0.546）与 BP（180.8）；C 阶梯 3 种子
  纪律已建立（`c_chart_aware.py --seed`）；
- **评价协议**：三目标分开评、centered R² 必报、fixed-chart 子集必报、逐玩家必报、
  单 seed 不作结论、`features_used` 写入每个结果 JSON（chart_repr.py 唯一登记处）。

## 2. 当前真正的瓶颈

1. **玩家规模与能力空间覆盖**（不是模型）：4 玩家 / ~5,700 样本下，learned player state
   没有可验证的生存空间；3.2 已证明 C 的条件化随数据扩大在改善（centered R² -4.2→-0.39）；
2. **ST4 以上区间的 interaction 识别条件几乎为零**（见 §3）；
3. **表外谱面的协同覆盖弱**：3,495 次表外首打中仅 404 张被 ≥2 人打过——muiclac 独有的
   游玩范围无法支撑交互学习；
4. 次要：nanji 的 acc/BP 预测仍是短板（分布外玩家代表）；BP 长尾导致 raw MAE 被超长谱主导。

## 3. Player Coverage Audit——当前玩家数据缺在哪

（SL/ST 仅作坐标：SL=satellite 等级，ST 映射为 12+stella 等级；基于首打事件）

| 坐标区间 | 首打数 | ≥2人共同谱 | 判定 |
|---|---|---|---|
| SL0-2 / SL3-5 / SL6-8 | 504 / 408 / 518 | 129 / 106 / 158 | ok |
| SL9-12 | 841 | 267 | ok |
| ST0-3（≈SL13-15） | 1,552 | **497** | ok（交互识别最强区） |
| ST4-7（≈SL16-19） | 445 | 121 | ok，但 **tzh 仅 23 次** |
| ST8-12（≈SL20-24） | **22** | **4** | **very sparse** |
| ST13+ | **0** | 0 | **空白** |
| 発狂（无SLST坐标） | 3,647 | 965 | ok |
| 表外 | 3,495 | 404 | 协同覆盖弱 |

- 四人共同首打谱面 **917** 张、≥3 人 1,623、≥2 人 2,651（unique 6,067）；
- 玩家画像：chuang=発狂+ST0-3 重度；muiclac=量最大+表外 1,884；tzh=SL9-12 重度（299）
  但 ST4-7 空洞；nanji=分布均衡、ST0-3 最多（476）；
- **有效 interaction 识别条件**目前集中在：発狂全域、ST0-3、SL6-12；ST8 以上与表外
  基本不具备识别条件；
- **新增 4-8 个玩家最优先补**：① 打 ST4-12 的中高段位玩家（当前最空白且是 lamp/BP
  交互最有价值区）；② 曲库与现有四人高度重叠的玩家（把表外首打变成共同首打）；
  ③ SL0-5 低段玩家（绝对量最薄、最容易招募、对推荐场景最重要）；④ 若目标是检验
  learned representation，优先招"与其他玩家大量同谱"的玩家，而非独立圈子的重度玩家。

## 4. 未来 8-12 个玩家规模下最值得做的实验

1. **C vs H/B 的规模响应曲线**（核心）：固定协议，按玩家数 4→6→8→12 重复 C0/C1/C2，
   画 centered R² 与 lamp QWK 随样本量的响应——直接回答"learned player state 何时开始
   有优势"，一次实验定生死；
2. **跨玩家 cold-start 评估**：留一玩家全外（leave-one-player-out），检验 player state
   对全新玩家的泛化——这是"新玩家进来"的真实场景；
3. **玩家相似度/能力空间结构**：H 特征空间 + 共同谱面表现做玩家聚类/嵌入，回答
   "8-12 人是否已覆盖有意义的能力空间"（不做人工 skill 维度命名）；
4. **客户端/来源对照**：beatoraja vs LR2、不同客户端版本的数据质量差异（provenance
   字段已预留）；
5. （需新数据配合）首打时间局部性分析：新玩家加入时间是否带来自然 cohort 结构。

## 5. 新数据到来前必须解决的技术债务

已在本阶段解决：
- ✅ 统一入口（`ingest_player.py`：校验→provenance→include=false 待审）；
- ✅ timestamp 容错（`time: synthetic` 模式：按游玩序号生成序数日，排序保留、绝对时间
  语义显式失效，报告需声明）；LR2 解析器本身仍是 stub（prompt 要求：不提前实现）；
- ✅ 特征清单固化（`chart_repr.py` 唯一登记处 + 结果 JSON 的 `features_used`）；
- ✅ 协议固化 + 复现性验证（`PROTOCOL.md`，HGB 重跑字节级一致）；
- ✅ 旧表依赖脚本标记 deprecated（baseline31.py / c_model.py）。

仍欠（新数据到来前做，均为小改动）：
- [ ] LR2 scorelog 解析器（等第一份真实 LR2 数据到位再写，避免对着文档空想）；
- [ ] `h_knn_acc` 的 k 与距离度量目前未调参（k=20 拍脑袋）——新数据到来后做一次 k∈
      {10,20,50} 的敏感性检查即可，不要现在做；
- [ ] 多玩家加入后 `data.py` 的 cutoffs 会随整体分位数移动——需要决定是否固定
      "历史 cutoff 注册表"以保证跨实验可比（建议：players.json 里固化每玩家 cutoff 时间戳）；
- [ ] 玩家资料/ 目录 549MB 含 423MB replay——**建议用户另行备份**（.gitignore 政策不
      提交原始数据，这些是唯一副本）。

## 6. 现在不要做的事

1. **不做推荐系统/训练价值模型**：接口链（history→state→chart repr→acc/lamp/BP→目标）
   已可衔接，但"预测表现好 ≠ 值得练"，training value 必须等 longitudinal/intervention
   数据；目前连 prefix 都不要写；
2. **不堆更大的 C 模型/调参**：瓶颈是数据不是容量；等规模响应曲线（§4.1）说话；
3. **不重新引入难度表**（任何形式：特征、筛选、加权），除非未来验证研究明确反向结论；
4. **不定义 skill taxonomy、不命名能力维度**：等 C 类模型有可解释的胜利再谈归因；
5. **不放宽清洗规则换样本量**（NO_PLAY/ex==0/BP 校验都是真实数据教训换来的）；
6. **不为 LR2 预写复杂架构**：`time: synthetic` 已保证数据结构不被 timestamp 卡死，
   解析器等真实样本到位再补。

## 7. 新玩家接入 SOP（给未来的自己）

```text
1. 收到存档 → 放入 玩家资料/<name>/
2. python bms_ml/phase3/ingest_player.py add <name> 玩家资料/<name> [--time synthetic?]
3. 检查 audit 输出（行数/时间跨度/mode 分布），在 players.json 设 include=true
4. python bms_ml/phase3/data.py            # 重建数据集
5. python bms_ml/phase3/embed_charts.py    # 增量补算新增谱面的 Phase2A 表征
6. python bms_ml/phase3/compare_nolevel.py # H/B 重跑
7. python bms_ml/phase3/c_chart_aware.py --variants C0,C1,C2 --seed 0,1,2
8. python bms_ml/phase3/coverage_audit.py  # 更新覆盖图
```
