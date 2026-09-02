# BMS AI 难度评价 / 玩家建模研究：Session Handoff

## 0. 文档目的

本 session 的目的不是立即实现模型，而是围绕一个潜在的研究项目进行头脑风暴，并逐步把最初模糊的想法收敛成一个可研究、可验证、可继续推进的方向。

用户目前没有机器学习训练经验，对相关技术只有基础概念，因此后续讨论需要优先从“研究问题、数据、实验”出发，再决定机器学习技术，而不是直接进入模型架构。

当前阶段已经决定：

> **收敛研究对象，但暂不冻结具体技术方案。**

下一阶段最重要的工作是把 replay 数据链路验证清楚，而不是立即训练 AI。

---

# 1. 最初想法

用户最初的想法是：

> 通过训练 AI 来识别 BMS 的难度，最终输出几个维度的数字，而不是单一的难度等级。

很快发现，这个目标比“训练一个难度预测器”更有价值。

用户真正想做的是一套评价体系以及围绕它的一些框架，类似 AlphaOSU 一类项目可以在此基础上实现：

* 给任意 BMS / OBJ 谱面进行多维评价；
* 建立玩家实力模型；
* 根据玩家能力推荐合适的谱面；
* 推荐训练谱面；
* 未来甚至根据玩家能力自动生成训练用 BMS。

因此，“难度”不是最终产品本身，而是一个底层模型。

目前最接近的总架构概念是：

```text
Chart
  ↓
Chart Representation
  ↓
Skill Demand / 谱面技能需求
  ↓
Performance Model
  ↑
Player Skill Profile / 玩家能力画像
```

上层应用：

```text
Skill Demand + Player Skill
        ↓
Difficulty / Performance prediction
        ↓
推荐 / 训练 / 玩家分析 / 后续生成
```

---

# 2. 用户对现有体系的不满

用户认为现有 BMS 难度评价存在两个主要问题：

### 2.1 只有一个难度表等级

例如某张谱面属于某个 ★ / sl / st 等级。

这种表示非常方便，但信息被压缩得非常厉害：

```text
A：高速 stream 为主
B：复杂 rhythm 为主
C：jack 为主
D：scratch / coordination 为主
```

可能被压成差不多的一个等级。

### 2.2 依赖 IR / 玩家成绩推定

这种方法可以比较好地估计“大家实际能不能过”，但仍然更接近：

> 这张谱面有多难被清掉？

而不是：

> 这张谱面要求玩家具有什么能力？

用户希望建立比单纯 clear difficulty 更丰富的描述。

---

# 3. 重要参考资料

## 3.1 BMS Chart Character Framework / Note Attributes

资料：

`horieyuuka.github.io/Note-attributes/Framework-paper`

当前阅读到的版本标题：

**BMS Chart Character Framework — Seven-Axis Radar, Tags, and Felt-Time Normalization**

核心定位：

> 描述“这张谱面是什么类型的难”，而不是把谱面压缩成一个一维难度分数。

它目前定义了七个主要 axis：

```text
chord
stream
scratch
soft
ln
stair
distraction
```

并使用 tags 和 density 等补充信息。

该 framework 的校准 / 验证语料为 8,555 张 BMS charts。它明确区分 chart character 和 IRT clear difficulty，并指出两者在 chart-by-chart 层面可以有很弱的相关性。

该项目对本研究的重要意义：

> 它可以作为“人类手工设计的多维 chart representation” baseline。

不要把它当成最终真理。

当前研究设想更倾向于：

```text
Note Attributes
    ↓
人工知识 / baseline / weak supervision
    ↓
研究 AI 是否能够学习更丰富的表示
```

而不是直接复刻它。

---

## 3.2 Permikon

资料：

`github.com/HANHITSI/Permikon`

项目定位：

> BMS/BME/BMSON 谱面 permutation 分析工具。

它可以针对 7-key 谱面分析全部 5040 种 permutation，并支持可配置评分权重；同时可以结合 difficulty table、beatoraja / LR2 数据库等。

它对当前研究的重要意义：

> 说明“键位排列本身”可以成为谱面表示的重要变量。

尤其值得注意的是：

```text
同一原始谱面
+
不同 permutation
→
玩家表现不同
```

这未来可能形成一种非常干净的实验。

例如：

```text
Chart X
Permutation A → Player performance 95%
Permutation B → 89%
Permutation C → 78%
```

这比仅仅比较两张完全不同的谱面，更容易研究“排列 / pattern 本身如何影响难度”。

---

## 3.3 AlphaOSU

资料：

`github.com/AlphaOSU/AlphaOSU`

项目定位：

> 使用机器学习帮助 osu! 玩家 farm PP。

其 README 中明确把系统拆成：

```text
data fetching
score model
pass model
inference / recommender
```

因此它对当前研究的意义主要不是“照搬算法”，而是说明：

> 玩家模型、谱面表现预测、推荐系统可以被组织成一个完整 pipeline，而不是单独训练一个“难度分类器”。

---

## 3.4 Mug-Diffusion / 类似生成方向

目前仅作为远期参考。

思路：

如果未来拥有：

```text
Chart → Skill Demand
```

那么可以尝试反过来：

```text
Target Skill Demand
        ↓
Generate Chart
        ↓
Difficulty / Skill evaluator
        ↓
验证生成结果
```

甚至最终：

```text
Player Profile
      ↓
能力缺口
      ↓
目标技能
      ↓
生成训练谱面
      ↓
玩家实际游玩
      ↓
Replay
      ↓
更新 Player Profile
```

这可能形成完整的 adaptive training system。

但目前明确：

> **生成不是当前项目目标。**
>
> 当前只需要保证未来 architecture 不妨碍这一方向。

---

# 4. 三个非常关键的认识

## 4.1 玩家看到的不是 BMS 文件

用户首先提出：

> 玩家是通过视觉来理解谱面的。

不能简单把：

```text
#10201:CLCM
#10201:HMHN
```

看作玩家接收到的输入。

更合理的过程是：

```text
BMS data
   ↓
Game rendering
   ↓
Visual information
   ↓
Player perception / pattern recognition
   ↓
Motor planning
   ↓
Physical execution
   ↓
Judge / performance
```

因此未来可能需要同时考虑：

```text
structured chart representation
+
rendered visual representation
```

而不是现在就决定哪一种一定正确。

潜在实验：

```text
Model A: chart structure only
Model B: visual input only
Model C: structure + visual
```

如果 C 显著优于 A，可以支持：

> 玩家视角的视觉表示确实提供了额外信息。

如果 A 已经足够，则反过来说明：

> 原始谱面结构已经包含了主要可预测信息。

两种结果都有研究价值。

---

## 4.2 人体限制意味着 difficulty 可能是非线性的

用户提出：

> 有些型本身很简单，但是只要加速就完全打不了。

因此：

```text
pattern complexity
≠
human difficulty
```

更可能类似：

```text
Difficulty
=
f(pattern,
  temporal density,
  player ability,
  physical constraints)
```

并且可能存在明显的非线性 / threshold：

```text
低速     → 非常容易
中速     → 仍可处理
更高速   → timing variance 增加
临界点   → miss 激增
```

这意味着未来模型可能不应只输出静态的：

```text
pattern demand = 0.8
```

而可能逐渐学习：

> 某种 pattern 随速度变化时，对玩家表现产生怎样的压力曲线。

这属于研究假设，而非当前已证实结论。

---

## 4.3 人类口述的维度不是 ground truth

用户明确提出：

> 自己能说出的 speed / jack / stream / rhythm / reading 等维度，本身可能只是个人理解。

而且不同玩家可能：

* 用不同概念描述同一种 pattern；
* 对同一个 pattern 使用不同术语；
* 根本意识不到自己是如何处理谱面的；
* 实际的视觉 / 认知过程和自我描述不同。

因此：

```text
Human concept
≠
Objective truth
```

更合理的使用方式是：

```text
Human-defined dimensions
→ reference / annotation / baseline / hypothesis
```

而不是：

```text
Human-defined dimensions
→ ground truth
```

未来允许模型形成自己的 latent representation，但需要进一步解释、验证其稳定性。

---

# 5. 关于“读谱过程”的额外研究方向

用户提出了一个非常有意思的认知模型：

```text
Visual input
  ↓
识别哪些键需要击打
  ↓
翻译为手指运动
  ↓
选择 / 执行手指运动
  ↓
结合音乐与 judge 控制 timing
```

并提出可能有：

```text
vertical reading
horizontal reading
mixed / chunking
```

等不同读谱策略。

这个方向实际上可能已经足够独立成为认知科学 / HCI / rhythm-game cognition 的研究课题。

但本项目目前不应该把它作为前置问题。

关键原因：

> 我们甚至无法确定玩家自己的 introspection 是否准确描述了真正的认知过程。

因此当前采用：

**把玩家视为 black box。**

即：

```text
Chart
 ↓
Human black box
 ↓
Observable behaviour / replay
```

先研究：

> 什么谱面输入稳定地导致什么行为？

而不是先要求我们回答：

> 人脑究竟如何读取谱面？

这可以避免项目在早期陷入一个远比 BMS difficulty 更大的认知科学问题。

---

# 6. Replay 成为研究核心数据

这是本 session 最重要的转折点。

用户已有自己的：

**BmsReplayViewer**

仓库：

`github.com/chuang1213/BmsReplayViewer`

该项目目前可以：

* 读取 BMS / BME / BML；
* 读取 `.brd` / `.lr2rep`；
* 逐帧复盘；
* 显示 replay 按键；
* 使用 LR2 / beatoraja 判定；
* 做逐 note 判定分析；
* 分析 FAST / SLOW；
* 计算 mean / stddev 等时间偏移统计；
* 后续可以导出图片 / 视频。

项目 README 明确写出了 replay 解析的版本解耦、统一 ReplayData 等基础设施。

这意味着用户已经拥有：

> 将 BMS chart 与 player replay 对齐的实际基础设施。

因此 replay 很可能比单纯：

```text
score
lamp
```

更适合作为研究数据。

注意：

**Replay 不是 ground truth。**

它只是：

> 更丰富的玩家行为观测。

它依然包含状态、疲劳、设备、练习、偶然失误等噪声。

---

# 7. 为什么 Replay 比 score / lamp 有价值

Score：

```text
“这次打了 97%”
```

Lamp：

```text
“过 / 没过 / 某个血量状态”
```

这些都是非常强的信息压缩。

Replay 则可以提供：

```text
什么时候开始失误
哪里开始失误
失误是否连续
FAST / SLOW
timing variance
哪些 pattern 导致 failure
错误是局部还是持续的
```

例如：

```text
Measure 12 → 99%
Measure 13 → 98%
Measure 14 → 96%
Measure 15 → 71%
Measure 16 → 73%
Measure 17 → 97%
```

这比整张谱：

```text
score = 91%
```

包含更多结构。

尤其重要的是可以比较不同玩家：

```text
              Player A   Player B   Player C
Measure 15      71%        94%        82%
Measure 16      73%        91%        65%
Measure 17      97%        96%        95%
```

这有可能帮助我们区分：

```text
Chart-specific difficulty
vs.
Player-specific weakness
```

---

# 8. 当前最重要的粒度：Measure

用户提出了一个非常具体而合理的数据组织方式：

> 以 measure（小节）为单位，把整个 BMS 和 replay 分解成一个个小节。

每个 measure 同时包含：

## Chart 部分

```text
该 measure 中的 note 排列
lane
time
chord
LN
等
```

## Replay 部分

例如：

```text
note coverage / hit ratio
FAST 数量
SLOW 数量
FAST ratio
SLOW ratio
MISS
其他 timing statistics
```

形成：

```text
Measure i
├── Chart representation
└── Replay behaviour
```

整个谱面：

```text
Measure 1
Measure 2
Measure 3
...
Measure N
```

这个粒度目前已经基本作为 v0 的数据组织方案。

注意：

> **Measure 是数据组织单位，不应直接假设它等于人的认知单位。**

---

# 9. 不应只保存 measure 的聚合统计

用户提出：

```text
这一小节打上了 x% 的 note
FS 有多少个
比例是多少
```

这是一个良好的起点，但后来讨论确定：

不能只保存 aggregate。

例如：

```text
A:
100 notes
95 hits

B:
100 notes
95 hits
```

可能一个是：

```text
前半完美
后半完全崩
```

另一个是：

```text
整个 measure 均匀随机 miss
```

两者完全不同。

因此最终数据应该保留：

```text
raw replay
+
raw chart events
+
measure boundary
+
measure-level aggregate
+
必要的 temporal distribution
```

而不是只保存：

```text
93% / 12 FAST / 5 SLOW
```

原则：

> **尽量保留原始信息，尽量晚进行不可逆汇总。**

未来如果发现需要：

```text
100ms window
200ms window
跨 measure context
局部 pattern
```

可以从原始数据重新生成。

---

# 10. Measure 之间不能被视为完全独立

比如：

```text
Measure 10 → 11 → 12
```

Measure 12 很难，可能不是因为 12 自己特别复杂，而是玩家经历了：

```text
10：高密度
11：高密度
12：继续高密度
```

已经产生疲劳。

因此未来更合理的形式可能是：

```text
Performance(M12)
=
f(M10, M11, M12, Player state)
```

而不是：

```text
Performance(M12)
=
f(M12)
```

因此 measure-level data 适合作为基本 token，但模型以后可能需要读取一个连续的 measure sequence。

这也允许研究：

* 瞬时负荷；
* 持续负荷；
* 累积疲劳；
* recovery section。

---

# 11. Replay Dataset 的概念

未来一次 gameplay 不应该只是一个 `.brd` 文件。

更合理的是一个研究数据单元：

```text
GameplayRecord
{
    player_id,
    chart_id,
    chart_hash,

    client,
    random / mirror / other option,
    rate,

    chart_data,
    replay_data,

    score,
    lamp,
    judgement statistics,

    timestamp
}
```

之后可以预处理成：

```text
Chart
 ├── Measure 1
 ├── Measure 2
 └── ...

Player A
 ├── Measure 1 → behaviour
 ├── Measure 2 → behaviour
 └── ...

Player B
 └── ...
```

必须特别注意：

> **同一个原始 chart 不一定对应玩家看到的同一个 chart。**

Random / Mirror / R-Random 等可能改变实际排列。

因此未来数据中应该保留：

```text
original chart
+
option / permutation
+
actual effective chart
```

这也是 Permikon 方向与 replay 数据可以形成结合的地方。

---

# 12. 关于 Replay 收集

这是项目目前潜在的真正瓶颈之一。

用户指出：

* beatoraja 有 4 个 replay 槽位；
* LR2 更少；
* replay 文件并不是严格意义上的普通明文，需要一定解析流程；
* 用户已经通过 BmsReplayViewer 处理过解析；
* 因此真正困难的是**持续、自动地收集 replay**。

可能方案：

### A. 魔改客户端

让客户端在每次游玩结束后自动导出 / 上传 replay。

优点：

* 自动；
* 长期运行；
* 可以拿到完整数据。

缺点：

* 工作量大；
* 要处理客户端行为、版本兼容；
* 对不同玩家部署不方便。

### B. 玩家安装 replay collector

由用户及朋友安装一个自动收集器：

```text
游戏
 ↓
产生 replay
 ↓
collector 自动发现
 ↓
归档
 ↓
标准化
```

优点：

* 不需要大改客户端；
* 可以快速建立小型真实数据集。

缺点：

* 依赖参与者；
* 行为标准可能不一致；
* replay 会被覆盖，因此 collector 必须足够及时。

### C. 两者结合

短期：

> 几个朋友 + collector，快速获得真实样本。

长期：

> 客户端级自动采集。

当前不需要立刻解决，但它是下一阶段必须验证的现实问题。

---

# 13. 数据量最重要的不是“总 replay 数”

当前已经形成一个非常重要的数据观念：

> **Chart × Player 的交叉覆盖比单纯 replay 数量更重要。**

理想数据不是：

```text
5000 charts
100000 scores
```

而是：

```text
               Chart
          A    B    C    D
Player 1  ✓    ✓    ✓
Player 2  ✓         ✓    ✓
Player 3       ✓    ✓
Player 4  ✓    ✓
```

因为模型最终需要同时学习：

```text
Player Skill
```

和：

```text
Chart Demand
```

如果数据过于稀疏，很容易学成：

> Player A 永远是高手。

或者：

> Chart X 永远很难。

而不是学习真正的 interaction。

因此后续采集策略必须考虑：

* 同一玩家玩多种类型谱面；
* 多玩家共同覆盖同一谱面；
* 不同玩家能力尽量有交集；
* 不只覆盖热门谱面；
* 尽可能覆盖不同作者、不同难度、不同 pattern。

---

# 14. 当前研究假设

目前最准确的研究命题是：

> **BMS 谱面难度不一定是存在于谱面文件中的单一标量，而可能更接近“谱面对人类玩家施加的任务需求”。通过结合谱面结构、视觉表示与玩家 replay，可以尝试学习谱面技能需求、玩家能力以及二者之间的表现关系。**

这只是 hypothesis，不是结论。

尤其要警惕：

> Replay + ML 可能最后只变成一个更复杂的 IRT。

因此不能满足于：

```text
prediction accuracy 很高
```

还要研究：

> 模型到底学到了什么？

以及：

> 它是否能够泛化到没有见过的谱面、玩家、作者、pattern？

---

# 15. 当前推荐的研究路线

暂时采用：

```text
Phase 0
明确数据结构
        ↓
Phase 1
Replay extraction / normalization
        ↓
Phase 2
建立小型真实 Dataset
        ↓
Phase 3
非常朴素的 baseline
        ↓
Phase 4
Chart representation
        ↓
Phase 5
Player representation
        ↓
Phase 6
Chart × Player performance model
        ↓
Phase 7
Skill demand / latent representation
        ↓
Phase 8
上层 difficulty / recommendation
```

其中：

**Phase 1～2 比训练模型更重要。**

---

# 16. 第一批 baseline 应该非常朴素

不要一开始就：

```text
Vision Transformer
Transformer
LLM
巨型 neural network
```

第一阶段应当首先证明：

> 数据本身存在可预测结构。

例如：

```text
NPS
BPM
density
chord rate
jack rate
LN rate
scratch rate
Note Attributes
Permikon 类分析
```

加上：

```text
player historical performance
```

先做简单模型：

```text
linear / logistic regression
random forest
gradient boosting
简单 MLP
```

看能达到什么水平。

如果一个很简单的模型就能解释很大部分现象，这是重要结果。

如果完全不行，再问：

> 缺失的是 feature，还是模型容量？

这样可以避免“模型很复杂，但不知道自己解决了什么”的问题。

---

# 17. 当前不应该冻结的东西

以下内容目前明确不做最终决定：

```text
最终到底有几个 skill dimension
```

例如 speed / stream / jack / chord / rhythm / reading 等都只是候选概念。

---

```text
latent space 是否存在
```

不能预设一定存在一个漂亮的、低维、可解释的空间。

---

```text
视觉是否必需
```

目前只是强假设，需要实验。

---

```text
IRT 是否作为核心数学框架
```

可能有用，但当前不应绑定。

---

```text
作者风格如何 disentangle
```

目前不确定是否应该分离。

作者风格可能本身就是造成技能需求的一部分。

---

```text
最终 AI architecture
```

完全不冻结。

---

# 18. 当前真正的下一步

目前 session 已经从“无限头脑风暴”进入：

> **收敛但不冻结**

阶段。

下一步不应该继续无边界扩张愿景。

最值得实际推进的是：

## Step 1：彻底检查 Replay Parser

回答：

```text
一个 replay 我们到底能恢复什么？
```

需要逐项确认：

```text
note event
player action
lane
timestamp
judge
FAST/SLOW
MISS
LN
chord
random / mirror
是否能精确 note alignment
```

以及：

> 哪些信息来自 replay，哪些来自 BMS，哪些目前无法可靠恢复？

---

## Step 2：定义 Measure Record v0

例如：

```text
MeasureRecord

chart:
    notes
    lanes
    timing
    ln
    bpm
    option

replay:
    hit_ratio
    miss_count
    fast_count
    slow_count
    fast_ratio
    slow_ratio
    timing_mean
    timing_std
    temporal_error_distribution
```

实际字段还没冻结。

---

## Step 3：做最小数据实验

目标甚至不应该是训练 AI。

可以先拿：

```text
10 charts
×
几个玩家
```

生成：

```text
chart → measure → replay behaviour
```

然后人工 inspect。

真正想回答：

> **这种数据结构看起来是否已经包含我们想要的信息？**

---

## Step 4：再决定 Collector

如果 MeasureRecord 确认有价值，再投入力量做：

```text
自动发现 replay
→
解析
→
标准化
→
measure segmentation
→
数据库
```

届时 BmsReplayViewer 将成为研究数据基础设施的一部分。

---

# 19. 当前项目的核心思想，用一句话概括

不要做：

> **“AI 给 BMS 打一个分。”**

而是尝试做：

> **“从谱面结构和玩家行为中学习 BMS 谱面对人类玩家提出了什么任务，以及不同玩家为什么会在不同地方失败。”**

最终：

```text
Chart
    ↓
Skill Demand
    ↕
Player Skill
    ↓
Performance
```

然后难度表、玩家评价、推荐和训练系统都是这个模型之上的应用。

---

# 20. 给下一位 Agent 的工作原则

1. 不要默认用户提出的 speed / jack / stream / reading 等概念是真实且完备的维度。

2. 不要把 score / lamp 当作最高质量 ground truth；replay 是当前最重要的行为数据。

3. 不要因为用户有 replay parser 就默认数据采集问题已经解决；持续采集、去重、交叉覆盖、metadata 仍然是重大工程问题。

4. 不要立即决定 Transformer / CNN / VLM / IRT 等具体技术。

5. 优先讨论：

   * 数据；
   * supervision；
   * baseline；
   * evaluation；
   * generalization；
   * 可解释性。

6. 尽量避免把“玩家如何读谱”这个认知科学问题强行变成本项目的前置条件。

7. Measure 是当前推荐的数据组织单位，但不是认知单位。

8. 保留 raw chart + raw replay，尽量晚聚合。

9. 特别关注新玩家、新谱面、新作者、新 pattern 的泛化，避免模型仅仅记忆历史 score。

10. 研究目标不是证明“AI 比人类聪明”，而是判断数据驱动模型是否能建立一种比现有单值难度表更有用、更可解释、能够支持玩家建模和推荐的 BMS skill representation。

---

## 当前状态

当前最重要的共识：

```text
研究目标：
多维 BMS Skill Demand + Player Skill Model

核心数据：
Chart + Replay

基本数据单位：
Measure

Replay：
行为监督，而非 ground truth

视觉：
重要候选输入，但尚未验证

人工维度：
baseline / hypothesis，而非真理

模型：
尚未决定

生成：
远期应用

当前阶段：
收敛研究对象，开始验证数据链路
```

下一次 session 最适合从：

> **“具体检查 BmsReplayViewer 能导出哪些 replay / chart 信息，并设计 MeasureRecord v0”**

开始，而不是重新讨论整个研究愿景。
