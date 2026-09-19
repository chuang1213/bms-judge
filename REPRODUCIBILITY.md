# 复现说明

本文件承载 README 不再展开的内容：环境安装、数据准备、各阶段的运行命令、期望输出，
以及**哪些实验无法完整复现**。

> 先说结论：仓库不含原始数据，所以**已发表的具体数字无法直接复现**。
> 能独立跑通的是 116 个回归测试；其余实验需要你自备玩家存档与 BMS 语料库，
> 跑出来的会是你自己的数字，而不是本文档里的数字。

## 1. 环境

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 回归测试（116 个，约 2 秒）
.venv/Scripts/python.exe -m unittest discover bms_ml/tests
```

- 核心依赖（numpy / pandas / scikit-learn / scipy / pyarrow / matplotlib）即可跑
  Phase 3 / 4 / 4B 的**全部基线与实验**，纯 CPU。
- 只有 Phase 2A 的序列编码器实验与 Phase 3 的 C 阶梯需要 PyTorch（CUDA 轮子，可选，见
  `requirements.txt` 注释）。
- 计算 Etterna 技能评级（MSD 轴）需要 Node.js ≥ 18，见 `bms_ml/phase3/msd_compute.mjs`。

## 2. 数据准备

仓库外的四类材料，按下面的位置放置：

| 材料 | 位置 | 可选项 |
|---|---|---|
| beatoraja 存档 | `PlayerData/beatoraja/<name>/player1/` | `scorelog.db` 必需（首打事件与时间戳都在其中）；`score.db` 在 Phase 4 用于重建「当前最佳成绩」 |
| LR2 存档 | `PlayerData/LunaticRave2/<name>.db` | 只有最佳成绩，无时间戳、无逐局 |
| BMS 谱面语料库 | 任意本地路径，经扫描生成清单 | 用于生成 `bms_ml/output/corpus/manifest.jsonl` 与 `sequences/*.npy` |
| 难度表 | `bms_ml/output/tables/` | 经 `fetch_tables.py` 下载 |

接入一名新玩家（Phase 3 SOP，顺序不能反）：

```powershell
.venv/Scripts/python.exe bms_ml/phase3/ingest_player.py add <name> PlayerData/beatoraja/<name>/player1
# 人工审阅 audit 输出 → 再在 bms_ml/phase3/players.json 里设 include=true
.venv/Scripts/python.exe bms_ml/phase3/data.py
```

语料库相关（按需）：

```powershell
.venv/Scripts/python.exe -m bms_ml.recon_corpus --root "<corpus-root>" --out bms_ml/output/corpus
.venv/Scripts/python.exe -m bms_ml.probe_corpus --charts bms_ml/output/corpus/charts.txt --n 1000
.venv/Scripts/python.exe -m bms_ml.build_corpus_manifest --charts bms_ml/output/corpus/charts.txt `
    --out bms_ml/output/corpus --tables-dir bms_ml/output/tables --workers 8
.venv/Scripts/python.exe -m bms_ml.corpus_report --manifest bms_ml/output/corpus/manifest.jsonl
```

## 3. 各阶段运行命令

Phase 3（首打表现预测）：

```powershell
.venv/Scripts/python.exe bms_ml/phase3/data.py                # 构建样本（需存档 + 语料库清单）
.venv/Scripts/python.exe bms_ml/phase3/compare_nolevel.py     # 固定基线 A / H / B / B_resp / B_full
.venv/Scripts/python.exe bms_ml/phase3/history_response.py    # 个人响应曲线
.venv/Scripts/python.exe bms_ml/phase3/response_dev.py        # 玩家相对偏差
.venv/Scripts/python.exe bms_ml/phase3/response_msd.py        # Etterna 技能评级轴
.venv/Scripts/python.exe bms_ml/phase3/uncertainty_eval.py    # 预测区间与通过概率
.venv/Scripts/python.exe bms_ml/phase3/recommend.py --serve   # 推荐工具（拖入 scorelog.db）
```

Phase 4（矩阵补全，负结果）与 Phase 4B（只预测没人打过的新谱）：

```powershell
.venv/Scripts/python.exe bms_ml/phase4/build_cross_section.py
.venv/Scripts/python.exe bms_ml/phase4/audit_cross_section.py
.venv/Scripts/python.exe bms_ml/phase4/evaluate_matrix.py     # 四固定留出 + 固定基线
.venv/Scripts/python.exe bms_ml/phase4/run_m3.py              # 矩阵补全 + 门禁判定

.venv/Scripts/python.exe bms_ml/phase4b/build_content_features.py
.venv/Scripts/python.exe bms_ml/phase4b/run_phase4b_features.py
.venv/Scripts/python.exe bms_ml/phase4b/run_phase4b_tuning.py
.venv/Scripts/python.exe bms_ml/phase4b/run_phase4b_v2check.py
```

## 4. 期望输出（用于判断「跑对了没有」）

在**原始样本空间**（18 名玩家、14,011 行）上：

```
compare_nolevel.py
  -> {'A': 11.724, 'H': 7.619, 'B': 6.901, 'B_resp': 6.436, 'B_full': 5.801}
  -> n_train / n_test = 7,001 / 7,010

unittest discover bms_ml/tests
  -> Ran 116 tests ... OK
```

换一批玩家时，这些数字会变化，但相对关系（H 明显优于 A、B_full 优于 B）应当成立。

结果 JSON 会落盘到 `bms_ml/output/phase3|phase4/`。每个结果都记录 `features_used`
以及数据配置，不会出现「不知道这份数字是哪套特征跑的」。

## 5. 哪些实验依赖外部语料

需要 `manifest.jsonl` 或 `sequences/*.npy` 的步骤：

- `data.py` 的谱面统计特征（26 维客观统计）；
- `chart_stats_v2.py`（无阈值分布统计）、`chart_perm_space.py`（手部移动几何）；
- `msd_prep.py` → `msd_compute.mjs` → `msd_finalize.py`（Etterna 技能评级，需要 Node）；
- Phase 2A 的全部实验（网格序列表示与自监督预训练）；
- Phase 4B 的特征铺满步骤。

不需要外部语料的步骤：

- `parser.py` / `timeline.py` 的语义（116 个回归测试中的大部分）；
- Phase 4 的横截面构建与基线（只需要玩家成绩矩阵）；
- 评估原语与协议不变量（切分无泄漏、基线只在训练集拟合、ALS 求解器回归）。

## 6. 无法完全复现的部分

- **已发表的具体数字**：需要同一批 18 份玩家存档与同一份语料库快照。存档是私有的，
  语料库也没有随仓库分发。
- **Phase 2A 的表示学习结论**：需要完整语料库序列，且当时用的是特定 GPU 与随机种子。
- **Etterna 技能评级（MSD）**：第三方 WASM（GPL-3.0）未随仓库分发，需要自行获取。
- **跨客户端对比**（LR2 与 beatoraja 的差异测量）：需要同一个人在两个客户端上的存档。

仓库中保留在 `bms_ml/output/corpus/analysis/` 下的小体积结果（JSON 与图表）是
Phase 1–2A 的已发表证据。它们无法在无语料的情况下重算，因此留在版本控制里。

## 7. 路径与盘符

历史报告和部分脚本注释里会出现早期双系统环境的绝对路径，按下表换算即可；
代码本身一律使用相对仓库根的路径。

| 历史路径 | 当前路径 |
|---|---|
| `F:\Projects\bms judge` | 仓库根目录 |
| `F:\games\BMS` | BMS 语料库所在位置 |
| `E:\Users\...\Python311` | `C:\Users\...\Python311` |

若 `.venv` 是跨系统迁移过来的，`pyvenv.cfg` 里的 `home` / `executable` / `command`
可能仍指向旧盘符，改成本机路径即可（环境里的包无需重装）。

## 8. `bms_ml/output/` 的提交策略

- **不提交**（可再生成 / 体积大 / 第三方）：`corpus/sequences/`、`manifest.jsonl`、
  `tables/`、`phase2a|phase3|phase4/` 下的数据集与结果全集、`runs/`、`*.pt`、
  以及 MB 级语料审计报告。规则见 `.gitignore`。
- **提交**：`corpus/analysis/` 下的小体积结果 JSON 与图表，理由见上文第 6 节。
