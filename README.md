# BMS Player–Chart Performance Prediction

**English** | [中文](README_zh.md)

Given a player's play history, predict how they will perform the first time they play a chart they have never seen before.

BMS is a chart format for 7-key vertical-scrolling rhythm games. Each chart is a file, and after playing it, the player gets a score. This repository documents a personal research project: can we use a player's past play records to predict how they will perform on a chart they are playing for the first time? The project ran in three phases and received voluntarily provided save data from 18 players, totaling 14,011 valid play records.

## Research Question

The community usually describes a chart by its difficulty level: is this chart ★11 or ★12? That number comes from discussions among some players and is the current mainstream rating system.

But the question I care about is different:

> Take two charts at 200 notes per second. One player fails on dense streams, another fails somewhere else. How hard is this chart for *this specific person*?

Traditional question: chart → difficulty.  
This project's question: player + chart → first-play performance.

The chart provides objective structural information (note density, long-note ratio, BPM changes, etc.); the player history provides their own skill level and preferences; what we really want to capture is the interaction between the two—whether the same chart is actually harder for different people.

The "first play" constraint is important: we predict performance the first time they encounter the chart, not after they have practiced it. So the player history can only use records strictly before that first play.

## Main Results

Evaluation uses 14,011 first-play records from 18 players. Data is split by time: each player's test set is the last quarter of new charts on their timeline. This is true extrapolation; random splits would not work.

Prediction error (out of 100), mean absolute error:

| Prediction method | Mean error |
|---|---|
| Guess the global average | 11.5 |
| Use only the player's own history | 7.6 |
| Full model (chart structure + player history + personal response curves) | 5.8 |

A few findings:

- Chart information alone is almost useless (error 11.7, worse than guessing the global average). Difficulty itself carries no personal information.
- Fitting a curve to each player's "score vs. difficulty" relationship on each dimension (personal response curve) gives a clear improvement over using averages alone. For the same one-standard-deviation increase in density, one player loses 1 point, another loses 12.
- "How unusual this chart is relative to this player's own history" is more useful than the predicted value from the curve itself.
- Per-player error is almost exactly that player's performance variance (correlation 0.87). There are no particularly hard-to-predict players, only players who are particularly inconsistent.

The model also predicts lamp and BP, but the "how many points off" error is the most understandable main line.

This model only predicts; it does not judge whether "practice will make you stronger"—there is no longitudinal data in the project.

## Failed Directions and Current Status

Several attempts were tested and rejected, but are worth recording:

- Treating the problem as matrix completion (collaborative filtering): decomposing the sparse "player × chart" score matrix. It did not hold up—it did not consistently beat a baseline that just memorizes averages, and even lost to a model that simply shrinks the averages. At this data scale, there is too little overlap between charts.
- Applying personal response curves to players with very short histories: harmful. The curves need enough history to estimate reliably.
- Several new chart features (hand-movement geometry, threshold-free distribution statistics): informative on their own, but no gain when added to existing features.
- Directly fitting a residual target of "within-player deviation": made everything worse.

Failed results are kept in the reports.

For completely new charts with no play data, the current bottleneck is around 5.9 mean absolute error. This number is hard to push down by tuning or switching models: tuning gains are smaller than random variation, feature coverage is already above 99%, and different model classes have been tried. The most promising next step is more player data, not more modeling.

Outside the main experiments, there is also a small tool: drag in a player's save file, and it gives predicted groups and prediction intervals for unplayed charts.

## What's in the Repository

```
README.md          This file
README_zh.md       Chinese version
PROTOCOL.md        How experiments were run: sample definition, time splits, evaluation metrics, rules
EXPERIMENT_LOG.md  Running log: what changed, what effect it had (newest first)
REPRODUCIBILITY.md Environment setup, data preparation, run commands for each phase, which experiments cannot be fully reproduced
docs/              Phase reports (docs/reports/) and early design documents
bms_ml/            All code: parsing, data pipeline, experiments, tests
```

## Data

The repository does not include any raw data; you need to provide your own, and each has its own constraints:

- Player saves (beatoraja / LR2 score files): contain real accounts and play history, cannot be made public;
- BMS chart corpus (~171 GB): used to compute chart features;
- Difficulty tables: community-maintained data, each with its own usage terms.

The Etterna skill rating module used is a GPL-3.0 third-party component; the repository only keeps the call entry point and does not distribute it. Reference papers in `docs/reference/` and third-party projects in `ref_repo/` are also not distributed with the repository.

As a result, some experiments cannot be fully reproduced without these external data. See `REPRODUCIBILITY.md` for what can and cannot be run.

## AI-Assisted Development

This project made heavy use of AI-assisted programming. I was responsible for the research question, data and experiment definitions, method choices, experiment judgments, debugging direction, and interpretation of results. Most of the actual code was generated or modified by AI, including the data pipeline, experiment scripts, tests, and documentation.

So this repository is best treated as a research artifact / experiment archive—a record of the question, process, and conclusions.

## Where to Start Reading

- For a quick look at conclusions: `docs/reports/`. Phase 3 main results → matrix completion failure report → final archive report.
- For experiment rules: `PROTOCOL.md`.
- For what changed: `EXPERIMENT_LOG.md`.
- To run things: `REPRODUCIBILITY.md`.
- To plug in your own save: the access instructions in `docs/reports/`.

## License

Code is released under MIT, see `LICENSE`; documents and reports are under the same license.

Third-party materials not included (BMS charts, difficulty tables, Etterna/MinaCalc, reference papers) belong to their original authors.