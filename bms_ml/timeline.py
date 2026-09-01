"""从 RawChart 构建统一谱面中间表示（IR）。

事件包含 time_sec / beat / lane / note_type / measure / duration，
并计算 chart-level 元数据（note 数、时长、BPM/STOP 统计、密度、和弦等）。

时间轴规则（依据 bmspec 与 hitkey memo）：
- 小节默认 4/4（长度 1.0），#xxx02 可改长度（小数）；
- 通道 03 BPM 为 16 进制 1-255；通道 08 引用 #BPMxx（实数，可为负）；
- STOP 单位 = 1/192 小节 = 1/48 拍；秒数 = value/48 * 60 / 当前BPM；
- 同一位置 BPM 先于 STOP 生效。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .parser import RawChart, Issue


LANE_OF = {"16": 0, "11": 1, "12": 2, "13": 3, "14": 4, "15": 5, "18": 6, "19": 7}
LN_LANE_OF = {"56": 0, "51": 1, "52": 2, "53": 3, "54": 4, "55": 5, "58": 6, "59": 7}


@dataclass
class NoteEvent:
    time_sec: float
    beat: float
    lane: int                 # 0=scratch, 1-7=keys
    note_type: str            # "normal" / "ln"
    measure: int
    end_time_sec: float = 0.0
    end_beat: float = 0.0
    duration_sec: float = 0.0
    wav: int = 0


@dataclass
class BpmEvent:
    time_sec: float
    beat: float
    bpm: float
    source: str


@dataclass
class StopEvent:
    time_sec: float
    beat: float
    stop_sec: float


@dataclass
class ChartMeta:
    total_notes: int = 0
    normal_notes: int = 0
    ln_count: int = 0
    ln_ratio: float = 0.0
    duration_sec: float = 0.0
    measures: int = 0
    initial_bpm: float = 0.0
    min_bpm: float = 0.0
    max_bpm: float = 0.0
    bpm_change_count: int = 0
    stop_count: int = 0
    stop_total_sec: float = 0.0
    lane_counts: List[int] = field(default_factory=lambda: [0] * 8)
    scratch_count: int = 0
    avg_nps: float = 0.0
    peak_nps_1s: float = 0.0
    peak_measure_nps: float = 0.0
    chord_count: int = 0
    chord_size_hist: List[int] = field(default_factory=lambda: [0] * 8)
    jack_count: int = 0
    negative_bpm: bool = False


@dataclass
class UnifiedChart:
    path: str
    title: str = ""
    artist: str = ""
    player: int = 1
    rank: int = 2
    base: int = 36
    used_channels: set = field(default_factory=set)
    notes: List[NoteEvent] = field(default_factory=list)
    bpm_events: List[BpmEvent] = field(default_factory=list)
    stop_events: List[StopEvent] = field(default_factory=list)
    meta: ChartMeta = field(default_factory=ChartMeta)
    quarantine: List[Issue] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)


def build_timeline(raw: RawChart) -> UnifiedChart:
    uc = UnifiedChart(path=raw.path, player=raw.player, rank=raw.rank, base=raw.base)
    uc.title = raw.headers.get("TITLE", "")
    uc.artist = raw.headers.get("ARTIST", "")
    uc.used_channels = {cl.channel for cl in raw.channels}
    uc.issues.extend(raw.issues)

    # ---------- 隔离检查 ----------
    if raw.control_flow:
        uc.quarantine.append(Issue("control_flow", "含 " + ",".join(sorted(set(raw.control_flow)))))
    if raw.player != 1:
        uc.quarantine.append(Issue("not_single_play", f"#PLAYER {raw.player}"))
    if raw.lntype != 1:
        uc.quarantine.append(Issue("mgq_ln", f"#LNTYPE {raw.lntype}"))

    # ---------- 小节长度 ----------
    measure_lengths: Dict[int, float] = {}
    for cl in raw.channels:
        if cl.channel == "02":
            try:
                measure_lengths[cl.measure] = float(cl.raw_values.strip())
            except ValueError:
                uc.issues.append(Issue("bad_ch02", f"measure {cl.measure}"))
    max_measure = max([cl.measure for cl in raw.channels] + list(measure_lengths.keys()), default=0)
    mlens = {m: 1.0 for m in range(max_measure + 1)}
    for m, v in measure_lengths.items():
        if v > 0:
            mlens[m] = v
    uc.meta.measures = len(mlens)

    # ---------- 通道对象收集 ----------
    # objects: (measure, channel, fraction) -> wav 值；后行覆盖前行（bms-js 语义）
    key_objects: Dict[Tuple[int, str, float], int] = {}
    ln_objects: Dict[Tuple[int, str, float], int] = {}
    bpm_ch03: Dict[Tuple[int, float], int] = {}
    bpm_ch08: Dict[Tuple[int, float], int] = {}
    stop_ch09: Dict[Tuple[int, float], int] = {}

    for cl in raw.channels:
        ch = cl.channel
        s = cl.raw_values.strip()
        if ch == "02":
            continue
        if ch == "03":
            vals = [int(s[i:i+2], 16) for i in range(0, len(s)-1, 2) if len(s[i:i+2]) == 2]
            n = max(len(vals), 1)
            for i, v in enumerate(vals):
                if v != 0:
                    bpm_ch03[(cl.measure, i / n)] = v
            continue
        if ch == "08":
            vals = [idx_of(s[i:i+2], raw.base) for i in range(0, len(s)-1, 2)]
            n = max(len(vals), 1)
            for i, v in enumerate(vals):
                if v and v > 0:
                    bpm_ch08[(cl.measure, i / n)] = v
            continue
        if ch == "09":
            vals = [idx_of(s[i:i+2], raw.base) for i in range(0, len(s)-1, 2)]
            n = max(len(vals), 1)
            for i, v in enumerate(vals):
                if v and v > 0:
                    stop_ch09[(cl.measure, i / n)] = v
            continue
        if ch in LANE_OF or ch in LN_LANE_OF:
            vals = [idx_of(s[i:i+2], raw.base) for i in range(0, len(s)-1, 2)]
            n = max(len(vals), 1)
            target = ln_objects if ch in LN_LANE_OF else key_objects
            for i, v in enumerate(vals):
                if v and v > 0:
                    target[(cl.measure, ch, i / n)] = v

    # ---------- 时间轴（对照 bms-js Timing/Speedcore 参考实现） ----------
    # 规则：
    #   - 每个小节默认 4 拍，#xxx02 修改长度（1.0 = 4/4）；
    #   - ch03 BPM 为 hex（0x00-0xFF）；ch08 引用 #BPMxx（可为负 → 取绝对值并标记）；
    #   - #STOPxx 单位 = 1/48 拍（bms-js: headers.stop / 48）；
    #   - 同一 beat 上 BPM 先于 STOP 生效；
    #   - STOP 是"时间跳变"：该 beat 上音符取停前时间，之后的音符取停后时间。
    try:
        initial_bpm = float(raw.headers.get("BPM", "130"))
    except ValueError:
        initial_bpm = 130.0
    if not initial_bpm or initial_bpm != initial_bpm:  # 0 / NaN
        uc.issues.append(Issue("zero_bpm", f"#BPM {raw.headers.get('BPM')!r}"))
        initial_bpm = 130.0
    elif initial_bpm < 0:
        uc.issues.append(Issue("negative_bpm", f"#BPM {initial_bpm}"))
        initial_bpm = abs(initial_bpm)
    uc.meta.initial_bpm = initial_bpm

    # 小节起点 beat 表
    measure_start_beat = [0.0] * (max_measure + 2)
    acc = 0.0
    for m in range(max_measure + 1):
        measure_start_beat[m] = acc
        acc += mlens[m] * 4.0
    total_beats = acc

    def beat_of(m: int, f: float) -> float:
        return measure_start_beat[m] + f * mlens[m] * 4.0

    # 收集时间动作：bpm / stop（stop 已换算成拍）
    actions = []  # (beat, kind, value, source)
    for (m, f), v in bpm_ch03.items():
        actions.append((beat_of(m, f), "bpm", float(v), "ch03"))
    for (m, f), v in bpm_ch08.items():
        if v in raw.bpm_defs:
            actions.append((beat_of(m, f), "bpm", raw.bpm_defs[v], "ch08"))
    for (m, f), v in stop_ch09.items():
        if v in raw.stop_defs:
            actions.append((beat_of(m, f), "stop", raw.stop_defs[v] / 48.0, "ch09"))
    precedence = {"bpm": 0, "stop": 1}
    actions.sort(key=lambda a: (a[0], precedence[a[1]]))

    # 单遍扫描：关键点 (beat, t_pre, t_post, bpm_after)
    keypoints: List[Tuple[float, float, float, float]] = []
    state_beat = 0.0
    state_sec = 0.0
    state_bpm = initial_bpm
    negative_bpm = False
    bpm_log: List[Tuple[float, float, str]] = []   # (beat, bpm, source)
    stop_log: List[Tuple[float, float]] = []       # (beat, stop_sec)

    i = 0
    while i < len(actions):
        beat = actions[i][0]
        state_sec += (beat - state_beat) * 60.0 / state_bpm
        state_beat = beat
        t_pre = state_sec
        # 该 beat 的所有 BPM（同 beat 按动作顺序，最后一个生效）
        j = i
        while j < len(actions) and abs(actions[j][0] - beat) < 1e-9 and actions[j][1] == "bpm":
            _, _, bpm, src = actions[j]
            if bpm < 0:
                negative_bpm = True
            state_bpm = max(abs(bpm), 1e-9)
            bpm_log.append((beat, state_bpm, src))
            j += 1
        # 该 beat 的所有 STOP（在 BPM 之后；stop 用当前 BPM 计算）
        while j < len(actions) and abs(actions[j][0] - beat) < 1e-9 and actions[j][1] == "stop":
            _, _, stop_beats, _ = actions[j]
            stop_sec = stop_beats * 60.0 / state_bpm
            state_sec += stop_sec
            stop_log.append((beat, stop_sec))
            j += 1
        keypoints.append((beat, t_pre, state_sec, state_bpm))
        i = j
    # 末尾虚拟关键点（总拍处）
    t_end = state_sec + (total_beats - state_beat) * 60.0 / state_bpm
    keypoints.append((total_beats, t_end, t_end, state_bpm))

    def t_of(beat: float) -> float:
        """beat → 秒。stop 时刻返回停前时间；之后按停后时间线性推进。"""
        if beat <= keypoints[0][0] + 1e-9:
            # 第一个时间动作之前：按初始 BPM 直线
            return beat * 60.0 / initial_bpm
        k = 0
        for idx, (bk, _, _, _) in enumerate(keypoints):
            if bk <= beat + 1e-9:
                k = idx
            else:
                break
        bk, t_pre, t_post, bpm = keypoints[k]
        if abs(beat - bk) < 1e-9:
            return t_pre
        if k == len(keypoints) - 1:
            return t_post + (beat - bk) * 60.0 / max(bpm, 1e-9)
        bk1, t1_pre, _, _ = keypoints[k + 1]
        return t_post + (t1_pre - t_post) * (beat - bk) / (bk1 - bk)

    uc.meta.duration_sec = t_of(total_beats)

    # ---------- note 事件（含 LN 配对） ----------
    note_by_measure = defaultdict(list)  # m -> [(fraction, lane, type, wav)]
    for (m, ch, f), v in key_objects.items():
        note_by_measure[m].append((f, LANE_OF[ch], "normal", v))
    for (m, ch, f), v in ln_objects.items():
        note_by_measure[m].append((f, LN_LANE_OF[ch], "ln_raw", v))

    items = []
    for m, entries in note_by_measure.items():
        base_beat = sum(mlens[mm] * 4.0 for mm in range(m))
        for f, lane, ntype, wav in entries:
            items.append((base_beat + f * mlens[m] * 4.0, lane, ntype, wav, m))
    items.sort(key=lambda x: (x[0], x[1]))

    notes: List[NoteEvent] = []
    ln_by_lane: Dict[int, List[NoteEvent]] = defaultdict(list)
    ln_channel_pending: Dict[int, Tuple[int, int, float]] = {}

    for beat, lane, ntype, wav, m in items:
        if ntype == "normal":
            if wav in raw.lnobj:
                # LNOBJ 结束标记：与同 lane 最近的潜在头配对（该头已作为普通音符在 notes 中）
                if ln_by_lane[lane]:
                    head = ln_by_lane[lane].pop()
                    head.note_type = "ln"
                    head.end_beat = beat
                    head.end_time_sec = t_of(beat)
                    head.duration_sec = head.end_time_sec - head.time_sec
                else:
                    uc.issues.append(Issue("lnobj_orphan", f"lane {lane} beat {beat:.2f}"))
            else:
                ev = NoteEvent(time_sec=t_of(beat), beat=beat, lane=lane,
                               note_type="normal", measure=m, wav=wav)
                notes.append(ev)
                ln_by_lane[lane].append(ev)
        else:  # LNTYPE 1 通道，成对
            if lane in ln_channel_pending:
                wav0, m0, b0 = ln_channel_pending.pop(lane)
                head = NoteEvent(time_sec=t_of(b0), beat=b0, lane=lane,
                                 note_type="ln", measure=m0, wav=wav0,
                                 end_beat=beat, end_time_sec=t_of(beat))
                head.duration_sec = head.end_time_sec - head.time_sec
                notes.append(head)
            else:
                ln_channel_pending[lane] = (wav, m, beat)

    # 未闭合的 LNTYPE1 头：按谱面末尾闭合
    for lane, (wav0, m0, b0) in ln_channel_pending.items():
        head = NoteEvent(time_sec=t_of(b0), beat=b0, lane=lane,
                         note_type="ln", measure=m0, wav=wav0,
                         end_beat=total_beats, end_time_sec=t_of(total_beats))
        head.duration_sec = head.end_time_sec - head.time_sec
        notes.append(head)
        uc.issues.append(Issue("ln_unclosed", f"lane {lane}"))

    notes.sort(key=lambda x: (x.time_sec, x.lane))
    uc.notes = notes

    # ---------- 元数据统计 ----------
    meta = uc.meta
    meta.negative_bpm = negative_bpm
    for ev in notes:
        meta.total_notes += 1
        meta.lane_counts[ev.lane] += 1
        if ev.note_type == "ln":
            meta.ln_count += 1
        else:
            meta.normal_notes += 1
    meta.scratch_count = meta.lane_counts[0]
    meta.ln_ratio = meta.ln_count / meta.total_notes if meta.total_notes else 0.0
    bpms = [initial_bpm] + [b for _, b, _ in bpm_log]
    meta.min_bpm = min(bpms)
    meta.max_bpm = max(bpms)
    meta.bpm_change_count = len(bpm_log)
    meta.stop_count = len(stop_log)
    meta.stop_total_sec = sum(s for _, s in stop_log)
    meta.avg_nps = meta.total_notes / meta.duration_sec if meta.duration_sec else 0.0

    if notes:
        ts = [n.time_sec for n in notes]
        j = 0
        peak = 0
        for i in range(len(ts)):
            while j < len(ts) and ts[j] - ts[i] < 1.0:
                j += 1
            peak = max(peak, j - i)
        meta.peak_nps_1s = peak
        per_measure = defaultdict(int)
        for n in notes:
            per_measure[n.measure] += 1
        if per_measure:
            # 用整曲有效 BPM 估算每小节时长（第一版近似）
            eff_bpm = total_beats / max(meta.duration_sec, 1e-9) * 60.0
            meta.peak_measure_nps = max(
                c / max(mlens[m] * 4.0 / eff_bpm * 60.0, 1e-6)
                for m, c in per_measure.items()
            )

    beat_groups = defaultdict(list)
    for n in notes:
        beat_groups[round(n.beat, 3)].append(n)
    for group in beat_groups.values():
        if len(group) >= 2:
            meta.chord_count += 1
            meta.chord_size_hist[min(len(group), 7)] += 1

    by_lane = defaultdict(list)
    for n in notes:
        by_lane[n.lane].append(n.time_sec)
    for lane, ts in by_lane.items():
        for a, b in zip(ts, ts[1:]):
            if b - a < 0.12:
                meta.jack_count += 1

    for beat, bpm, src in bpm_log:
        uc.bpm_events.append(BpmEvent(time_sec=t_of(beat), beat=beat, bpm=bpm, source=src))
    for beat, s in stop_log:
        uc.stop_events.append(StopEvent(time_sec=t_of(beat), beat=beat, stop_sec=s))
    return uc


def idx_of(s: str, base: int) -> int:
    """两位索引字符 → 值（36/62 进制）。非法返回 0。

    36 进制下大小写等价（zz == ZZ）；62 进制下小写 a-z = 36-61（#BASE 62）。
    """
    if len(s) != 2:
        return 0
    v = 0
    for ch in s:
        if "0" <= ch <= "9":
            d = ord(ch) - ord("0")
        elif "A" <= ch <= "Z":
            d = ord(ch) - ord("A") + 10
        elif "a" <= ch <= "z":
            d = ord(ch) - ord("a") + 10
            if base == 62:
                d += 26
        else:
            return 0
        if d >= base:
            return 0
        v = v * base + d
    return v
