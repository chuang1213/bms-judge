"""BMS parser / timeline 单元测试。

时间轴期望值来源：
- bemusic/bmspec 的可执行规范（Gherkin 场景）；
- bms-js（参考实现）的实际输出，见本会话记录的基准：
    Multiple BPM Changes: 01@1.0/0.6s  02@2.0/1.2167s  03@3.0/1.7375s  04@4.0/2.05s
    Basic Stop:           01@4.0/4.0s  02@6.0/8.0s
    Stop+Object:          01@4.0/2.0s  02@5.0/4.5s  03@6.0/5.0s  04@7.0/5.5s
    Time signature:       01@4/4s  04@5.5/5.5s  02@7/7s  03@11/11s
"""

from __future__ import annotations

import unittest

from bms_ml.parser import parse_bms_text
from bms_ml.timeline import build_timeline


def build(text: str):
    return build_timeline(parse_bms_text(text))


def note_by_wav(uc, wav: int):
    return [n for n in uc.notes if n.wav == wav]


def assert_time(tc, actual, expected, delta=0.002):
    tc.assertAlmostEqual(actual, expected, delta=delta)


class TestBasicParsing(unittest.TestCase):
    def test_sentences_and_comments(self):
        uc = build("#TITLE BY MY SIDE\n#ARTIST flicknote\n#00101:0100010001\nThis is a comment\n")
        self.assertEqual(uc.title, "BY MY SIDE")
        self.assertEqual(uc.artist, "flicknote")
        self.assertEqual(uc.notes, [])  # BGM 通道不产生可玩 note

    def test_header_case_and_duplicate(self):
        uc = build("#TiTlE BEAT MUSIC SEQUENCE\n#tItLe BY*MY*SIDE\n")
        self.assertEqual(uc.title, "BY*MY*SIDE")

    def test_basic_objects(self):
        uc = build("#BPM 60\n#00111:01000002\n#00311:0003\n")
        n1 = note_by_wav(uc, 1)[0]
        n2 = note_by_wav(uc, 2)[0]
        n3 = note_by_wav(uc, 3)[0]
        assert_time(self, n1.beat, 4.0)
        assert_time(self, n1.time_sec, 4.0)
        assert_time(self, n2.beat, 7.0)
        assert_time(self, n3.beat, 14.0)
        self.assertEqual(len(uc.notes), 3)

    def test_overlap_replace(self):
        # 同一 (measure, channel, fraction) 后行覆盖（bms-js 语义）
        uc = build(
            "#BPM 60\n"
            "#00113:11111111\n"
            "#00113:0022332255224400\n"
            "#00113:0066\n"
        )
        self.assertEqual(len(uc.notes), 7)
        # "66"（base36 = 222）在第 3 行覆盖了 fraction 4/8 处的对象
        hit = [n for n in uc.notes if n.lane == 3 and abs(n.beat - 6.0) < 1e-6]
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0].wav, 222)


class TestTiming(unittest.TestCase):
    def test_bpm_change_ch03(self):
        uc = build("#BPM 60\n#00003:0078\n#00111:01\n")
        n = note_by_wav(uc, 1)[0]
        assert_time(self, n.time_sec, 3.0)  # 0x78=120 BPM，beat2 变 120，beat4 → 2+1=3
        self.assertEqual(len(uc.bpm_events), 1)
        assert_time(self, uc.bpm_events[0].bpm, 120.0)

    def test_zero_bpm_header_no_crash(self):
        # 真实 corpus 中出现的畸形头部：#BPM 0 → 不崩溃，回落默认 130 并记录 issue
        uc = build("#BPM 0\n#00111:01\n")
        n = note_by_wav(uc, 1)[0]
        assert_time(self, n.time_sec, 4 * 60 / 130.0)  # beat 4 @130BPM
        self.assertEqual(uc.meta.initial_bpm, 130.0)
        self.assertTrue(any(i.code == "zero_bpm" for i in uc.issues))

    def test_negative_bpm_header_abs(self):
        uc = build("#BPM -120\n#00111:01\n")
        n = note_by_wav(uc, 1)[0]
        assert_time(self, n.time_sec, 2.0)  # |−120| = 120 → beat 4 = 2s
        self.assertTrue(any(i.code == "negative_bpm" for i in uc.issues))

    def test_multiple_bpm_changes(self):
        uc = build("#BPM 100\n#00003:0060C0\n#00011:00010203\n#00111:04\n")
        e1 = note_by_wav(uc, 1)[0]
        e2 = note_by_wav(uc, 2)[0]
        e3 = note_by_wav(uc, 3)[0]
        e4 = note_by_wav(uc, 4)[0]
        assert_time(self, e1.time_sec, 0.6)
        assert_time(self, e2.time_sec, 1.2167)
        assert_time(self, e3.time_sec, 1.7375)
        assert_time(self, e4.time_sec, 2.05)
        bpms = sorted(e.bpm for e in uc.bpm_events)
        self.assertEqual(bpms, [96.0, 192.0])  # 0x60=96, 0xC0=192

    def test_extended_bpm_ch08(self):
        uc = build("#BPM 60\n#BPM01 120\n#00008:0001\n#00111:05\n")
        n = note_by_wav(uc, 5)[0]
        assert_time(self, n.time_sec, 3.0)

    def test_stop_basic(self):
        uc = build("#BPM 60\n#STOP11 96\n#00111:01000200\n#00109:00110000\n")
        n1 = note_by_wav(uc, 1)[0]
        n2 = note_by_wav(uc, 2)[0]
        assert_time(self, n1.time_sec, 4.0)
        assert_time(self, n2.time_sec, 8.0)  # STOP 96 = 2 拍 = 2s @60BPM
        self.assertEqual(len(uc.stop_events), 1)
        assert_time(self, uc.stop_events[0].stop_sec, 2.0)

    def test_stop_non_4_4(self):
        uc = build("#BPM 60\n#STOP11 96\n#00102:0.75\n#00111:010002\n#00109:001100\n")
        assert_time(self, note_by_wav(uc, 1)[0].time_sec, 4.0)
        assert_time(self, note_by_wav(uc, 2)[0].time_sec, 8.0)

    def test_stop_and_bpm_same_beat(self):
        for order in (0, 1):
            lines = ["#BPM 60", "#BPM11 30", "#STOP11 96",
                     "#00111:01000200", "#00108:00110000", "#00109:00110000"]
            if order == 0:  # stop 行在文件里写在 bpm 行后面
                lines[4], lines[5] = lines[5], lines[4]
            uc = build("\n".join(lines))
            assert_time(self, note_by_wav(uc, 2)[0].time_sec, 11.0, delta=0.01)

    def test_stop_same_beat_as_object(self):
        # 关键：与 STOP 同 beat 的音符取"停前"时间
        uc = build("#BPM 120\n#STOP11 192\n#00111:01020304\n#00109:11000000\n")
        assert_time(self, note_by_wav(uc, 1)[0].time_sec, 2.0)
        assert_time(self, note_by_wav(uc, 2)[0].time_sec, 4.5)
        assert_time(self, note_by_wav(uc, 3)[0].time_sec, 5.0)
        assert_time(self, note_by_wav(uc, 4)[0].time_sec, 5.5)

    def test_time_signature(self):
        uc = build("#BPM 60\n#00102:0.750\n#00111:0104\n#00211:02\n#00311:03\n")
        assert_time(self, note_by_wav(uc, 1)[0].time_sec, 4.0)
        assert_time(self, note_by_wav(uc, 4)[0].time_sec, 5.5)
        assert_time(self, note_by_wav(uc, 2)[0].time_sec, 7.0)
        assert_time(self, note_by_wav(uc, 3)[0].time_sec, 11.0)

    def test_measure_length_02(self):
        # 0.5 = 2/4：measure 0 只有 2 拍，measure 1 从 beat 2 开始
        uc = build("#BPM 60\n#00002:0.5\n#00111:01\n")
        n = note_by_wav(uc, 1)[0]
        assert_time(self, n.beat, 2.0)
        assert_time(self, n.time_sec, 2.0)


class TestLongNotes(unittest.TestCase):
    def test_lnobj_single_pair(self):
        # #LNOBJ ZZ：ZZ 为结束标记，与同 lane 最近可见音符配对；不重复计数
        uc = build(
            "#BPM 60\n"
            "#LNOBJ ZZ\n"
            "#00111:00220000\n"          # 22 在 measure1 step1 → beat 5
            "#06411:0000000000zz\n"      # ZZ 在 measure64 step9
        )
        self.assertEqual(len(uc.notes), 1)
        ln = uc.notes[0]
        self.assertEqual(ln.note_type, "ln")
        self.assertEqual(ln.lane, 1)
        assert_time(self, ln.time_sec, 5.0)
        # 6 个 token，zz 在 fraction 5/6 → beat 256 + 3.333 = 259.333
        assert_time(self, ln.end_time_sec, 259.333, delta=0.01)
        assert_time(self, ln.duration_sec, 254.333, delta=0.01)

    def test_lntype1_pair(self):
        uc = build(
            "#BPM 60\n"
            "#LNTYPE 1\n"
            "#00151:00220000\n"   # LN 头（lane1）beat 5
            "#06451:000000000033\n"  # LN 尾 beat 259.6
        )
        self.assertEqual(len(uc.notes), 1)
        ln = uc.notes[0]
        self.assertEqual(ln.note_type, "ln")
        assert_time(self, ln.time_sec, 5.0)
        assert_time(self, ln.end_time_sec, 259.333, delta=0.01)

    def test_ln_unclosed_closes_at_end(self):
        uc = build("#BPM 60\n#00151:22\n")  # 只有头，无尾
        self.assertEqual(len(uc.notes), 1)
        self.assertEqual(uc.notes[0].note_type, "ln")
        assert_time(self, uc.notes[0].end_time_sec, 8.0)  # 谱面末尾（measure 1 结束）
        self.assertTrue(any(i.code == "ln_unclosed" for i in uc.issues))


class TestMappingAndQuarantine(unittest.TestCase):
    def test_bme_7key_mapping(self):
        uc = build(
            "#BPM 60\n"
            "#00116:01\n"   # scratch → lane 0
            "#00111:02\n"   # key1 → lane 1
            "#00118:03\n"   # key6 → lane 6
            "#00119:04\n"   # key7 → lane 7
        )
        lanes = {n.wav: n.lane for n in uc.notes}
        self.assertEqual(lanes, {1: 0, 2: 1, 3: 6, 4: 7})

    def test_dp_quarantine(self):
        uc = build("#PLAYER 3\n#00111:01\n")
        self.assertTrue(any(i.code == "not_single_play" for i in uc.quarantine))

    def test_control_flow_quarantine(self):
        uc = build("#RANDOM 2\n#IF 1\n#00111:01\n#ENDIF\n")
        self.assertTrue(any(i.code == "control_flow" for i in uc.quarantine))

    def test_mgq_ln_quarantine(self):
        uc = build("#LNTYPE 2\n#00111:01\n")
        self.assertTrue(any(i.code == "mgq_ln" for i in uc.quarantine))

    def test_empty_chart(self):
        uc = build("#BPM 60\n#00101:0101\n")
        self.assertEqual(uc.notes, [])

    def test_base62(self):
        uc = build(
            "#BPM 60\n"
            "#BASE 62\n"
            "#00111:aa\n"   # 62 进制：a=36 → 36*62+36 = 2268
        )
        self.assertEqual([n.wav for n in uc.notes], [2268])

    def test_abnormal_channel_ignored(self):
        # channel 17（free zone/pedal）、1A（MGQ 扩展）、D1（地雷）不产生 note
        uc = build("#BPM 60\n#00117:01\n#0011A:02\n#001D1:03\n#00111:04\n")
        self.assertEqual([n.wav for n in uc.notes], [4])


if __name__ == "__main__":
    unittest.main()
