#!/usr/bin/env python3
"""생성 MIDI의 '연주 불가능' 패턴 진단.

베이스 연주의 물리적 한계 기준:
  - burst: 1초 창 안에 16노트 초과 (최속 16비트 @240bpm = 16/s)
  - retrigger: 같은 음 재타격 간격 < 60ms (최속 트레몰로 한계)
  - jump: 인접 노트 간격 < 100ms인데 19반음(옥타브+5도) 초과 도약
  - overlap: 모노포닉인데 3ms 초과 겹침
  - sliver: 30ms 미만 노트

Usage: python eval/diagnose_playability.py <generated.mid>
"""
import sys

import pretty_midi

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def nm(p):
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def main():
    path = sys.argv[1]
    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted(
        [n for i in pm.instruments if not i.is_drum for n in i.notes],
        key=lambda n: n.start,
    )
    print(f"{len(notes)} notes, {notes[-1].end - notes[0].start:.0f}s span")

    # burst
    bursts = []
    for i in range(len(notes)):
        cnt = sum(1 for n in notes[i:] if n.start < notes[i].start + 1.0)
        if cnt > 16:
            bursts.append((notes[i].start, cnt))
    merged_bursts = []
    for t, c in bursts:
        if merged_bursts and t - merged_bursts[-1][0] < 1.0:
            merged_bursts[-1] = (merged_bursts[-1][0], max(merged_bursts[-1][1], c))
        else:
            merged_bursts.append((t, c))

    retrigs = []
    jumps = []
    overlaps = []
    for a, b in zip(notes, notes[1:]):
        ioi = b.start - a.start
        if a.pitch == b.pitch and ioi < 0.06:
            retrigs.append((a.start, a.pitch, ioi))
        if ioi < 0.10 and abs(b.pitch - a.pitch) > 19:
            jumps.append((a.start, a.pitch, b.pitch, ioi))
        if a.end > b.start + 0.003:
            overlaps.append((a.start, b.start, a.end - b.start))

    slivers = [(n.start, n.pitch, n.end - n.start) for n in notes if n.end - n.start < 0.03]

    print(f"\nburst(>16notes/s) 구간: {len(merged_bursts)}")
    for t, c in merged_bursts[:10]:
        print(f"  {t:7.2f}s: {c} notes/s")
    print(f"retrigger(<60ms 동일음): {len(retrigs)}")
    for t, p, ioi in retrigs[:10]:
        print(f"  {t:7.2f}s {nm(p)} ioi={ioi*1000:.0f}ms")
    print(f"impossible jump(<100ms에 >19st): {len(jumps)}")
    for t, p1, p2, ioi in jumps[:10]:
        print(f"  {t:7.2f}s {nm(p1)}→{nm(p2)} ioi={ioi*1000:.0f}ms")
    print(f"overlap(>3ms): {len(overlaps)}")
    for s1, s2, ov in overlaps[:5]:
        print(f"  {s1:7.2f}s↔{s2:.2f}s {ov*1000:.0f}ms")
    print(f"sliver(<30ms): {len(slivers)}")
    for t, p, d in slivers[:5]:
        print(f"  {t:7.2f}s {nm(p)} {d*1000:.0f}ms")


if __name__ == "__main__":
    main()
