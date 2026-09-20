"""드럼 정답 MIDI 시간 정렬 도구.

**이 프로젝트에서 반복적으로 사고를 낸 지점** — bass_sample3(50s 오차), bass_sample4/5,
drums_sample4(505ms 오차) 전부 "정답 첫 노트 시각을 빼면 된다"는 가정이 틀렸다.
정답 MIDI는 원곡 타임라인 기준으로 익스포트되는 경우가 많아, 반드시 오디오 온셋과
cross-correlation으로 실제 오프셋을 찾아야 한다.

방법: 후보 오프셋마다 "정답 온셋 위치의 오디오 onset strength 평균"을 재서 최대화.
(임펄스 열 상관과 동등하되 프레임 양자화에 강하도록 ±1프레임 max를 취함.)

사용법:
    python eval/align_drum_gt.py <src.wav> <src.mid> <out.mid> [--lo -5 --hi 60]
"""

import argparse
from pathlib import Path

import numpy as np
import librosa
import pretty_midi

SR = 44100
HOP = 256


def onset_env(y, sr):
    oe = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    return (oe - oe.mean()) / (oe.std() + 1e-9)


def score_offset(a, gt, off, sr, tol_fr=1):
    se = sr / HOP
    n = len(a)
    idx = ((gt - off) * se).round().astype(int)
    vals = []
    for i in idx:
        lo, hi = max(0, i - tol_fr), min(n, i + tol_fr + 1)
        if lo < hi:
            vals.append(a[lo:hi].max())
    return float(np.mean(vals)) if vals else -1e9


def hit_frac(a, gt, off, sr, tol_fr=2, thresh=1.0):
    """정답 온셋 중 실제 오디오 온셋 피크에 안착한 비율.

    ⚠ 정렬 목적함수로 mean score보다 **이쪽이 신뢰도가 높다** (drums_sample5 실측):
    조밀한 주기 패턴 곡에서 3초(1마디) 배수 alias가 mean score는 거의 같게 나오지만
    (4.63 vs 4.58), 안착률은 정답을 뚜렷이 구분한다(98.3% vs 97.4% vs 79.7%).
    mean은 소수의 강한 온셋에 끌려가고, 안착률은 "몇 개나 맞았나"를 직접 센다.
    """
    se = sr / HOP
    n = len(a)
    idx = ((gt - off) * se).round().astype(int)
    hits = 0
    for i in idx:
        lo, hi = max(0, i - tol_fr), min(n, i + tol_fr + 1)
        if lo < hi and a[lo:hi].max() > thresh:
            hits += 1
    return hits / len(gt)


def _valid(gt, off, audio_dur, max_drop_frac=0.01):
    """정답이 오디오 구간 밖으로 밀려나면 그 오프셋은 무효.

    ⚠ drums_sample5 실측 교훈: shaker/hihat이 조밀한 규칙적 그리드인 곡은 **주기적
    자기유사성 때문에 가짜 최적점**이 생긴다(offset 57.99s가 score 4.82로 정답
    -0.015s의 4.58보다 높게 나왔지만, 실제론 551노트가 잘리고 안착률 79.7%).
    score만 믿으면 안 되고, "정답이 오디오 안에 들어오는가"를 하드 제약으로 걸어야 한다.
    """
    shifted = gt - off
    dropped = int((shifted < 0).sum()) + int((shifted > audio_dur).sum())
    return dropped <= max_drop_frac * len(gt)


def find_offset(wav, mid, lo, hi):
    y, sr = librosa.load(str(wav), sr=SR, mono=True)
    a = onset_env(y, sr)
    audio_dur = len(y) / sr
    pm = pretty_midi.PrettyMIDI(str(mid))
    gt = np.array(sorted(n.start for ins in pm.instruments for n in ins.notes))

    # 1) 성긴 탐색(10ms) — 정답이 오디오 안에 들어오는 오프셋만 후보로 인정.
    #    목적함수는 mean score가 아니라 **안착률**(주기 alias에 강함).
    coarse = [o for o in np.arange(lo, hi, 0.010) if _valid(gt, o, audio_dur)]
    if not coarse:
        raise SystemExit("유효한 오프셋 후보가 없습니다 — 오디오/정답이 다른 곡일 수 있음")
    sc = [hit_frac(a, gt, o, sr) for o in coarse]
    best_c = coarse[int(np.argmax(sc))]

    # 상위 후보 진단 출력 — alias가 있으면 사람이 알아챌 수 있게
    order = np.argsort(sc)[::-1][:3]
    print("  상위 오프셋 후보(안착률):",
          ", ".join(f"{coarse[i]:+.3f}s={sc[i]*100:.1f}%" for i in order))

    # 2) 정밀 탐색(1ms)
    fine = [o for o in np.arange(best_c - 0.05, best_c + 0.05, 0.001)
            if _valid(gt, o, audio_dur)]
    sf = [hit_frac(a, gt, o, sr) for o in fine]
    best = float(fine[int(np.argmax(sf))])

    # 진단: 정답 온셋이 실제 오디오 온셋에 얼마나 안착하는가
    se = sr / HOP
    idx = ((gt - best) * se).round().astype(int)
    hits = sum(1 for i in idx if 0 <= i < len(a) and a[max(0, i - 2):i + 3].max() > 1.0)

    return {
        "offset": best,
        "score": max(sf),
        "naive_first_note": float(gt[0]),
        "naive_error_ms": (float(gt[0]) - best) * 1000.0,
        "hit_frac": hits / len(gt),
        "n_notes": len(gt),
        "audio_dur": len(y) / sr,
        "gt_span": (float(gt.min() - best), float(gt.max() - best)),
    }


def write_shifted(mid, out, offset):
    pm = pretty_midi.PrettyMIDI(str(mid))
    out_pm = pretty_midi.PrettyMIDI()
    for ins in pm.instruments:
        ni = pretty_midi.Instrument(program=ins.program, is_drum=ins.is_drum, name=ins.name)
        for n in ins.notes:
            s, e = n.start - offset, n.end - offset
            if e <= 0:
                continue
            ni.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch,
                start=max(0.0, s), end=max(0.02, e)))
        out_pm.instruments.append(ni)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_pm.write(str(out))
    return sum(len(i.notes) for i in out_pm.instruments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("mid")
    ap.add_argument("out")
    ap.add_argument("--lo", type=float, default=-5.0)
    ap.add_argument("--hi", type=float, default=60.0)
    args = ap.parse_args()

    info = find_offset(args.wav, args.mid, args.lo, args.hi)
    print(f"offset        = {info['offset']:.3f}s   (안착률 {info['score']*100:.1f}%)")
    print(f"naive 첫노트  = {info['naive_first_note']:.3f}s "
          f"→ 오차 {info['naive_error_ms']:+.0f}ms")
    print(f"GT 온셋 안착률 = {info['hit_frac']*100:.1f}%  ({info['n_notes']} notes)")
    print(f"GT span       = {info['gt_span'][0]:.2f}..{info['gt_span'][1]:.2f}s "
          f"(audio {info['audio_dur']:.2f}s)")

    n = write_shifted(args.mid, args.out, info["offset"])
    print(f"wrote {args.out}  ({n} notes)")

    if info["hit_frac"] < 0.7:
        print("⚠ 안착률이 낮습니다 — 오프셋이 틀렸거나 오디오/정답이 다른 곡일 수 있음")


if __name__ == "__main__":
    main()
