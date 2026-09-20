"""Drum ADT 채점 하니스 — class별 onset F1.

Bass Mode의 `eval/score.py`와 같은 역할. "귀 판단이 아니라 점수로 판단"하기 위한 도구.

드럼은 pitch가 아니라 **"어떤 타악기(class) + 언제(onset)"** 의 문제라, class별로
독립적으로 onset matching F1을 낸다(greedy, 기본 허용오차 50ms — bass harness와 동일).

사용법:
    python eval/score_drums.py <생성된.mid> <정답.mid> [--tol 0.05]

정답/생성 MIDI의 GM pitch가 서로 달라도(예: 정답 kick=35, 생성 kick=36) 아래
CLASS_MAP으로 같은 class에 묶어서 비교한다 — 프로덕션이 어떤 GM 번호를 쓰든
"악기 종류가 맞았는가"를 채점하기 위함.
"""

import argparse
from collections import defaultdict

import numpy as np
import pretty_midi


# GM 드럼 pitch → 지각적 class. 드럼 킷/DAW마다 번호가 달라 그룹으로 묶는다.
CLASS_MAP = {
    35: "kick", 36: "kick",
    37: "snare", 38: "snare", 39: "snare", 40: "snare",  # 사이드스틱/클랩 포함
    42: "hihat_closed", 44: "hihat_closed",
    46: "hihat_open",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    49: "crash", 52: "crash", 55: "crash", 57: "crash",
    51: "ride", 53: "ride", 59: "ride",
    80: "triangle", 81: "triangle",
    # 지속형 노이즈 퍼커션(쉐이커/카바사/마라카스) — drums_sample5에 1108개
    69: "shaker", 70: "shaker", 82: "shaker",
}

CLASS_ORDER = [
    "kick", "snare", "hihat_closed", "hihat_open",
    "tom", "crash", "ride", "triangle", "shaker",
]


def load_by_class(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    by_class = defaultdict(list)
    unmapped = defaultdict(int)

    for ins in pm.instruments:
        for n in ins.notes:
            cls = CLASS_MAP.get(n.pitch)
            if cls is None:
                unmapped[n.pitch] += 1
                continue
            by_class[cls].append(n.start)

    for cls in by_class:
        by_class[cls] = np.array(sorted(by_class[cls]))

    return by_class, unmapped


def match_onsets(est, ref, tol):
    """greedy onset matching. 반환: (matched, est_only, ref_only)."""
    if len(est) == 0 or len(ref) == 0:
        return 0, len(est), len(ref)

    used_ref = np.zeros(len(ref), dtype=bool)
    matched = 0

    for t in est:
        # 허용오차 내 가장 가까운 미사용 정답
        lo = np.searchsorted(ref, t - tol)
        hi = np.searchsorted(ref, t + tol)
        best_i, best_d = -1, None
        for i in range(lo, hi):
            if used_ref[i]:
                continue
            d = abs(ref[i] - t)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            used_ref[best_i] = True
            matched += 1

    return matched, len(est) - matched, len(ref) - matched


def prf(matched, est_only, ref_only):
    p = matched / (matched + est_only) if (matched + est_only) else 0.0
    r = matched / (matched + ref_only) if (matched + ref_only) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description="Drum ADT class별 onset F1")
    ap.add_argument("est_midi")
    ap.add_argument("ref_midi")
    ap.add_argument("--tol", type=float, default=0.05, help="onset 허용오차(초), 기본 50ms")
    args = ap.parse_args()

    est_by, est_unmapped = load_by_class(args.est_midi)
    ref_by, ref_unmapped = load_by_class(args.ref_midi)

    print(f"tolerance: {args.tol*1000:.0f}ms")
    print(f"{'class':14s} {'ref':>5s} {'est':>5s} {'match':>6s} "
          f"{'P':>6s} {'R':>6s} {'F1':>6s}")
    print("-" * 56)

    tot_m = tot_eo = tot_ro = 0
    for cls in CLASS_ORDER:
        est = est_by.get(cls, np.array([]))
        ref = ref_by.get(cls, np.array([]))
        if len(est) == 0 and len(ref) == 0:
            continue
        m, eo, ro = match_onsets(est, ref, args.tol)
        p, r, f = prf(m, eo, ro)
        tot_m += m; tot_eo += eo; tot_ro += ro
        print(f"{cls:14s} {len(ref):5d} {len(est):5d} {m:6d} "
              f"{p:6.3f} {r:6.3f} {f:6.3f}")

    p, r, f = prf(tot_m, tot_eo, tot_ro)
    print("-" * 56)
    print(f"{'MICRO (전체)':14s} {tot_m+tot_ro:5d} {tot_m+tot_eo:5d} {tot_m:6d} "
          f"{p:6.3f} {r:6.3f} {f:6.3f}")

    # class를 무시하고 "타격 시점만" 맞췄는지 — 검출(onset) 능력과 분류 능력을 분리해서 본다.
    est_all = np.array(sorted(t for v in est_by.values() for t in v))
    ref_all = np.array(sorted(t for v in ref_by.values() for t in v))
    m, eo, ro = match_onsets(est_all, ref_all, args.tol)
    p, r, f = prf(m, eo, ro)
    print(f"{'ONSET-ONLY':14s} {len(ref_all):5d} {len(est_all):5d} {m:6d} "
          f"{p:6.3f} {r:6.3f} {f:6.3f}   <- class 무시, 검출력만")

    if est_unmapped:
        print(f"\n[생성] CLASS_MAP에 없는 pitch: {dict(est_unmapped)}")
    if ref_unmapped:
        print(f"[정답] CLASS_MAP에 없는 pitch: {dict(ref_unmapped)}")


if __name__ == "__main__":
    main()
