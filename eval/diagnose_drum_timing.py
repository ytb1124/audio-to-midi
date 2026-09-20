"""드럼 ADT 타이밍/듀레이션 진단.

사용자 신고("박자·타이밍·듀레이션 일치에 신경 써달라", "하이햇이 불안정") 대응.
class별 onset 편차(bias/std)와 듀레이션 오차를 정량화한다.

- onset_dev = 생성 onset − 정답 onset (양수면 늦음). **계통 편향(bias)** 이 있으면
  전역 보정으로 바로 고칠 수 있고, std가 크면 검출 지터 문제.
- 하이햇 지터는 별도로 IOI(연속 타격 간격) 규칙성으로도 본다.

사용법: python eval/diagnose_drum_timing.py <생성.mid> <정답.mid> [--tol 0.05]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_drums import CLASS_MAP, CLASS_ORDER  # noqa: E402


def load(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    by = defaultdict(list)
    for ins in pm.instruments:
        for n in ins.notes:
            c = CLASS_MAP.get(n.pitch)
            if c:
                by[c].append((n.start, n.end - n.start))
    for c in by:
        by[c].sort()
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("est_midi")
    ap.add_argument("ref_midi")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    est, ref = load(args.est_midi), load(args.ref_midi)

    print(f"{'class':14s} {'matched':>7s} {'bias_ms':>8s} {'std_ms':>7s} "
          f"{'|dev|>25ms':>10s} {'dur_est':>8s} {'dur_ref':>8s}")
    print("-" * 70)

    all_dev = []
    for c in CLASS_ORDER:
        e, r = est.get(c, []), ref.get(c, [])
        if not e or not r:
            continue
        rt = np.array([x[0] for x in r])
        used = np.zeros(len(rt), bool)
        devs, edur, rdur = [], [], []
        for t, d in e:
            lo = np.searchsorted(rt, t - args.tol)
            hi = np.searchsorted(rt, t + args.tol)
            best, bd = -1, None
            for i in range(lo, hi):
                if used[i]:
                    continue
                dd = abs(rt[i] - t)
                if bd is None or dd < bd:
                    best, bd = i, dd
            if best >= 0:
                used[best] = True
                devs.append((t - rt[best]) * 1000.0)
                edur.append(d)
                rdur.append(r[best][1])
        if not devs:
            continue
        devs = np.array(devs)
        all_dev.extend(devs.tolist())
        big = int((np.abs(devs) > 25).sum())
        print(f"{c:14s} {len(devs):7d} {np.median(devs):8.1f} {devs.std():7.1f} "
              f"{big:6d}({big/len(devs)*100:3.0f}%) "
              f"{np.median(edur):8.3f} {np.median(rdur):8.3f}")

    if all_dev:
        a = np.array(all_dev)
        print("-" * 70)
        print(f"{'전체':14s} {len(a):7d} {np.median(a):8.1f} {a.std():7.1f} "
              f"{int((np.abs(a)>25).sum()):6d}({(np.abs(a)>25).mean()*100:3.0f}%)")

    # 하이햇 지터: 연속 하이햇 간격의 규칙성(정답 대비)
    for c in ["hihat_closed", "shaker"]:
        e, r = est.get(c, []), ref.get(c, [])
        if len(e) > 10 and len(r) > 10:
            ei = np.diff([x[0] for x in e]) * 1000
            ri = np.diff([x[0] for x in r]) * 1000
            print(f"\n[{c}] IOI 중앙 {np.median(ei):.0f}ms(생성) vs "
                  f"{np.median(ri):.0f}ms(정답), "
                  f"IOI std {ei.std():.0f} vs {ri.std():.0f}ms  "
                  f"(std가 크게 높으면 지터/과검출)")


if __name__ == "__main__":
    main()
