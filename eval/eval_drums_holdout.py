"""드럼 ADT 정직한 홀드아웃 end-to-end 채점.

`train_drum_classifier.py`의 per-class 지표는 "후보별 이진 분류 정확도"라 파이프라인
최종 산출물(MIDI)의 품질과 다르다. 이 스크립트는 **앞 70%로만 학습한 모델로 뒤 30%
구간의 MIDI를 생성해 정답과 대조** — 실제 배포 시 처음 보는 구간에서의 성능 추정.

⚠ 정답 곡이 1곡뿐이라 이건 "같은 킷/같은 곡의 뒷부분"이다. **다른 킷/곡 일반화는 여전히
미검증** — 새 정답 곡이 확보되면 leave-one-song-out으로 교체할 것.

사용법: python eval/eval_drums_holdout.py [--split 0.7]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "backend" / "scripts"))
sys.path.insert(0, str(BASE / "eval"))

from drum_features import CLASSES  # noqa: E402
from score_drums import match_onsets, prf  # noqa: E402
from train_drum_classifier import build_dataset, fit_class  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print("후보 검출 + 피처 추출 중...")
    times, X, Y, gt = build_dataset()

    t_split = times[0] + (times[-1] - times[0]) * args.split
    tr = times < t_split
    te = ~tr
    print(f"학습: ~{t_split:.1f}s 이전 후보 {tr.sum()}개 / "
          f"평가: 이후 후보 {te.sum()}개\n")

    print(f"{'class':14s} {'ref':>5s} {'est':>5s} {'match':>6s} "
          f"{'P':>6s} {'R':>6s} {'F1':>6s}")
    print("-" * 56)

    tot_m = tot_eo = tot_ro = 0
    est_all, ref_all = [], []

    for ci, cls in enumerate(CLASSES):
        ref = gt[cls]
        ref = ref[ref >= t_split]          # 평가 구간 정답만
        ref_all.extend(ref.tolist())

        ytr = Y[tr, ci]
        if ytr.sum() < 5:
            est = np.array([])
        else:
            clf = fit_class(X[tr], ytr)
            prob = clf.predict_proba(X[te])[:, 1]
            est = times[te][prob >= args.threshold]
        est_all.extend(est.tolist())

        if len(ref) == 0 and len(est) == 0:
            continue

        m, eo, ro = match_onsets(np.sort(est), np.sort(ref), 0.05)
        p, r, f = prf(m, eo, ro)
        tot_m += m; tot_eo += eo; tot_ro += ro
        note = "  (평가구간 정답 없음)" if len(ref) == 0 else ""
        print(f"{cls:14s} {len(ref):5d} {len(est):5d} {m:6d} "
              f"{p:6.3f} {r:6.3f} {f:6.3f}{note}")

    p, r, f = prf(tot_m, tot_eo, tot_ro)
    print("-" * 56)
    print(f"{'MICRO (전체)':14s} {tot_m+tot_ro:5d} {tot_m+tot_eo:5d} {tot_m:6d} "
          f"{p:6.3f} {r:6.3f} {f:6.3f}")

    m, eo, ro = match_onsets(np.sort(est_all), np.sort(ref_all), 0.05)
    p, r, f = prf(m, eo, ro)
    print(f"{'ONSET-ONLY':14s} {len(ref_all):5d} {len(est_all):5d} {m:6d} "
          f"{p:6.3f} {r:6.3f} {f:6.3f}   <- class 무시, 검출력만")


if __name__ == "__main__":
    main()
