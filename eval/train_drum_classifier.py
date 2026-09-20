"""드럼 class별 multi-label 분류기 학습 (다곡 + leave-one-song-out).

온셋 후보마다 9개 class 각각을 **독립 이진 분류**(동시타격 표현을 위해).
Bass Mode에서 검증된 패턴(HistGradientBoosting + 정직한 홀드아웃)을 그대로 적용.

**leave-one-song-out(LOSO)** — 한 곡을 통째로 빼고 학습해서 그 곡으로 평가.
드럼은 킷마다 음색이 완전히 달라(스네어 튜닝, 심벌 종류, 룸) **cross-kit 일반화가
핵심 리스크**다. 같은 곡 시간 홀드아웃은 같은 킷·같은 샘플이라 낙관적 —
LOSO만이 "처음 보는 킷"에서의 성능을 정직하게 추정한다.

사용법:
    python eval/train_drum_classifier.py            # LOSO 평가 + 전체재학습 후 저장
    python eval/train_drum_classifier.py --no-save  # 평가만
    python eval/train_drum_classifier.py --songs 4 6  # 특정 곡만
"""

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np
import librosa
import pretty_midi
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "backend" / "scripts"))
sys.path.insert(0, str(BASE / "eval"))

from drum_features import (  # noqa: E402
    detect_candidates, extract_features, feature_names, CLASSES, SR,
)
from score_drums import CLASS_MAP, match_onsets, prf  # noqa: E402


GT_DIR = BASE / "eval" / "ground_truth"
MODEL_OUT = BASE / "backend" / "scripts" / "drum_class_model.joblib"
CACHE_DIR = BASE / "eval" / "_drum_cache"  # 곡별 캐시(부분집합 실험을 빠르게)

SONGS = [f"drums_sample{i}" for i in (4, 5, 6, 7, 8, 9, 10, 11)]
MATCH_TOL = 0.05  # score_drums.py와 동일 기준


def build_song(name):
    """한 곡의 (후보시각, 피처, multi-label 라벨, class별 정답온셋)."""
    y, sr = librosa.load(str(GT_DIR / f"{name}.wav"), sr=SR, mono=True)
    times = detect_candidates(y, sr)
    X = extract_features(y, times, sr)

    pm = pretty_midi.PrettyMIDI(str(GT_DIR / f"{name}.mid"))
    gt = {c: [] for c in CLASSES}
    for ins in pm.instruments:
        for n in ins.notes:
            c = CLASS_MAP.get(n.pitch)
            if c in gt:
                gt[c].append(n.start)
    for c in gt:
        gt[c] = np.array(sorted(gt[c]))

    Y = np.zeros((len(times), len(CLASSES)), dtype=np.int8)
    for ci, c in enumerate(CLASSES):
        ref = gt[c]
        if len(ref) == 0:
            continue
        for i, t in enumerate(times):
            if np.searchsorted(ref, t + MATCH_TOL) > np.searchsorted(ref, t - MATCH_TOL):
                Y[i, ci] = 1

    return times, X, Y, gt


def load_all(songs, use_cache=True):
    """곡별로 캐시 — 부분집합 실험(특정 곡 제외 등)을 재추출 없이 돌릴 수 있게."""
    CACHE_DIR.mkdir(exist_ok=True)
    data = {}
    for s in songs:
        f = CACHE_DIR / f"{s}.npz"
        if use_cache and f.exists():
            d = np.load(f, allow_pickle=True)
            data[s] = tuple(d["v"])
            continue
        print(f"  {s} 피처 추출 중...")
        data[s] = build_song(s)
        np.savez(f, v=np.array(data[s], dtype=object))
    return data


def fit_class(Xtr, ytr):
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=0,
    ).fit(Xtr, ytr)


def loso(data, songs, threshold=0.5):
    """leave-one-song-out: 한 곡을 빼고 학습 → 그 곡 전체를 MIDI 수준으로 채점."""
    print("\n" + "=" * 66)
    print("Leave-One-Song-Out (처음 보는 킷에서의 정직한 성능)")
    print("=" * 66)

    summary = {}
    for held in songs:
        tr_songs = [s for s in songs if s != held]
        Xtr = np.vstack([data[s][1] for s in tr_songs])
        Ytr = np.vstack([data[s][2] for s in tr_songs])
        times_te, Xte, _, gt_te = data[held]

        print(f"\n--- holdout: {held}  (학습 {len(tr_songs)}곡 {Xtr.shape[0]}후보) ---")
        print(f"{'class':14s} {'ref':>5s} {'est':>5s} {'match':>6s} "
              f"{'P':>6s} {'R':>6s} {'F1':>6s}")
        print("-" * 56)

        tot_m = tot_eo = tot_ro = 0
        for ci, cls in enumerate(CLASSES):
            ref = gt_te[cls]
            ytr = Ytr[:, ci]
            if ytr.sum() < 5:
                est = np.array([])
            else:
                clf = fit_class(Xtr, ytr)
                prob = clf.predict_proba(Xte)[:, 1]
                est = times_te[prob >= threshold]

            if len(ref) == 0 and len(est) == 0:
                continue
            m, eo, ro = match_onsets(np.sort(est), np.sort(ref), MATCH_TOL)
            p, r, f = prf(m, eo, ro)
            tot_m += m; tot_eo += eo; tot_ro += ro
            flag = "  (정답 없음)" if len(ref) == 0 else ""
            print(f"{cls:14s} {len(ref):5d} {len(est):5d} {m:6d} "
                  f"{p:6.3f} {r:6.3f} {f:6.3f}{flag}")

        p, r, f = prf(tot_m, tot_eo, tot_ro)
        summary[held] = f
        print("-" * 56)
        print(f"{'MICRO':14s} {tot_m+tot_ro:5d} {tot_m+tot_eo:5d} {tot_m:6d} "
              f"{p:6.3f} {r:6.3f} {f:6.3f}")

    print("\n" + "=" * 66)
    print("LOSO 요약 (곡별 MICRO F1)")
    for s, f in summary.items():
        print(f"  {s:18s} {f:.4f}")
    print(f"  {'평균':18s} {np.mean(list(summary.values())):.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None,
                    help="모델 저장 경로. 기본은 배포 모델 경로. "
                         "⚠ LOSO 폴드 모델을 만들 땐 반드시 다른 경로를 줄 것 — "
                         "기본값이면 배포 모델을 덮어쓴다.")
    ap.add_argument("--no-loso", action="store_true",
                    help="LOSO 평가 루프 건너뛰고 학습·저장만(폴드 모델 대량 생성용)")
    args = ap.parse_args()
    out_path = pathlib.Path(args.out) if args.out else MODEL_OUT

    songs = [f"drums_sample{s}" for s in args.songs] if args.songs else SONGS

    print("데이터 준비...")
    data = load_all(songs, use_cache=not args.no_cache)

    total = sum(len(data[s][0]) for s in songs)
    print(f"\n곡 {len(songs)}개, 후보 총 {total}개, 피처 {data[songs[0]][1].shape[1]}차원")
    print(f"{'song':18s} {'후보':>6s}  class별 정답 노트 수")
    for s in songs:
        gt = data[s][3]
        nz = {c: int(len(v)) for c, v in gt.items() if len(v)}
        print(f"{s:18s} {len(data[s][0]):6d}  {nz}")

    if not args.no_loso:
        loso(data, songs, args.threshold)

    if args.no_save:
        return

    print("\n전체 곡으로 재학습 후 저장 중...")
    X = np.vstack([data[s][1] for s in songs])
    Y = np.vstack([data[s][2] for s in songs])
    models = {}
    for ci, c in enumerate(CLASSES):
        if Y[:, ci].sum() < 5:
            print(f"  {c}: 양성 {int(Y[:, ci].sum())}개 — 스킵")
            continue
        models[c] = fit_class(X, Y[:, ci])

    joblib.dump(
        {"models": models, "classes": CLASSES, "feature_names": feature_names(),
         "trained_on": songs},
        out_path,
    )
    print(f"저장: {out_path}  (class {len(models)}개, {len(songs)}곡 학습)")


if __name__ == "__main__":
    main()
