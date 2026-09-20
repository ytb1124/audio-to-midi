#!/usr/bin/env python3
"""
eval/octave_features.csv (4곡: bass_test, bass_sample, bass_sample2, bass_sample3)로
후보-옥타브 분류기를 학습하고 두 가지 지표로 평가한다.

지표 1: raw accuracy — 후보 중 정답을 top-1으로 고르는 비율 (기존 방식)
지표 2: sporadic error rate — **사용자 우선순위 반영**. 파일 전체가 균일하게
        틀린 건 나중에 피아노롤에서 한 번에 고치면 되는 낮은 우선순위 문제.
        진짜 문제는 "주변 노트는 다 맞는 옥타브인데 이 노트만 튀는" 경우.
        그래서 근처(±2.5s) 같은 pitch class 노트들의 '정답' 옥타브가 서로
        일치하는(local context가 안정적인) 노트들만 골라서, 그 안에서
        top-1이 틀리는 비율을 별도로 측정한다.

비교 기준(baseline):
- always_offset0: 후보를 아예 안 바꾸는 경우
- cqt_harmonic_argmax: 지금까지 쓰던 CQT harmonic score로만 고르는 경우
- trained classifier: 이번에 학습한 분류기 (neighbor_support_mean 포함)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_COLS = [
    "cqt_e0", "cqt_e12", "cqt_e_minus12", "cqt_e19", "cqt_e24", "cqt_harmonic",
    "bp_note_conf", "bp_note_conf_up", "bp_note_conf_down", "bp_onset_conf",
    "pyin_dist", "pyin_conf", "note_duration",
    "neighbor_support_mean", "neighbor_count",
    "crepe_periodicity", "crepe_periodicity_up", "crepe_periodicity_down",
]

NEIGHBOR_WINDOW_SEC = 2.5


def note_level_accuracy(df_subset, score_col, note_filter=None):
    correct = 0
    total = 0
    for (song_id, note_start), group in df_subset.groupby(["song_id", "note_start"]):
        if note_filter is not None and (song_id, note_start) not in note_filter:
            continue
        total += 1
        best_idx = group[score_col].idxmax()
        if df_subset.loc[best_idx, "label"] == 1:
            correct += 1
    return (correct / total if total else 0.0), correct, total


def locally_stable_notes(df_song):
    """song 내에서, 근처(±2.5s) 같은 pitch class 노트들의 true_pitch가
    이 노트의 true_pitch와 전부(또는 대다수) 일치하는 노트 집합을 찾는다."""
    notes = df_song[["note_start", "note_end", "true_pitch"]].drop_duplicates().values.tolist()
    notes.sort(key=lambda x: x[0])

    stable_keys = set()
    for start, end, true_pitch in notes:
        pitch_class = int(true_pitch) % 12
        neighbors = [
            tp for (ns, ne, tp) in notes
            if abs(ns - start) <= NEIGHBOR_WINDOW_SEC and int(tp) % 12 == pitch_class and ns != start
        ]
        if not neighbors:
            continue
        agree = sum(1 for tp in neighbors if tp == true_pitch)
        if agree / len(neighbors) >= 0.8:  # 80% 이상의 이웃이 나와 같은 옥타브
            stable_keys.add(start)

    return stable_keys


def main():
    df = pd.read_csv("eval/octave_features.csv")

    bass_test = df[df["song_id"] == "bass_test"].copy()
    others = df[df["song_id"] != "bass_test"].copy()

    unique_starts = sorted(bass_test["note_start"].unique())
    split_idx = int(len(unique_starts) * 0.8)
    split_time = unique_starts[split_idx]

    train_bt = bass_test[bass_test["note_start"] < split_time]
    holdout_bt = bass_test[bass_test["note_start"] >= split_time].copy()

    print(f"bass_test train: {train_bt['note_start'].nunique()} notes, holdout: {holdout_bt['note_start'].nunique()} notes", file=sys.stderr)
    print(f"other songs (sample/sample2/sample3): {others['note_start'].nunique()} notes total", file=sys.stderr)

    # locally-stable(문맥 안정) 노트 집합 계산 (holdout 부분만)
    stable_starts = locally_stable_notes(bass_test[bass_test["note_start"] >= split_time])
    stable_keys = {("bass_test", s) for s in stable_starts}
    print(f"locally-stable notes in holdout: {len(stable_starts)} / {holdout_bt['note_start'].nunique()}", file=sys.stderr)

    # variant C: leave-one-song-out — bass_sample3(Contemporary Soul, 다른 패치)를
    # 완전히 학습에서 빼고, 나머지 전부(Crunchy Pop x2 + Punchy Bottom x2 + bass_test)로
    # 학습해서 "본 적 없는 systematic-tone 악기"에 일반화되는지 정직하게 테스트.
    not_sample3 = df[df["song_id"] != "bass_sample3"]
    # variant D: bass_sample5(toto-africa) 완전 홀드아웃, 같은 패치인 bass_sample4는 학습에 남김
    not_sample5 = df[df["song_id"] != "bass_sample5"]

    for variant_name, train_df in [
        ("A) bass_test만으로 학습", train_bt),
        ("B) bass_test + 다른 5곡 전부 포함 (주의: 아래 sample* 결과는 학습에 쓴 데이터라 참고만)", pd.concat([train_bt, others])),
        ("C) bass_sample3 제외 나머지 전부 학습, bass_sample3(다른 패치)는 완전 홀드아웃", not_sample3),
        ("D) bass_sample5 제외 나머지 전부 학습, bass_sample5(같은 패치의 다른 곡)는 완전 홀드아웃", not_sample5),
    ]:
        print(f"\n########## {variant_name} ##########")
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0)
        clf.fit(train_df[FEATURE_COLS], train_df["label"])

        holdout_bt["clf_score"] = clf.predict_proba(holdout_bt[FEATURE_COLS])[:, 1]

        print("\n=== bass_test holdout: RAW accuracy (전체) ===")
        for label, col in [
            ("cqt_harmonic argmax (기존 방식)", "cqt_harmonic"),
            ("trained classifier", "clf_score"),
        ]:
            acc, correct, total = note_level_accuracy(holdout_bt, col)
            print(f"  {label:35s} acc={acc:.4f} ({correct}/{total})")

        print("\n=== bass_test holdout: SPORADIC error rate (문맥상 안정된 노트만) ===")
        print("    (사용자 우선순위: 이게 진짜 중요한 지표)")
        for label, col in [
            ("cqt_harmonic argmax (기존 방식)", "cqt_harmonic"),
            ("trained classifier", "clf_score"),
        ]:
            acc, correct, total = note_level_accuracy(holdout_bt, col, note_filter=stable_keys)
            print(f"  {label:35s} acc={acc:.4f} ({correct}/{total}) -> error_rate={1-acc:.4f}")

        songs_trained_on = set(train_df["song_id"].unique())
        for song in ["bass_sample", "bass_sample2", "bass_sample3", "bass_sample4", "bass_sample5"]:
            sub = df[df["song_id"] == song].copy()
            sub["clf_score"] = clf.predict_proba(sub[FEATURE_COLS])[:, 1]
            acc_cqt, c1, t1 = note_level_accuracy(sub, "cqt_harmonic")
            acc_clf, c2, t2 = note_level_accuracy(sub, "clf_score")
            leaked = " [경고: 이 곡 데이터로 학습함 — 정직한 테스트 아님, 참고만]" if song in songs_trained_on else " [진짜 홀드아웃]"
            print(f"\n=== {song} (systematic-tone, 낮은 우선순위){leaked} ===")
            print(f"  cqt_harmonic argmax: {acc_cqt:.4f} ({c1}/{t1})  |  trained classifier: {acc_clf:.4f} ({c2}/{t2})")

    print("\n=== Feature importances (variant B, permutation) ===")
    try:
        from sklearn.inspection import permutation_importance

        train_df = pd.concat([train_bt, others])
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0)
        clf.fit(train_df[FEATURE_COLS], train_df["label"])
        r = permutation_importance(clf, holdout_bt[FEATURE_COLS], holdout_bt["label"], n_repeats=5, random_state=0)
        for name, imp in sorted(zip(FEATURE_COLS, r.importances_mean), key=lambda x: -x[1]):
            print(f"  {name:22s} {imp:.4f}")
    except Exception as exc:
        print("  (skipped:", exc, ")")

    # 배포용: CREPE/BasicPitch confidence는 permutation importance ~0이라 제외
    # (계산 비용만 크고 기여 없음). CQT 변형 + pYIN + neighbor context만 사용.
    DEPLOY_FEATURE_COLS = [
        "cqt_e0", "cqt_e12", "cqt_e_minus12", "cqt_e19", "cqt_e24", "cqt_harmonic",
        "pyin_dist", "pyin_conf", "note_duration",
        "neighbor_support_mean", "neighbor_count",
    ]

    import joblib

    final_clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0)
    final_clf.fit(df[DEPLOY_FEATURE_COLS], df["label"])
    model_path = "backend/scripts/bass_octave_final_model.joblib"
    joblib.dump({"model": final_clf, "feature_cols": DEPLOY_FEATURE_COLS}, model_path)
    print(f"\nSaved final deployment model (no CREPE/BasicPitch, 6곡 전체 학습) to {model_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
