#!/usr/bin/env python3
"""
경계(boundary)마다 merge(0)/keep_separate(1)를 분류하는 모델을 학습하고,
(1) bass_test 시간 홀드아웃, (2) cross-song 홀드아웃으로 평가한다.

비교 기준(baseline):
- always_keep: 절대 안 합침 (현재 파이프라인의 기본 동작과 사실상 동일)
- rms_heuristic: 이전에 시도했다 실패한 RMS rise ratio 임계값 방식
  (rms_ratio_short >= 1.2 면 keep, 아니면 merge)
- trained classifier: 이번에 학습한 분류기
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

FEATURE_COLS = [
    "gap", "dur_a", "dur_b", "dur_ratio", "vel_a", "vel_b", "vel_ratio",
    "rms_pre_short", "rms_post_short", "rms_ratio_short",
    "rms_pre_long", "rms_post_long", "rms_ratio_long",
    "cqt_pre", "cqt_post", "cqt_ratio", "onset_strength", "pitch",
]


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # fp: merge인데 keep으로 잘못 판단(분절 유지 = 기존 문제 재현)
    # fn: keep_separate인데 merge로 잘못 판단(진짜 노트 삭제 = 새로운 문제, 더 위험)
    print(f"  {name:28s} acc={acc:.4f}  (오분류: merge를 못 잡음={fp}, 진짜노트를 지워버림={fn})  n={len(y_true)}")
    return acc, fn


def main():
    df = pd.read_csv("eval/fragment_features.csv")

    bass_test = df[df["song_id"] == "bass_test"].copy()
    others = df[df["song_id"] != "bass_test"].copy()

    unique_b = sorted(bass_test["boundary"].unique())
    split_time = unique_b[int(len(unique_b) * 0.8)]
    train_bt = bass_test[bass_test["boundary"] < split_time]
    holdout_bt = bass_test[bass_test["boundary"] >= split_time].copy()

    print(f"bass_test train: {len(train_bt)}, holdout: {len(holdout_bt)}", file=sys.stderr)

    def rms_heuristic_pred(d, thresh=1.2):
        return (d["rms_ratio_short"] >= thresh).astype(int)

    for variant_name, train_df in [
        ("A) bass_test만 학습", train_bt),
        ("B) bass_test + 다른 3곡 전부", pd.concat([train_bt, others])),
    ]:
        print(f"\n########## {variant_name} ##########")
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0, class_weight="balanced")
        clf.fit(train_df[FEATURE_COLS], train_df["label"])

        print("\n=== bass_test holdout (마지막 20%) ===")
        y_true = holdout_bt["label"].values
        report("always_keep_separate", y_true, np.ones_like(y_true))
        report("rms_heuristic (기존 실패작)", y_true, rms_heuristic_pred(holdout_bt).values)
        report("trained classifier", y_true, clf.predict(holdout_bt[FEATURE_COLS]))

        songs_trained_on = set(train_df["song_id"].unique())
        for song in ["bass_sample", "bass_sample2", "bass_sample3", "bass_sample4", "bass_sample5"]:
            sub = df[df["song_id"] == song]
            leaked = " [학습에 씀 — 참고만]" if song in songs_trained_on else " [진짜 홀드아웃]"
            print(f"\n=== {song}{leaked} ===")
            y_true = sub["label"].values
            report("always_keep_separate", y_true, np.ones_like(y_true))
            report("rms_heuristic (기존 실패작)", y_true, rms_heuristic_pred(sub).values)
            report("trained classifier", y_true, clf.predict(sub[FEATURE_COLS]))

    # leave-one-out 1: bass_sample3(Contemporary Soul, 완전히 다른 패치) 홀드아웃
    print("\n########## C) 나머지 전부 학습, bass_sample3(다른 패치) 완전 홀드아웃 ##########")
    train_df = df[df["song_id"] != "bass_sample3"]
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0, class_weight="balanced")
    clf.fit(train_df[FEATURE_COLS], train_df["label"])
    held_out = df[df["song_id"] == "bass_sample3"]
    y_true = held_out["label"].values
    print("\n=== bass_sample3 (완전 홀드아웃) ===")
    report("always_keep_separate", y_true, np.ones_like(y_true))
    report("rms_heuristic (기존 실패작)", y_true, rms_heuristic_pred(held_out).values)
    report("trained classifier", y_true, clf.predict(held_out[FEATURE_COLS]))

    # leave-one-out 2: bass_sample5(toto-africa) 완전 홀드아웃, 같은 패치(Punchy Bottom)인
    # bass_sample4는 학습에 남겨서 "같은 패치, 다른 곡" 일반화를 본다.
    print("\n########## D) 나머지 전부 학습, bass_sample5(같은 패치의 다른 곡) 완전 홀드아웃 ##########")
    train_df = df[df["song_id"] != "bass_sample5"]
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0, class_weight="balanced")
    clf.fit(train_df[FEATURE_COLS], train_df["label"])
    held_out = df[df["song_id"] == "bass_sample5"]
    y_true = held_out["label"].values
    print("\n=== bass_sample5 (완전 홀드아웃) ===")
    report("always_keep_separate", y_true, np.ones_like(y_true))
    report("rms_heuristic (기존 실패작)", y_true, rms_heuristic_pred(held_out).values)
    report("trained classifier", y_true, clf.predict(held_out[FEATURE_COLS]))

    print("\n=== Feature importances (variant B) ===")
    from sklearn.inspection import permutation_importance

    train_df = pd.concat([train_bt, others])
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0, class_weight="balanced")
    clf.fit(train_df[FEATURE_COLS], train_df["label"])
    r = permutation_importance(clf, holdout_bt[FEATURE_COLS], holdout_bt["label"], n_repeats=8, random_state=0)
    for name, imp in sorted(zip(FEATURE_COLS, r.importances_mean), key=lambda x: -x[1]):
        print(f"  {name:18s} {imp:.4f}")

    # 최종 배포용 모델: 4곡 전체(bass_test 전체 포함, holdout 나누지 않고)로 재학습해서 저장.
    import joblib

    final_train_df = df  # 전체 데이터 사용 (평가는 위에서 이미 holdout으로 끝냈음)
    final_clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=0, class_weight="balanced")
    final_clf.fit(final_train_df[FEATURE_COLS], final_train_df["label"])
    model_path = "backend/scripts/bass_fragment_merge_model.joblib"
    joblib.dump({"model": final_clf, "feature_cols": FEATURE_COLS}, model_path)
    print(f"\nSaved final deployment model to {model_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
