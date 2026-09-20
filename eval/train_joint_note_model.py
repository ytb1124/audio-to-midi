#!/usr/bin/env python3
"""
공유 트렁크 멀티태스크 신경망 — 옥타브 판정 + fragment(재타격/병합) 판정을
하나의 표현으로 함께 학습한다. (사용자 제안 아키텍처 3단계: "octave + fragment
joint model", 2026-07-13, 브랜치 bass-note-verifier)

가설: 두 판정 모두 "이 순간 오디오 증거가 얼마나 확실한가"라는 공통 잠재
표현을 공유할 수 있다면(어택 강도, CQT 하모닉 안정성, pYIN 신뢰도 등이
두 태스크 모두에서 "증거 강도"를 나타냄), 트렁크를 공유해서 학습하는 쪽이
완전히 분리된 두 개의 sklearn 분류기보다 일반화가 나을 수 있다.

기존 체계와의 관계: octave_features.csv/fragment_features.csv는 이미 검증된
피처 추출 파이프라인(eval/extract_octave_features.py,
eval/extract_fragment_features.py) 그대로 재사용 — 이번 실험은 "모델 구조"만
바꾼다. 라벨/피처 정의는 배포 중인 sklearn 분류기와 동일해서 정직하게
비교 가능하다.

평가 방식은 eval/train_octave_classifier.py와 동일한 leave-one-song-out
(variant C: bass_sample3 완전 홀드아웃, variant D: bass_sample5 완전 홀드아웃)
— 같은 기준으로 sklearn HGB(현재 배포 모델과 동일 방식)와 직접 비교한다.
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OCTAVE_FEATURE_COLS = [
    "cqt_e0", "cqt_e12", "cqt_e_minus12", "cqt_e19", "cqt_e24", "cqt_harmonic",
    "pyin_dist", "pyin_conf", "note_duration",
    "neighbor_support_mean", "neighbor_count",
]

FRAGMENT_FEATURE_COLS = [
    "gap", "dur_a", "dur_b", "dur_ratio", "vel_ratio",
    "rms_pre_short", "rms_post_short", "rms_ratio_short",
    "rms_pre_long", "rms_post_long", "rms_ratio_long",
    "cqt_pre", "cqt_post", "cqt_ratio", "onset_strength",
]

# note-validity 헤드 (사용자 제안 아키텍처 "note verification" — 2026-07-13):
# proposal이 진짜 노트인가(no_onset이 아닌가)를 판정. 정답 매칭 라벨이라
# 완벽하게 깨끗하고, 유령 꾸밈음 제거 + 누락 복구를 하나로 통합한다.
# 옵션2(2026-07-13): 서스테인 안정성 피처(sustain_*/pitch_std/harm_continuity)
# 추가 — 어려운 톤의 약한 어택 진짜 노트를 no_onset과 구분.
PROPOSAL_FEATURE_COLS = [
    "att_peak", "rms_ratio", "flux_peak", "pyin_voiced", "pyin_dist",
    "cqt_harmonic", "gap_prev", "gap_next", "audio_strength",
    "src_audio", "src_raw", "dur",
    "sustain_voiced", "sustain_dist", "pitch_std", "harm_continuity",
    "cqt_decay", "rms_decay",
]

# offset(듀레이션) regression 헤드 (옵션3, 2026-07-13): 노트 길이를 학습으로
# 예측. GT note-off 라벨이 깨끗. proposal 인코더/트렁크 표현을 공유하고
# 헤드만 별도 — 입력 피처(감쇠/gap 포함)는 PROPOSAL_FEATURE_COLS 그대로 사용.
# 타깃 gt_duration은 validity=1(진짜 노트)일 때만 정의되므로 loss는 마스킹.

HIDDEN = 32
DEVICE = "cpu"


class JointNoteModel(nn.Module):
    """태스크별 입력 인코더 -> 공유 트렁크 -> 태스크별 출력 헤드.

    입력 피처 공간이 달라도(옥타브 11개, fragment 15개) 공유가 되는 지점은
    인코더 *이후*의 은닉 표현 — 표준적인 hard-parameter-sharing MTL 구성.
    """

    def __init__(self, n_octave_feat, n_frag_feat, n_prop_feat, hidden=HIDDEN):
        super().__init__()
        self.octave_encoder = nn.Sequential(nn.Linear(n_octave_feat, hidden), nn.ReLU())
        self.frag_encoder = nn.Sequential(nn.Linear(n_frag_feat, hidden), nn.ReLU())
        self.prop_encoder = nn.Sequential(nn.Linear(n_prop_feat, hidden), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.octave_head = nn.Linear(hidden, 1)
        self.frag_head = nn.Linear(hidden, 1)
        self.validity_head = nn.Linear(hidden, 1)
        self.offset_head = nn.Linear(hidden, 1)  # log-duration regression
        self.onset_head = nn.Linear(hidden, 1)   # onset-correction (초, GT-proposal)
        self.velocity_head = nn.Linear(hidden, 1)  # velocity (0~1 정규화)

    def forward_octave(self, x):
        return self.octave_head(self.trunk(self.octave_encoder(x))).squeeze(-1)

    def forward_fragment(self, x):
        return self.frag_head(self.trunk(self.frag_encoder(x))).squeeze(-1)

    def _prop_repr(self, x):
        return self.trunk(self.prop_encoder(x))

    def forward_validity(self, x):
        return self.validity_head(self._prop_repr(x)).squeeze(-1)

    def forward_offset(self, x):
        # 노트 길이는 양수라 log 공간에서 회귀 (안정적, 분포 skew 완화)
        return self.offset_head(self._prop_repr(x)).squeeze(-1)

    def forward_onset(self, x):
        return self.onset_head(self._prop_repr(x)).squeeze(-1)

    def forward_velocity(self, x):
        return self.velocity_head(self._prop_repr(x)).squeeze(-1)


def standardize_fit(df, cols):
    mean = df[cols].mean()
    std = df[cols].std().replace(0, 1.0)
    return mean, std


def standardize_apply(df, cols, mean, std):
    return ((df[cols] - mean) / std).values.astype(np.float32)


def _balanced_weights(y):
    n1 = float((y == 1).sum()) or 1.0
    n0 = float((y == 0).sum()) or 1.0
    return torch.where(y == 1, 0.5 / n1 * (n0 + n1), 0.5 / n0 * (n0 + n1))


def train_joint(
    octave_train, fragment_train, proposal_train,
    octave_mean, octave_std, frag_mean, frag_std, prop_mean, prop_std,
    epochs=60, lr=1e-3, seed=0,
):
    torch.manual_seed(seed)
    model = JointNoteModel(
        len(OCTAVE_FEATURE_COLS), len(FRAGMENT_FEATURE_COLS), len(PROPOSAL_FEATURE_COLS)
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    ox = torch.tensor(standardize_apply(octave_train, OCTAVE_FEATURE_COLS, octave_mean, octave_std))
    oy = torch.tensor(octave_train["label"].values.astype(np.float32))

    fx = torch.tensor(standardize_apply(fragment_train, FRAGMENT_FEATURE_COLS, frag_mean, frag_std))
    fy = torch.tensor(fragment_train["label"].values.astype(np.float32))
    fw = _balanced_weights(fy)

    px = torch.tensor(standardize_apply(proposal_train, PROPOSAL_FEATURE_COLS, prop_mean, prop_std))
    py = torch.tensor(proposal_train["validity"].values.astype(np.float32))
    pw = _balanced_weights(py)
    # 회귀 타깃들: validity=1(진짜)일 때만 유효 → 공통 마스크
    dur = proposal_train["gt_duration"].values.astype(np.float32)
    p_logdur = torch.tensor(np.log(np.clip(dur, 0.03, None)))
    p_onset = torch.tensor(proposal_train["onset_delta"].values.astype(np.float32))  # 초
    p_vel = torch.tensor((proposal_train["gt_velocity"].values.astype(np.float32)) / 127.0)
    p_offmask = torch.tensor((proposal_train["validity"].values == 1).astype(np.float32))

    octave_loss_fn = nn.BCEWithLogitsLoss()
    frag_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    val_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    offset_loss_fn = nn.HuberLoss(reduction="none", delta=0.3)
    onset_loss_fn = nn.HuberLoss(reduction="none", delta=0.02)  # 초 단위, 20ms delta
    vel_loss_fn = nn.HuberLoss(reduction="none", delta=0.1)

    batch_o = min(256, len(ox))
    batch_f = min(64, len(fx))
    batch_p = min(128, len(px))

    for epoch in range(epochs):
        perm_o = torch.randperm(len(ox))
        perm_f = torch.randperm(len(fx))
        perm_p = torch.randperm(len(px))
        n_steps = max(len(ox) // batch_o, 1)

        for step in range(n_steps):
            model.train()
            opt.zero_grad()

            o_idx = perm_o[(step * batch_o) % len(ox): (step * batch_o) % len(ox) + batch_o]
            if len(o_idx) == 0:
                o_idx = perm_o[:batch_o]
            loss_o = octave_loss_fn(model.forward_octave(ox[o_idx]), oy[o_idx])

            f_start = (step * batch_f) % max(len(fx), 1)
            f_idx = perm_f[f_start: f_start + batch_f]
            if len(f_idx) == 0:
                f_idx = perm_f[:batch_f]
            loss_f = (frag_loss_fn(model.forward_fragment(fx[f_idx]), fy[f_idx]) * fw[f_idx]).mean()

            p_start = (step * batch_p) % max(len(px), 1)
            p_idx = perm_p[p_start: p_start + batch_p]
            if len(p_idx) == 0:
                p_idx = perm_p[:batch_p]
            loss_p = (val_loss_fn(model.forward_validity(px[p_idx]), py[p_idx]) * pw[p_idx]).mean()

            # 회귀 헤드들: 진짜 노트만 (공통 마스크)
            m = p_offmask[p_idx]
            if float(m.sum()) > 0:
                loss_off = (offset_loss_fn(model.forward_offset(px[p_idx]), p_logdur[p_idx]) * m).sum() / m.sum()
                loss_on = (onset_loss_fn(model.forward_onset(px[p_idx]), p_onset[p_idx]) * m).sum() / m.sum()
                loss_v = (vel_loss_fn(model.forward_velocity(px[p_idx]), p_vel[p_idx]) * m).sum() / m.sum()
            else:
                loss_off = loss_on = loss_v = torch.zeros(())

            (loss_o + loss_f + loss_p + 0.5 * loss_off + 0.5 * loss_on + 0.3 * loss_v).backward()
            opt.step()

    return model


def octave_note_accuracy(model, df, mean, std, score_fn=None):
    model.eval()
    x = torch.tensor(standardize_apply(df, OCTAVE_FEATURE_COLS, mean, std))
    with torch.no_grad():
        scores = torch.sigmoid(model.forward_octave(x)).numpy()
    d = df.copy()
    d["_score"] = scores
    correct, total = 0, 0
    for (song, start), g in d.groupby(["song_id", "note_start"]):
        total += 1
        best = g["_score"].idxmax()
        if d.loc[best, "label"] == 1:
            correct += 1
    return (correct / total if total else 0.0), correct, total


def fragment_metrics(model, df, mean, std):
    model.eval()
    x = torch.tensor(standardize_apply(df, FRAGMENT_FEATURE_COLS, mean, std))
    with torch.no_grad():
        scores = torch.sigmoid(model.forward_fragment(x)).numpy()
    pred = (scores >= 0.5).astype(int)
    y = df["label"].values
    tp = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 0) & (y == 1)).sum())
    fn = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 1) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (pred == y).mean()
    return {
        "merge_precision": precision, "merge_recall": recall, "merge_f1": f1,
        "accuracy": float(acc), "n_merge": int((y == 0).sum()), "n_keep": int((y == 1).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def validity_metrics(model, df, mean, std):
    """no_onset(가짜) 검출 성능. 파이프라인의 유령 꾸밈음 제거와 직결.
    no_onset을 positive로 놓고 precision/recall 측정 — precision이 높아야
    진짜 노트를 안 지운다(안전), recall이 높아야 가짜를 잘 잡는다."""
    model.eval()
    x = torch.tensor(standardize_apply(df, PROPOSAL_FEATURE_COLS, mean, std))
    with torch.no_grad():
        scores = torch.sigmoid(model.forward_validity(x)).numpy()  # P(real)
    pred_real = (scores >= 0.5).astype(int)
    y = df["validity"].values  # 1=real, 0=no_onset
    # no_onset(=0) 검출 관점
    pred_fake = 1 - pred_real
    true_fake = 1 - y
    tp = int(((pred_fake == 1) & (true_fake == 1)).sum())
    fp = int(((pred_fake == 1) & (true_fake == 0)).sum())
    fn = int(((pred_fake == 0) & (true_fake == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "noonset_precision": precision, "noonset_recall": recall, "noonset_f1": f1,
        "n_fake": int(true_fake.sum()), "n_real": int(y.sum()),
        "deleted_real": fp,  # 진짜 노트를 no_onset으로 잘못 지운 수
    }


def offset_metrics(model, df, mean, std):
    """진짜 노트에 대해 예측 duration vs GT duration 오차(초). 파이프라인의
    현재 CQT-감쇠 규칙(cqt_decay 피처)과도 비교."""
    real = df[df["validity"] == 1].copy()
    if len(real) == 0:
        return None
    model.eval()
    x = torch.tensor(standardize_apply(real, PROPOSAL_FEATURE_COLS, mean, std))
    with torch.no_grad():
        pred_ld = model.forward_offset(x).numpy()
    pred_dur = np.exp(pred_ld)
    gt_dur = real["gt_duration"].values
    mae = float(np.mean(np.abs(pred_dur - gt_dur)))
    median_ae = float(np.median(np.abs(pred_dur - gt_dur)))
    # baseline: cqt_decay 피처를 그대로 duration 추정으로 (현재 규칙 근사)
    base_mae = float(np.mean(np.abs(real["cqt_decay"].values - gt_dur)))
    return {"mae": mae, "median_ae": median_ae, "baseline_cqt_decay_mae": base_mae, "n": len(real)}


def onset_metrics(model, df, mean, std):
    """onset-correction: 예측 이동량 vs 실제 GT 시각차. 보정 후 잔차가
    작아야 함. baseline은 '보정 안 함'(항상 0 예측)일 때의 잔차."""
    real = df[df["validity"] == 1].copy()
    if len(real) == 0:
        return None
    model.eval()
    x = torch.tensor(standardize_apply(real, PROPOSAL_FEATURE_COLS, mean, std))
    with torch.no_grad():
        pred = model.forward_onset(x).numpy()
    target = real["onset_delta"].values
    resid_before = np.abs(target)          # 보정 전 (proposal이 이미 얼마나 벗어남)
    resid_after = np.abs(target - pred)     # 학습 보정 후 잔차
    return {
        "before_mae": float(np.mean(resid_before)), "after_mae": float(np.mean(resid_after)),
        "before_median": float(np.median(resid_before)), "after_median": float(np.median(resid_after)),
        "n": len(real),
    }


def velocity_metrics(model, df, mean, std):
    real = df[df["validity"] == 1].copy()
    if len(real) == 0:
        return None
    model.eval()
    x = torch.tensor(standardize_apply(real, PROPOSAL_FEATURE_COLS, mean, std))
    with torch.no_grad():
        pred = model.forward_velocity(x).numpy() * 127.0
    gt = real["gt_velocity"].values
    mae = float(np.mean(np.abs(pred - gt)))
    # GT velocity가 사실상 평탄한 곡(bass_test 등)에서는 상관 자체가 의미 없음
    corr = float(np.corrcoef(pred, gt)[0, 1]) if np.std(gt) > 1 else 0.0
    return {"mae": mae, "corr": corr, "gt_std": float(np.std(gt)), "n": len(real)}


def main():
    octave_df = pd.read_csv("eval/octave_features.csv")
    fragment_df = pd.read_csv("eval/fragment_features.csv")
    proposal_df = pd.read_csv("eval/proposal_features.csv")

    print(f"octave rows: {len(octave_df)}, fragment rows: {len(fragment_df)}, "
          f"proposal rows: {len(proposal_df)} (no_onset={int((proposal_df['validity']==0).sum())})",
          file=sys.stderr)

    for variant_name, holdout_song in [
        ("C) bass_sample3 완전 홀드아웃", "bass_sample3"),
        ("D) bass_sample5 완전 홀드아웃", "bass_sample5"),
    ]:
        print(f"\n########## {variant_name} ##########")

        o_train = octave_df[octave_df["song_id"] != holdout_song]
        o_hold = octave_df[octave_df["song_id"] == holdout_song]
        f_train = fragment_df[fragment_df["song_id"] != holdout_song]
        f_hold = fragment_df[fragment_df["song_id"] == holdout_song]
        p_train = proposal_df[proposal_df["song_id"] != holdout_song]
        p_hold = proposal_df[proposal_df["song_id"] == holdout_song]

        o_mean, o_std = standardize_fit(o_train, OCTAVE_FEATURE_COLS)
        f_mean, f_std = standardize_fit(f_train, FRAGMENT_FEATURE_COLS)
        p_mean, p_std = standardize_fit(p_train, PROPOSAL_FEATURE_COLS)

        model = train_joint(o_train, f_train, p_train,
                            o_mean, o_std, f_mean, f_std, p_mean, p_std)

        acc, c, t = octave_note_accuracy(model, o_hold, o_mean, o_std)
        print(f"\n[옥타브] {holdout_song} 홀드아웃 joint-MTL note accuracy: {acc:.4f} ({c}/{t})")
        cqt_acc_correct = int((o_hold.loc[o_hold.groupby("note_start")["cqt_harmonic"].idxmax()]["label"] == 1).sum())
        cqt_acc_total = o_hold["note_start"].nunique()
        print(f"[옥타브] {holdout_song} cqt_harmonic argmax baseline: {cqt_acc_correct/cqt_acc_total:.4f} ({cqt_acc_correct}/{cqt_acc_total})")

        v = validity_metrics(model, p_hold, p_mean, p_std)
        print(
            f"[note-validity] {holdout_song} 홀드아웃 no_onset 검출: "
            f"precision={v['noonset_precision']:.4f} recall={v['noonset_recall']:.4f} "
            f"f1={v['noonset_f1']:.4f} (가짜 n={v['n_fake']}, 진짜 n={v['n_real']}, "
            f"진짜 오삭제={v['deleted_real']})"
        )

        om2 = offset_metrics(model, p_hold, p_mean, p_std)
        if om2:
            print(
                f"[offset] {holdout_song} 홀드아웃 duration 예측 MAE={om2['mae']*1000:.0f}ms "
                f"(median {om2['median_ae']*1000:.0f}ms) vs cqt_decay 규칙 baseline "
                f"{om2['baseline_cqt_decay_mae']*1000:.0f}ms (n={om2['n']})"
            )

        onm = onset_metrics(model, p_hold, p_mean, p_std)
        if onm:
            print(
                f"[onset-correction] {holdout_song} 홀드아웃: 보정 전 잔차 "
                f"median {onm['before_median']*1000:.0f}ms -> 보정 후 {onm['after_median']*1000:.0f}ms "
                f"(mean {onm['before_mae']*1000:.0f}->{onm['after_mae']*1000:.0f}ms, n={onm['n']})"
            )

        vm = velocity_metrics(model, p_hold, p_mean, p_std)
        if vm:
            print(
                f"[velocity] {holdout_song} 홀드아웃: MAE={vm['mae']:.1f} corr={vm['corr']:.2f} "
                f"(GT std={vm['gt_std']:.1f} — 낮으면 GT가 프로그래밍값이라 무의미)"
            )

        if len(f_hold) > 0:
            m = fragment_metrics(model, f_hold, f_mean, f_std)
            print(
                f"[fragment] {holdout_song} 홀드아웃 joint-MTL: "
                f"acc={m['accuracy']:.4f} merge_f1={m['merge_f1']:.4f} "
                f"(merge n={m['n_merge']}, keep n={m['n_keep']})"
            )

    print("\n########## 전체 6곡 학습 (참고용, 리키지 있음) ##########")
    o_mean, o_std = standardize_fit(octave_df, OCTAVE_FEATURE_COLS)
    f_mean, f_std = standardize_fit(fragment_df, FRAGMENT_FEATURE_COLS)
    p_mean, p_std = standardize_fit(proposal_df, PROPOSAL_FEATURE_COLS)
    model = train_joint(octave_df, fragment_df, proposal_df,
                        o_mean, o_std, f_mean, f_std, p_mean, p_std)
    acc, c, t = octave_note_accuracy(model, octave_df, o_mean, o_std)
    v = validity_metrics(model, proposal_df, p_mean, p_std)
    print(f"[옥타브] 전체 note accuracy: {acc:.4f} ({c}/{t})")
    print(f"[note-validity] 전체 no_onset f1: {v['noonset_f1']:.4f} "
          f"(precision {v['noonset_precision']:.4f}, recall {v['noonset_recall']:.4f})")

    # 아티팩트 저장 (파이프라인 연결 실험용). 표준화 통계도 함께 저장해야
    # 추론 시 동일 전처리 재현 가능.
    ckpt = {
        "state_dict": model.state_dict(),
        "octave_cols": OCTAVE_FEATURE_COLS, "octave_mean": o_mean.to_dict(), "octave_std": o_std.to_dict(),
        "frag_cols": FRAGMENT_FEATURE_COLS, "frag_mean": f_mean.to_dict(), "frag_std": f_std.to_dict(),
        "prop_cols": PROPOSAL_FEATURE_COLS, "prop_mean": p_mean.to_dict(), "prop_std": p_std.to_dict(),
        "hidden": HIDDEN,
    }
    torch.save(ckpt, "backend/scripts/bass_joint_note_model.pt")
    print("\nSaved joint model checkpoint to backend/scripts/bass_joint_note_model.pt", file=sys.stderr)


if __name__ == "__main__":
    main()
