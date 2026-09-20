"""Drum Automatic Transcription (ADT).

research 문서 2번(Drum → MIDI). Stage 2-D.
드럼 오디오(가급적 Demucs로 분리한 drums.wav)를 GM 드럼 MIDI로 변환.

**v2 (학습 기반, 기본)** — 8 class: kick/snare/hihat_closed/hihat_open/tom/crash/ride/triangle
- drums_sample4 정답(1114노트) 실측: 온셋 이벤트 626개 중 **61%가 동시타격**이라
  "온셋 1개 = 악기 1개"는 recall 56%가 상한. → 온셋마다 class를 **독립 판정(multi-label)**.
- 5밴드 union으로 후보를 넉넉히 뽑고(이벤트 recall 95.2%), class별 이진 분류기가 걸러냄.
- 학습: `eval/train_drum_classifier.py`, 모델: `drum_class_model.joblib`, 피처: `drum_features.py`(공유).

**v1 (규칙 기반, 폴백)** — 모델 파일이 없으면 자동으로 3-class(kick/snare/hihat) 대역 규칙 사용.
전체 변환이 실패하지 않도록 안전 폴백(Bass Mode의 MTL fallback과 같은 방식).

velocity는 두 경로 모두 오디오 어택 에너지의 dB 다이나믹 레인지 매핑(Callender et al. —
velocity가 청감 품질을 크게 좌우).

madmom(Vogl CRNN)은 numpy 2.x 미지원이라 스택을 깨서 배제. librosa/scipy/sklearn만 사용.

사용법:
    python drum_audio_to_midi.py <input_audio> <output_midi> [--audit <csv>] [--rules]

stdout 마지막 줄에 JSON으로 class별 카운트를 찍는다(백엔드가 파싱).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import pretty_midi
from scipy.signal import butter, sosfiltfilt


SR = 44100
HOP = 256  # ~5.8ms @ 44.1kHz — 드럼 타격 분해능으로 충분

# GM 드럼 맵 (channel 10).
KICK = 36   # Bass Drum 1
SNARE = 38  # Acoustic Snare
HIHAT = 42  # Closed Hi-Hat

# class별 (대역 Hz, peak-pick delta, 최소 IOI ms, 노트 길이 s)
# delta는 온셋 envelope의 local-mean 대비 임계값. 대역/악기 특성에 맞춰 튜닝.
DRUM_CLASSES = {
    "kick":  {"pitch": KICK,  "band": (30, 130),    "delta": 0.18, "min_ioi_ms": 90, "dur": 0.12},
    "snare": {"pitch": SNARE, "band": (180, 2600),  "delta": 0.22, "min_ioi_ms": 80, "dur": 0.10},
    "hihat": {"pitch": HIHAT, "band": (7000, 16000),"delta": 0.16, "min_ioi_ms": 60, "dur": 0.05},
}


def bandpass(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = sr / 2.0
    high = min(high, nyq * 0.999)
    low = max(low, 1.0)
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, y)


def band_envelope(y_band: np.ndarray, sr: int):
    """대역 신호의 온셋 envelope(양의 flux)와 프레임 RMS(velocity용)를 반환."""
    # 프레임 RMS
    rms = librosa.feature.rms(y=y_band, frame_length=1024, hop_length=HOP)[0]
    # 양의 방향 차분 = 에너지 상승(어택) 지점
    flux = np.maximum(0.0, np.diff(rms, prepend=rms[:1]))
    # 정규화(대역별 스케일 차이 제거)
    if flux.max() > 1e-9:
        flux = flux / flux.max()
    return flux, rms


def detect_band_hits(y_band, sr, delta, min_ioi_ms, band):
    flux, rms = band_envelope(y_band, sr)
    wait = max(1, int((min_ioi_ms / 1000.0) * sr / HOP))

    peaks = librosa.util.peak_pick(
        flux,
        pre_max=int(0.03 * sr / HOP),
        post_max=int(0.03 * sr / HOP),
        pre_avg=int(0.10 * sr / HOP),
        post_avg=int(0.10 * sr / HOP),
        delta=delta,
        wait=wait,
    )
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP)
    # velocity 원자료: 각 피크 프레임의 RMS(dB)
    peak_rms = rms[peaks] if len(peaks) else np.array([])
    return times, peak_rms


def rms_to_velocity(peak_rms: np.ndarray) -> np.ndarray:
    """대역 내 실측 다이나믹 레인지(dB p10~p95)를 45~120으로 비례 매핑.

    Bass Mode의 dB 벨로시티 매핑과 같은 철학: 고른 연주는 고르게, 다이나믹한 연주는
    그만큼만. 컴프레션/게인에 무관하게 파일 내부 상대값으로 정규화.
    """
    if len(peak_rms) == 0:
        return np.array([], dtype=int)
    db = 20.0 * np.log10(np.maximum(peak_rms, 1e-6))
    lo = np.percentile(db, 10)
    hi = np.percentile(db, 95)
    span = max(hi - lo, 6.0)  # 최소 6dB 분모 — 평탄한 연주 과증폭 방지
    norm = np.clip((db - lo) / span, 0.0, 1.0)
    vel = np.round(45 + norm * (120 - 45)).astype(int)
    return np.clip(vel, 1, 127)


# v2 8-class → GM 드럼 pitch.
CLASS_PITCH = {
    "kick": 36,           # Bass Drum 1
    "snare": 38,          # Acoustic Snare
    "hihat_closed": 42,   # Closed Hi-Hat
    "hihat_open": 46,     # Open Hi-Hat
    "tom": 45,            # Low Tom (톰 세부(하이/로우/플로어)는 그룹으로 묶음)
    "crash": 49,          # Crash Cymbal 1
    "ride": 51,           # Ride Cymbal 1
    "triangle": 81,       # Open Triangle
    "shaker": 82,         # Shaker
}

# class별 폴백 길이(초) — 감쇠 측정이 실패했을 때만 사용.
#
# ⚠ 정답 듀레이션을 그대로 따라가지 않는 이유(8곡 실측): 정답 MIDI의 듀레이션은
# 대부분 **0.005s 트리거**이고 곡마다 제각각(snare 0.005~0.766s)이라 음악적 의미가 없다.
# 그대로 쓰면 피아노롤에서 노트가 보이지도 않는다. 대신 **실제 음향 감쇠**로 길이를 정하고
# 다음 같은-악기 타격을 넘지 않게 캡한다(Bass Mode duration_trim과 같은 철학).
CLASS_DUR = {
    "kick": 0.10, "snare": 0.08, "hihat_closed": 0.05, "hihat_open": 0.16,
    "tom": 0.14, "crash": 0.50, "ride": 0.25, "triangle": 0.20,
    "shaker": 0.05,
}

# 감쇠 측정용 대역(해당 악기 에너지가 실제로 사는 곳) + 길이 상·하한.
CLASS_DECAY_BAND = {
    "kick": (30, 160), "snare": (180, 2600), "hihat_closed": (5000, 16000),
    "hihat_open": (5000, 16000), "tom": (80, 700), "crash": (3000, 16000),
    "ride": (2500, 12000), "triangle": (4000, 16000), "shaker": (6000, 18000),
}
CLASS_DUR_RANGE = {
    "kick": (0.04, 0.40), "snare": (0.03, 0.40), "hihat_closed": (0.02, 0.20),
    "hihat_open": (0.05, 0.60), "tom": (0.05, 0.60), "crash": (0.10, 2.00),
    "ride": (0.05, 1.00), "triangle": (0.05, 1.00), "shaker": (0.02, 0.15),
}


def estimate_durations(y, sr, cls, times):
    """각 타격의 길이를 실제 음향 감쇠로 추정.

    해당 class 대역의 에너지가 어택 피크의 25% 아래로 떨어지는 지점까지를 길이로 본다.
    class별 상·하한으로 클램프하고, 호출부에서 다음 같은-악기 타격 직전으로 캡한다.
    """
    if len(times) == 0:
        return np.array([])

    lo, hi = CLASS_DECAY_BAND.get(cls, (30, 18000))
    dmin, dmax = CLASS_DUR_RANGE.get(cls, (0.03, 0.5))
    yb = bandpass(y, sr, lo, hi)

    k = max(1, int(0.002 * sr))
    env = np.convolve(np.abs(yb), np.ones(k) / k, mode="same")

    out = []
    for t in times:
        i0 = int(t * sr)
        i1 = min(len(env), i0 + int(dmax * sr))
        seg = env[i0:i1]
        if len(seg) < k * 2:
            out.append(CLASS_DUR.get(cls, 0.1))
            continue
        peak = seg[: max(k, int(0.015 * sr))].max()
        thr = peak * 0.25
        below = np.flatnonzero(seg < thr)
        d = below[0] / sr if len(below) else dmax
        out.append(float(np.clip(d, dmin, dmax)))

    return np.array(out)

MODEL_PATH = Path(__file__).resolve().parent / "drum_class_model.joblib"

# ★ 클래스별 불응기(ms) — 같은 악기를 이보다 빨리 다시 칠 수 없다는 물리 한계.
#
# 근거(정답 7곡 실측 연속 타격 간격의 최솟값): kick 125ms(2008개 간격 전부), hihat_closed
# 125ms, hihat_open 250ms, ride 177ms, crash 125ms. 그런데 생성 결과엔 43~56ms 간격의
# 킥 쌍이 있었고, **초과 킥 개수와 80ms 미만 쌍 개수가 정확히 일치**했다
# (sample11 +56개/56쌍, sample7 +35개/35쌍) — 즉 여분 킥은 전부 중복 트리거였다.
# 값은 정답 최솟값보다 넉넉히 낮게 잡아 정당한 빠른 연주를 자르지 않게 한다.
# snare/tom/shaker는 정답에서 최소 간격이 0ms(서로 다른 톰 동시타 등 같은 class로 묶임)라
# 불응기를 걸지 않는다.
REFRACTORY_MS = {
    "kick": 80,           # GT 바닥 125
    "hihat_closed": 60,   # GT 바닥 125 (다른 음악의 빠른 롤 여지를 남김)
    "hihat_open": 120,    # GT 바닥 250
    "ride": 100,          # GT 바닥 177
    "crash": 100,         # GT 바닥 125
}

# 1~2개 킷에서만 관측된 클래스 — cross-kit 검증이 불가능하고, pitch 81/82라
# 피아노롤에서 드럼 클러스터(35~51)와 동떨어져 보여 오검출이 특히 거슬린다.
# LOSO 실측: 셰이커가 없는 킷(sample11)에서 85개 오생성. 기본 출력에서 제외하고
# --exotic 으로만 켠다.
# 기본 출력에서 제외하고 `--exotic`으로만 켜는 class.
#
# triangle/shaker (v6): 1~2킷으로만 학습돼 cross-kit 검증 자체가 불가능했고, 없는 킷에서
# 대량 오생성했다(셰이커가 0개인 sample11에서 85개).
#
# ★ crash/ride 추가 (2026-07-26): 같은 구조의 문제가 심벌에서 실측됐다.
# - LOSO(처음 보는 킷)에서 crash 0.010 / ride 0.000 — 사실상 검출 불가.
# - E-GMD 41킷(10.3h)으로 학습해도 **모델이 "이 킷에 ride가 있는가"를 판별하지 못한다**:
#   7킷 풀링 정밀도 ride 0.294, 그 FP 453개 중 **385개(85%)가 ride가 아예 없는 킷**에서
#   나왔다(s5 혼자 252개). crash도 풀링 정밀도 0.588 / 재현율 0.058.
# - 파일 단위 점수 정규화(백분위, (x−med)/IQR)로 고치려 했으나 **오탐이 오히려 늘었다** —
#   절대 점수 수준이 "이 파일에 그 악기가 있는가"를 담고 있어서 정규화가 그 정보를 지운다.
# - v6가 톰에서 실측한 "확신도 게이팅 불가"(가짜 톰이 진짜보다 확신도가 높음)와 같은 층의 문제다.
#
# 가짜 심벌은 빠진 심벌보다 나쁘다 — 곡 구조를 잘못 말하기 때문에 사용자가 곡 전체를
# 훑어야 한다. 실사용 판정에서 진짜 품질을 가리지 않도록 기본 출력에서 뺀다.
EXOTIC_CLASSES = {"triangle", "shaker", "crash", "ride"}


# 톰 오생성 억제 — 킥+스네어가 동시에 울리면 그 합성 스펙트럼의 저중역(80~700Hz)이
# 톰처럼 보여 오발화한다.
#
# 실측 근거: 톰이 0개인 킷(sample6)에서 톰 FP 118개가 **전부 kick과 동시**, 101개는
# snare까지 동시였다. 반면 **정답에서 톰이 kick+snare와 동시에 울리는 경우는 2.0%**
# (305개 중 6개)뿐 — 드러머가 킥+스네어를 함께 치며 톰까지 치는 일은 드물다.
# LOSO 검증: sample6 오생성 118→19개(84% 제거), 다른 5곡은 **진짜 톰 손실 0**
# (sample4만 136→133). class를 독립 판정하는 구조의 한계를 물리 규칙으로 보완.
def suppress_tom(tom_hits, kick_hits, snare_hits):
    """kick과 snare가 같은 온셋에서 동시에 발화하면 그 자리의 톰을 억제."""
    return tom_hits & ~(kick_hits & snare_hits)


def apply_refractory(times, probs, cls):
    """같은 악기의 물리적 불응기 위반을 제거. 겹칠 땐 확신도가 높은 쪽을 남긴다."""
    gap = REFRACTORY_MS.get(cls)
    if gap is None or len(times) == 0:
        return np.ones(len(times), dtype=bool)

    gap /= 1000.0
    keep = np.ones(len(times), dtype=bool)
    last = -1
    for i in range(len(times)):
        if last >= 0 and times[i] - times[last] < gap:
            # 중복 트리거: 확신도가 낮은 쪽을 버린다.
            if probs[i] > probs[last]:
                keep[last] = False
                last = i
            else:
                keep[i] = False
        else:
            last = i
    return keep


def transcribe_model(input_audio: Path, output_midi: Path, audit_path, threshold: float,
                     allow_exotic: bool = False, model_path: Path = None):
    """v2: 온셋 후보 → class별 이진 분류기(multi-label) → GM 드럼 MIDI.

    model_path: 기본은 배포 모델(MODEL_PATH). LOSO 평가처럼 "그 곡을 빼고 학습한
    모델"로 채점해야 할 때만 다른 경로를 준다 — 배포 모델은 정답 8곡 전부로
    학습돼 있어 그 곡들로 채점하면 in-sample이라 성능이 낙관적으로 나온다.
    """
    import joblib
    import drum_features as df

    bundle = joblib.load(model_path or MODEL_PATH)
    models = bundle["models"]

    y, sr = librosa.load(str(input_audio), sr=df.SR, mono=True)
    times = df.detect_candidates(y, sr)
    if len(times) == 0:
        raise RuntimeError("온셋 후보를 찾지 못했습니다.")

    X = df.extract_features(y, times, sr)

    # 피처 정의를 바꾸고 모델을 재학습하지 않으면 차원이 어긋난다.
    # 조용히 이상한 예측을 내는 대신 명시적으로 실패시켜 규칙 폴백으로 보낸다.
    expected = bundle.get("feature_names")
    if expected is not None and X.shape[1] != len(expected):
        raise RuntimeError(
            f"피처 차원 불일치: 모델 {len(expected)} vs 현재 {X.shape[1]}. "
            "eval/train_drum_classifier.py로 재학습 필요."
        )
    energy = df.attack_energy(y, times, sr)
    vels = rms_to_velocity(energy)

    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    counts = {c: 0 for c in models}
    audit_rows = []

    # 1차: 모든 class 확률 → 발화 여부. class 간 규칙(톰 억제)에 다른 class 결정이 필요.
    probs, hits = {}, {}
    for cls, clf in models.items():
        if cls in EXOTIC_CLASSES and not allow_exotic:
            continue
        probs[cls] = clf.predict_proba(X)[:, 1]
        hits[cls] = probs[cls] >= threshold

    if "tom" in hits and "kick" in hits and "snare" in hits:
        hits["tom"] = suppress_tom(hits["tom"], hits["kick"], hits["snare"])

    # 2차: class별 불응기 적용 후 노트 생성.
    for cls in hits:
        prob = probs[cls]
        idx = np.flatnonzero(hits[cls])
        if len(idx) == 0:
            continue

        # 물리적 불응기 위반(중복 트리거) 제거 — "한 번 쳤는데 노트 두 개" 대응.
        keep = apply_refractory(times[idx], prob[idx], cls)
        idx = idx[keep]
        if len(idx) == 0:
            continue

        pitch = CLASS_PITCH[cls]
        hit_times = times[idx]
        durs = estimate_durations(y, sr, cls, hit_times)

        for k, i in enumerate(idx):
            t = float(hit_times[k])
            d = float(durs[k])
            # 같은 악기의 다음 타격을 넘지 않게 캡 — 피아노롤에서 겹쳐 보이지 않도록.
            # 빠른 연타(하이햇 롤 등)에선 간격이 좁으므로 20ms 최소를 강제하지 않고
            # 간격에 맞춰 줄인다(겹침 0 보장). 하한은 2ms.
            if k + 1 < len(hit_times):
                gap = float(hit_times[k + 1]) - t
                d = min(d, max(0.002, gap - 0.002))
            drum.notes.append(
                pretty_midi.Note(
                    velocity=int(vels[i]),
                    pitch=pitch,
                    start=t,
                    end=t + d,
                )
            )
            counts[cls] += 1
            audit_rows.append((cls, t, int(vels[i]), round(float(prob[i]), 4)))

    drum.notes.sort(key=lambda n: n.start)
    pm.instruments.append(drum)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_midi))

    if audit_path is not None:
        import csv

        audit_rows.sort(key=lambda r: r[1])
        with audit_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "time_s", "velocity", "confidence"])
            w.writerows(audit_rows)

    return counts


def transcribe(input_audio: Path, output_midi: Path, audit_path: Path | None):
    y, sr = librosa.load(str(input_audio), sr=SR, mono=True)

    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    counts = {}
    audit_rows = []

    for cls, cfg in DRUM_CLASSES.items():
        y_band = bandpass(y, sr, *cfg["band"])
        times, peak_rms = detect_band_hits(
            y_band, sr, cfg["delta"], cfg["min_ioi_ms"], cfg["band"]
        )
        vels = rms_to_velocity(peak_rms)

        for t, v in zip(times, vels):
            drum.notes.append(
                pretty_midi.Note(
                    velocity=int(v),
                    pitch=cfg["pitch"],
                    start=float(t),
                    end=float(t) + cfg["dur"],
                )
            )
            audit_rows.append((cls, float(t), int(v)))

        counts[cls] = int(len(times))

    pm.instruments.append(drum)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_midi))

    if audit_path is not None:
        import csv

        audit_rows.sort(key=lambda r: r[1])
        with audit_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "time_s", "velocity"])
            w.writerows(audit_rows)

    return counts


def main():
    parser = argparse.ArgumentParser(description="Drum ADT (kick/snare/hihat) → GM drum MIDI")
    parser.add_argument("input_audio")
    parser.add_argument("output_midi")
    parser.add_argument("--audit", default=None, help="선택: class/time/velocity 감사 CSV 경로")
    parser.add_argument("--rules", action="store_true",
                        help="학습 모델을 무시하고 v1 규칙 기반(3-class)으로 강제")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="v2 class 분류 임계값(기본 0.5)")
    parser.add_argument("--exotic", action="store_true",
                        help="triangle/shaker도 출력(1~2개 킷으로만 학습돼 기본 비활성)")
    parser.add_argument("--model", default=None,
                        help="분류기 모델 경로. 기본은 배포 모델. **LOSO 평가 전용** — "
                             "배포 모델은 정답 8곡 전부로 학습돼 있어 그 곡으로 채점하면 "
                             "in-sample이 된다. 그 곡을 뺀 모델을 여기에 준다.")
    args = parser.parse_args()

    input_audio = Path(args.input_audio).resolve()
    output_midi = Path(args.output_midi).resolve()
    audit_path = Path(args.audit).resolve() if args.audit else None
    model_path = Path(args.model).resolve() if args.model else None

    if not input_audio.exists():
        raise SystemExit(f"입력 오디오가 없습니다: {input_audio}")
    if model_path is not None and not model_path.exists():
        raise SystemExit(f"모델이 없습니다: {model_path}")

    # 기본은 v2(학습). 모델이 없거나 로드 실패하면 v1 규칙으로 안전 폴백 —
    # 전체 변환이 실패하는 것보다 3-class라도 내놓는 게 낫다(Bass Mode MTL fallback과 동일 방침).
    engine = "rules"
    used_model = None
    if not args.rules and (model_path or MODEL_PATH).exists():
        try:
            counts = transcribe_model(input_audio, output_midi, audit_path, args.threshold,
                                      allow_exotic=args.exotic, model_path=model_path)
            engine = "model"
            used_model = str(model_path or MODEL_PATH)
        except Exception as e:
            # ⚠ 명시적으로 --model을 준 평가 실행에서 조용히 규칙 폴백되면 그 결과가
            # LOSO 점수로 잘못 기록된다. 평가 경로에서는 폴백하지 않고 실패시킨다.
            if model_path is not None:
                raise SystemExit(f"지정 모델로 전사 실패(평가 경로라 폴백 안 함): {e}")
            print(f"[warn] 학습 모델 경로 실패 → 규칙 기반으로 폴백: {e}", file=sys.stderr)
            counts = transcribe(input_audio, output_midi, audit_path)
    else:
        counts = transcribe(input_audio, output_midi, audit_path)

    print(json.dumps({
        "counts": counts,
        "total": int(sum(counts.values())),
        "engine": engine,
        "model": used_model,
    }))


if __name__ == "__main__":
    main()
