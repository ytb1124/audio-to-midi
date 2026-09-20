"""드럼 ADT 공유 모듈 — 온셋 후보 검출 + 피처 추출.

학습(eval/train_drum_classifier.py)과 추론(drum_audio_to_midi.py)이 **반드시 같은 코드**를
쓰도록 여기 한 곳에만 둔다. (CLAUDE.md 교훈: 학습/추론 피처가 어긋나면 오프라인 점수가
실전에서 재현되지 않는다.)

구조적 근거(drums_sample4 정답 실측):
- 1114노트 = 626 온셋 이벤트. **61%가 동시타격**(kick+crash, snare+open HH 등).
  → "온셋 1개 = 악기 1개"는 recall 56%가 상한. 온셋마다 class를 **독립 판정(multi-label)** 해야 함.
- 5밴드 union 후보 검출로 이벤트 recall 95.2% 확보(후보 1484개). 후보를 넉넉히 뽑고
  class별 이진 분류기가 걸러내는 구조.
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt


SR = 44100
HOP = 256
N_FFT = 1024  # 23ms 창 — 드럼 트랜지언트 시간 분해능 우선

# 후보 검출용 대역(넓게 걸쳐 어떤 악기든 잡히게)
DETECT_BANDS = [
    (30, 130),      # kick
    (130, 600),     # tom / snare body
    (600, 2600),    # snare
    (2600, 8000),   # hihat / ride
    (8000, 18000),  # cymbal / triangle 고역
]

# 피처용 로그 간격 대역 16개
FEATURE_BANDS = [
    (20, 45), (45, 70), (70, 110), (110, 170), (170, 260), (260, 400),
    (400, 600), (600, 900), (900, 1400), (1400, 2100), (2100, 3200),
    (3200, 4800), (4800, 7000), (7000, 10000), (10000, 14000), (14000, 20000),
]

# 어택 이후 감쇠 궤적을 보는 시간 지점(프레임). hop 256 @44.1k = 5.805ms/frame
#
# ⚠ 180ms까지만 보던 것을 600ms로 확장(2026-07-24, 하이햇 불안정 대응).
# 실측 근거: 단독 온셋 기준 180ms 시점 고역 감쇠가 closed-HH -2.09 vs ride -0.72로
# **decay가 핵심 판별자(Cohen d=0.80)** 인데, 그 시점엔 ride가 아직 감쇠 중이라 분리가 덜 됨.
# ride/crash/open-HH는 500ms+ 울리므로 더 긴 지평이 있어야 "짧은 금속음(closed HH)"과 갈린다.
TIME_OFFSETS = [0, 2, 6, 14, 31, 60, 103]  # 0, 12, 35, 81, 180, 350, 600 ms

CLASSES = ["kick", "snare", "hihat_closed", "hihat_open", "tom", "crash", "ride",
           "triangle", "shaker"]


def _bandpass(y, sr, lo, hi):
    nyq = sr / 2.0
    sos = butter(4, [max(lo, 1) / nyq, min(hi, nyq * 0.999) / nyq],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, y)


def detect_candidates(y, sr=SR, delta=0.05, merge_ms=25.0):
    """5개 대역에서 독립적으로 온셋을 뽑아 합집합. 재현율 우선(정밀도는 분류기가 담당).

    반환: 후보 시각(초) ndarray (오름차순)
    """
    cands = []
    for lo, hi in DETECT_BANDS:
        yb = _bandpass(y, sr, lo, hi)
        oe = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=HOP)
        # ⚠ p99 정규화 실험 기각 기록 (2026-07-24):
        # 전역 max 정규화는 크래시 같은 큰 소리가 max를 지배하면 약한 하이햇을 delta
        # 아래로 깔아버린다(sample11 하이햇 후보 커버리지 86.9%). p99로 바꾸면 커버리지가
        # 98.0%로 회복되지만, **분류기가 그 약한 후보에 발화하지 않아 recall은 그대로**
        # (sample11 closed-HH recall 0.642→0.641)이고 후보만 늘어 FP가 증가
        # (LOSO 평균 0.7436→0.7322, s11 closed F1 0.755→0.709). 병목이 검출에서 분류로
        # 옮겨갔을 뿐이라 순손실 — max 유지. 재시도하려면 "약한 금속 타격"을 분류기가
        # 인식하도록 만드는 게 선행되어야 함.
        m = oe.max()
        if m <= 1e-9:
            continue
        oe = oe / m
        pk = librosa.util.peak_pick(
            oe, pre_max=5, post_max=5, pre_avg=17, post_avg=17,
            delta=delta, wait=8,
        )
        if len(pk):
            cands.append(librosa.frames_to_time(pk, sr=sr, hop_length=HOP))

    if not cands:
        return np.array([])

    allc = np.sort(np.concatenate(cands))
    merged = [allc[0]]
    gap = merge_ms / 1000.0
    for t in allc[1:]:
        if t - merged[-1] > gap:
            merged.append(t)

    # 어택 스냅 — 학습/추론 모두 같은 시각 기준을 쓰도록 여기서 적용.
    snapped = snap_to_attack(y, np.array(merged), sr)

    # 스냅이 인접 후보를 같은 어택으로 모을 수 있어 재중복 제거(중복 노트/겹침 방지).
    if len(snapped) == 0:
        return snapped
    dedup = [snapped[0]]
    for t in snapped[1:]:
        if t - dedup[-1] > 0.010:
            dedup.append(t)
    return np.array(dedup)


def snap_to_attack(y, times, sr=SR, max_move_ms=15.0, search_ms=22.0):
    """후보 시각을 파형의 물리적 어택 시점으로 스냅.

    실측(8곡): onset_strength 기반 후보는 실제 어택보다 **+4~9ms 늦게** 잡힌다
    (STFT 창/flux 특성). 작지만 일관된 지연이라 제거 가능.
    ⚠ 이웃 이벤트로 건너뛰지 않도록 이동량을 ±max_move_ms로 제한한다
    (bass mode의 온셋 스냅에서 얻은 교훈: 큰 이동은 다른 타격으로 점프한다).

    방법: t 주변 국소 진폭 피크를 찾고, 그 피크의 25% 교차점까지 역방향 탐색.
    """
    if len(times) == 0:
        return times

    k = max(1, int(0.001 * sr))
    kernel = np.ones(k) / k
    out = []
    for t in times:
        i0 = max(0, int((t - search_ms / 1000.0) * sr))
        i1 = min(len(y), int((t + search_ms / 1000.0) * sr))
        if i1 - i0 < 4 * k:
            out.append(t)
            continue
        env = np.convolve(np.abs(y[i0:i1]), kernel, mode="same")
        pk = int(env.argmax())
        thr = env[pk] * 0.25
        j = pk
        while j > 0 and env[j] > thr:
            j -= 1
        cand = (i0 + j) / sr
        if abs(cand - t) <= max_move_ms / 1000.0:
            out.append(cand)
        else:
            out.append(t)

    out = np.array(out)
    out.sort()
    return out


def _band_matrix(S, freqs):
    """STFT 파워 → 대역별 에너지 행렬 (n_bands, n_frames)."""
    out = np.zeros((len(FEATURE_BANDS), S.shape[1]), dtype=np.float32)
    for i, (lo, hi) in enumerate(FEATURE_BANDS):
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            out[i] = S[sel].sum(axis=0)
    return out


def feature_names():
    names = []
    for i in range(len(FEATURE_BANDS)):
        for t in TIME_OFFSETS:
            names.append(f"b{i}_t{t}")
    # 다중 지평 감쇠 — 짧은 금속음(closed HH) vs 지속 금속음(ride/crash/open HH) 판별의 핵심.
    for i in range(len(FEATURE_BANDS)):
        names += [f"b{i}_decay180", f"b{i}_decay350", f"b{i}_decay600"]
    for tag in ["t0", "t6", "t31", "t103"]:
        names += [f"centroid_{tag}", f"rolloff_{tag}"]
    names += ["total_t0", "total_decay180", "total_decay600", "attack_rise",
              "sustain_hf_ratio"]
    # 시간 문맥 — 하이햇/셰이커는 박자를 지키는 악기라 규칙적 패턴을 이룬다.
    # 온셋을 독립 판정만 하면 이 정보를 통째로 버리게 됨(bass의 neighbor_support와 같은 발상).
    names += ["ioi_prev", "ioi_next", "ioi_regularity", "local_density",
              "spec_sim_prev", "spec_sim_next"]
    return names


def extract_features(y, times, sr=SR):
    """각 후보 시각에 대해 피처 벡터 추출.

    피처 구성:
    - 16대역 × 5시점 로그에너지 (어택 + 감쇠 궤적) → 악기별 스펙트럼/감쇠 특성
    - 대역별 감쇠비(180ms/0ms) → closed HH(빠름) vs ride/crash(느림) 구분
    - centroid/rolloff/flatness 3시점 → 음색(토널 vs 노이즈): triangle/ride vs crash/snare
    - 전체 에너지/감쇠/어택상승
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    B = _band_matrix(S, freqs)
    n_frames = S.shape[1]

    eps = 1e-10
    logB = np.log10(B + eps)
    total = B.sum(axis=0)
    logtotal = np.log10(total + eps)

    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    # spectral_flatness는 제외 — 실측 결과 closed-HH vs ride 분리도 Cohen d=0.03으로
    # 사실상 정보가 없었음(값이 ~0.0001에 몰림). 피처 낭비라 제거.

    # 고역(5k~16k) 합 — 지속 금속음 판별용 보조 신호
    hf_idx = [i for i, (lo, hi) in enumerate(FEATURE_BANDS) if lo >= 4800]
    hf = logB[hf_idx].mean(axis=0) if hf_idx else logtotal

    def clamp(i):
        return min(max(i, 0), n_frames - 1)

    feats = []
    for t in times:
        f0 = int(round(t * sr / HOP))
        row = []

        for i in range(len(FEATURE_BANDS)):
            for off in TIME_OFFSETS:
                row.append(logB[i, clamp(f0 + off)])

        # 다중 지평 감쇠(180/350/600ms) — 핵심 판별자.
        a = clamp(f0)
        for i in range(len(FEATURE_BANDS)):
            row += [
                logB[i, clamp(f0 + 31)] - logB[i, a],
                logB[i, clamp(f0 + 60)] - logB[i, a],
                logB[i, clamp(f0 + 103)] - logB[i, a],
            ]

        for off in [0, 6, 31, 103]:
            fi = clamp(f0 + off)
            row += [cent[fi], roll[fi]]

        pre = clamp(f0 - 2)
        row += [
            logtotal[a],
            logtotal[clamp(f0 + 31)] - logtotal[a],
            logtotal[clamp(f0 + 103)] - logtotal[a],
            logtotal[a] - logtotal[pre],
            hf[clamp(f0 + 60)] - hf[a],  # 고역이 350ms 뒤까지 얼마나 남는가
        ]

        feats.append(row)

    feats = np.array(feats, dtype=np.float32)
    if len(feats):
        feats = np.hstack([feats, _context_features(feats, times, logB)])
    return feats


def _context_features(base, times, logB):
    """시간 문맥 피처 — 규칙적 타임키핑 패턴에 속하는 타격인지.

    스퓨리어스 하이햇은 그리드를 벗어나고 이웃과 음색이 다르다. 반대로 진짜 하이햇은
    앞뒤 이웃과 간격이 규칙적이고 스펙트럼이 유사하다.
    ⚠ 그리드를 **강제하지 않는다** — 분류기가 참고할 증거로만 제공(CLAUDE.md 교훈:
    bass에서 run 단위 강제 통일은 정당한 변화까지 뭉개서 실패했음).
    """
    n = len(times)
    out = np.zeros((n, 6), dtype=np.float32)
    if n < 2:
        return out

    # 고역 프로파일(음색 유사도용): 마지막 6개 대역의 어택 시점 로그에너지
    nb, nt = len(FEATURE_BANDS), len(TIME_OFFSETS)
    prof = np.stack([base[:, i * nt + 0] for i in range(nb - 6, nb)], axis=1)
    prof = prof - prof.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(prof, axis=1) + 1e-9

    for i in range(n):
        ip = times[i] - times[i - 1] if i > 0 else np.nan
        inx = times[i + 1] - times[i] if i < n - 1 else np.nan
        out[i, 0] = np.log10(ip) if ip == ip and ip > 0 else -2.0
        out[i, 1] = np.log10(inx) if inx == inx and inx > 0 else -2.0
        if ip == ip and inx == inx and (ip + inx) > 0:
            out[i, 2] = abs(ip - inx) / ((ip + inx) / 2.0)  # 0에 가까울수록 규칙적
        else:
            out[i, 2] = 1.0
        lo = np.searchsorted(times, times[i] - 1.0)
        hi = np.searchsorted(times, times[i] + 1.0)
        out[i, 3] = hi - lo
        if i > 0:
            out[i, 4] = float(prof[i] @ prof[i - 1] / (norm[i] * norm[i - 1]))
        if i < n - 1:
            out[i, 5] = float(prof[i] @ prof[i + 1] / (norm[i] * norm[i + 1]))

    return out


def attack_energy(y, times, sr=SR):
    """각 온셋의 어택 RMS 피크 — velocity 매핑용."""
    out = []
    win = int(0.020 * sr)
    for t in times:
        i = int(t * sr)
        seg = y[max(0, i - int(0.005 * sr)): i + win]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    return np.array(out)
