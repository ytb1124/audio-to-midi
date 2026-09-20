"""E-GMD 피처 추출 — 90GB zip에서 스트리밍, 디스크 쓰기 0.

설계 근거(2026-07-26 실측):
- E-GMD는 **고유 연주 1,059개 × 43킷**이다(파일명 끝 숫자는 take가 아니라 킷 인덱스).
  고유 연주 내용은 10.3시간. 모든 연주가 43킷 전부에 존재한다.
- 이 구조 덕분에 **총량을 고정한 채 킷 수만 바꾸는 통제 실험**이 가능하다.
  킷을 43→7로 줄이면서 데이터량도 같이 줄면 성능 하락이 킷 수 때문인지
  데이터량 때문인지 구분이 안 된다. 각 연주를 **정확히 한 킷에** 배정하면
  어떤 설정에서도 총량이 10.3시간으로 같다.

      설정  킷 수   총량      킷당
      A      1    10.3h    10.3h
      C      7    10.3h     1.5h   ← 우리 현재 상태와 직접 비교 지점
      E     43    10.3h    14.4분

- 배정은 라운드로빈이 아니라 **층화**다. 연주마다 심벌 함량이 크게 다르므로
  단순 i%K면 ride/crash가 0인 킷이 생긴다. 심벌 함량 순으로 정렬한 뒤 딜링해서
  각 킷이 고르게 받도록 한다.

라벨 구성은 `train_drum_classifier.build_song`과 **동일**하다(detect_candidates로
후보를 뽑고 GT 온셋과 MATCH_TOL 내 매칭으로 multi-label). 피처/라벨 구성이
어긋나면 조용히 이상한 모델이 나오므로 재구현하지 않고 같은 함수를 쓴다.

사용법:
    python eval/build_egmd_features.py --kits 43 --out eval/_egmd_cache/k43
    python eval/build_egmd_features.py --kits 7  --no-edge-map --out eval/_egmd_cache/k7_noedge
"""

import argparse
import collections
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "scripts"))

from drum_features import SR, detect_candidates, extract_features  # noqa: E402
from score_drums import CLASS_MAP, CLASS_ORDER  # noqa: E402

MATCH_TOL = 0.05

EGMD_ROOT = Path(os.environ.get("EGMD_ROOT", "/Volumes/TAEBIN/egmd"))
ZIP_PATH = EGMD_ROOT / "e-gmd-v1.0.0.zip"
META_PATH = EGMD_ROOT / "egmd_per_file.json"
ZIP_PREFIX = "e-gmd-v1.0.0/"

# TD-17 하이햇 엣지 존 추정 pitch. 오디오 검증 결과 42/46과 음색이 실제로 다르지만
# (킷 홀드아웃 AUC 0.876 / 0.952, 널 0.499), 우리 9-class는 타격 위치를 구분하지
# 않으므로 합친다 — 이미 42(Bow)와 44(Pedal)를 합치고 있는 것과 같은 판단이다.
EDGE_MAP = {22: "hihat_closed", 26: "hihat_open"}

# ★ 미매핑 타격 근처의 "라벨 없는" 후보를 버리는 창(초).
#
# sample10 교훈의 근본 처리다. 미매핑 타악기는 오디오에 실재하므로 detect_candidates가
# 잡아내는데 GT에는 없어서 9-class 전부에 음성으로 라벨된다 → "진짜 온셋을 거부하도록"
# 가르치는 라벨 노이즈. 파일 단위로 걸러내면(구 MAX_RESIDUAL_RATIO 방침) 한도 안의
# 오염은 그대로 남는다.
#
# 그리고 이 오염은 무작위가 아니다: pitch 58이 톰 계열로 판명됐으므로(최근접 클래스
# tom 36/36, decay600 −1.141) **톰 비슷한 온셋이 "톰 아님"으로 학습된다.** 톰은 우리
# GT에 268노트뿐이고 sample6에서 가짜 톰 118개 문제가 있었던 이미 불안정한 클래스라,
# 통제되지 않은 채 특정 클래스만 겨냥하는 오염이다.
#
# 처방: MIDI에 미매핑 타격의 정확한 시각이 있으므로, 그 근처의 후보 중 **어떤 클래스에도
# 양성이 아닌 것**만 학습에서 제외한다(양성도 음성도 아닌 것으로 취급). 유효한 양성은
# 그대로 두므로 미매핑과 동시 타격된 킥/스네어를 잃지 않는다. 부분 라벨 데이터의 표준 처리.
UNMAPPED_MASK_SEC = 0.05

# ★ 피처와 라벨을 분리해 저장한다(2026-07-26).
#
# EDGE_MAP은 **라벨링 결정**이지 피처와 무관하다 — 같은 후보의 179차원 피처는
# 매핑 유무와 관계없이 동일하다. 그런데 라벨을 확정해서 저장하면 매핑을 바꿀 때마다
# 30~60분짜리 재추출이 필요해진다. 앞으로 나올 라벨 결정이 여럿이므로
# (22/26 처리, exotic 재분류, 마스킹 창 T, 잔여 pitch 처리) 지금 분리한다.
#
# 저장 형식: 후보별 **(pitch, delta_ms) 이벤트 리스트**를 flat 배열 3개로.
#   ev_cand[k]  = k번째 이벤트가 속한 후보 인덱스
#   ev_pitch[k] = GT pitch (PITCH_VOCAB 인덱스가 아니라 실제 pitch)
#   ev_dt[k]    = GT시각 − 후보시각 (ms)
#
# ⚠ 멀티핫으로 저장하면 안 된다 — 멀티핫은 이미 특정 창(50ms)으로 매칭한 결과라
# 창을 좁히거나 넓히는 걸 되돌릴 수 없다. 향후 라벨 실험 목록에 "마스킹 창 T"가
# 있으므로 delta를 그대로 남긴다. 매칭은 넉넉하게 EVENT_WINDOW로 잡고, 라벨 생성
# 시점에 T로 필터링한다 → T ≤ EVENT_WINDOW인 모든 값이 재라벨링으로 처리된다.
# 후보당 평균 이벤트가 1~3개라 용량은 무시할 수준이다.
EVENT_WINDOW = 0.150
PITCH_VOCAB = [22, 26, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49,
               50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
PITCH_IDX = {p: i for i, p in enumerate(PITCH_VOCAB)}

# 우리 taxonomy와 근본적으로 안 맞는 킷(탬버린이 하이햇 역할 등) — 잔여의 80%가 여기 몰려 있다.
EXCLUDE_KITS = {"Compact Lite (w/ Tambourine HH)", "Dark Hybrid"}


def labels_from_events(n_cand, ev_cand, ev_pitch, ev_dt, tol_ms=50.0,
                       edge_map=True, exclude=(), unmapped_tol_ms=None):
    """(pitch, delta) 이벤트 → 9-class 멀티라벨. **재추출 없이 매핑도 창도 바꾸는 지점.**

    tol_ms: 라벨 매칭 창. unmapped_tol_ms: 미매핑 근접 판정 창(기본은 tol_ms와 동일).
    반환: (Y, unmapped_hit)
    """
    mapping = dict(CLASS_MAP)
    if edge_map:
        mapping.update(EDGE_MAP)
    for k in exclude:
        mapping.pop(k, None)
    if unmapped_tol_ms is None:
        unmapped_tol_ms = tol_ms

    Y = np.zeros((n_cand, len(CLASS_ORDER)), dtype=np.int8)
    unmapped_hit = np.zeros(n_cand, dtype=bool)
    if len(ev_cand):
        adt = np.abs(ev_dt)
        for pitch in np.unique(ev_pitch):
            c = mapping.get(int(pitch))
            sel = ev_pitch == pitch
            if c is None:
                m = sel & (adt <= unmapped_tol_ms)
                unmapped_hit[ev_cand[m]] = True
            else:
                m = sel & (adt <= tol_ms)
                Y[ev_cand[m], CLASS_ORDER.index(c)] = 1
    return Y, unmapped_hit


def apply_mask(Y, unmapped_hit):
    """미매핑 근처의 '라벨 없는' 후보만 제외하는 keep 마스크. 양성이 붙은 후보는 유지."""
    return ~(unmapped_hit & (Y.sum(axis=1) == 0))


def perf_key(rec):
    """연주 식별자. 파일명 끝의 `_<킷인덱스>`를 떼면 같은 연주가 묶인다."""
    base = os.path.basename(rec["midi"]).replace(".midi", "")
    return (rec["drummer"], rec["midi"].split("/")[1], re.sub(r"_\d+$", "", base))


def load_meta():
    if not META_PATH.exists():
        raise SystemExit(
            f"메타데이터가 없습니다: {META_PATH}\n"
            "EGMD_ROOT를 확인하거나 파일 단위 집계를 먼저 생성하세요."
        )
    return json.loads(META_PATH.read_text())


def choose_assignment(meta, n_kits, seed=0):
    """연주 → 킷 배정. 총량 고정, 킷 다양성만 변화.

    반환: {kit_name: [record, ...]}  (각 연주는 정확히 한 번만 등장)
    """
    by_perf = collections.defaultdict(dict)
    for r in meta:
        by_perf[perf_key(r)][r["kit"]] = r

    kits = sorted({r["kit"] for r in meta} - EXCLUDE_KITS)
    rng = np.random.RandomState(seed)
    kits = [kits[i] for i in rng.permutation(len(kits))][:n_kits]

    # 파일 단위 잔여 필터는 제거했다 — 후보 단위 마스킹(UNMAPPED_MASK_SEC)이
    # 오염을 정확히 잘라내므로 파일을 통째로 버릴 이유가 없다. 잔여 포함 파일이
    # 시간 기준 48.2%였으므로 이 변경으로 그만큼을 되찾는다.
    def usable(r):
        return sum(r[c] for c in CLASS_ORDER if c in r) + r["p22"] + r["p26"] > 0

    # 심벌(ride+crash) 함량 순으로 정렬 후 딜링 — 각 킷이 심벌 풍부한 연주를 고르게 받는다.
    perfs = []
    for key, kitmap in by_perf.items():
        avail = [k for k in kits if k in kitmap and usable(kitmap[k])]
        if not avail:
            continue
        any_rec = kitmap[avail[0]]
        perfs.append((any_rec["ride"] + any_rec["crash"], key, kitmap, avail))
    perfs.sort(key=lambda x: -x[0])

    out = collections.defaultdict(list)
    for i, (_, key, kitmap, avail) in enumerate(perfs):
        kit = avail[i % len(avail)]
        out[kit].append(kitmap[kit])
    return dict(out), kits


def build_one(zf, rec, edge_map: bool):
    """한 파일의 (times, X, Y). train_drum_classifier.build_song과 같은 구성."""
    raw = zf.read(ZIP_PREFIX + rec["audio"])
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < SR:
        return None

    times = detect_candidates(y, SR)
    if len(times) == 0:
        return None
    X = extract_features(y, times, SR)

    pm = pretty_midi.PrettyMIDI(str(EGMD_ROOT / "e-gmd-v1.0.0" / rec["midi"]))
    by_pitch = collections.defaultdict(list)
    for ins in pm.instruments:
        for n in ins.notes:
            by_pitch[n.pitch].append(n.start)

    # 후보별 (pitch, delta) 이벤트 — 라벨도 창도 확정하지 않고 원자료로 남긴다.
    ec, ep, ed = [], [], []
    for pitch, ts in by_pitch.items():
        ts = np.array(sorted(ts))
        lo = np.searchsorted(ts, times - EVENT_WINDOW)
        hi = np.searchsorted(ts, times + EVENT_WINDOW)
        for i in np.nonzero(hi > lo)[0]:
            for t_gt in ts[lo[i]:hi[i]]:
                ec.append(i); ep.append(pitch); ed.append((t_gt - times[i]) * 1000.0)
    return (times, X,
            np.asarray(ec, dtype=np.int32),
            np.asarray(ep, dtype=np.int16),
            np.asarray(ed, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description="E-GMD 피처 추출(zip 스트리밍)")
    ap.add_argument("--kits", type=int, default=43)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="디버그용: 연주 수 제한")
    ap.add_argument("--no-edge-map", action="store_true",
                    help="pitch 22/26을 하이햇에 매핑하지 않음(대조군)")
    args = ap.parse_args()

    edge_map = not args.no_edge_map
    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "eval" / "_egmd_cache" / f"k{args.kits}{'' if edge_map else '_noedge'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta()
    assign, kits = choose_assignment(meta, args.kits, args.seed)
    total_perf = sum(len(v) for v in assign.values())
    total_dur = sum(r["dur"] for v in assign.values() for r in v)
    print(f"킷 {len(assign)}개 / 연주 {total_perf:,}개 / {total_dur/3600:.2f}시간 "
          f"(edge_map={edge_map})", flush=True)

    zf = zipfile.ZipFile(ZIP_PATH)
    done = 0
    masked = 0
    counts = collections.Counter()
    for kit, recs in sorted(assign.items()):
        if args.limit:
            recs = recs[: args.limit]
        T, XS, EC, EP, ED = [], [], [], [], []
        base_n = 0
        for r in recs:
            try:
                got = build_one(zf, r, edge_map)
            except Exception as e:
                print(f"  [skip] {r['audio']}: {e}", file=sys.stderr)
                continue
            if got is None:
                continue
            t, X, ec, ep, ed = got
            if len(ec):
                ec = ec + base_n
            T.append(t); XS.append(X)
            EC.append(ec); EP.append(ep); ED.append(ed)
            base_n += len(t)
            done += 1
        if not XS:
            continue
        X = np.vstack(XS)
        ec = np.concatenate(EC) if EC else np.zeros(0, np.int32)
        ep = np.concatenate(EP) if EP else np.zeros(0, np.int16)
        ed = np.concatenate(ED) if ED else np.zeros(0, np.float32)
        Y, un = labels_from_events(len(X), ec, ep, ed)
        keep = apply_mask(Y, un)
        masked += int((~keep).sum())
        for ci, c in enumerate(CLASS_ORDER):
            counts[c] += int(Y[keep][:, ci].sum())
        np.savez_compressed(out_dir / f"{kit.replace('/', '_')}.npz",
                            X=X, ev_cand=ec, ev_pitch=ep, ev_dt=ed,
                            event_window_ms=EVENT_WINDOW * 1000,
                            kit=kit, n_perf=len(XS))
        print(f"  {kit[:34]:36s} 연주 {len(XS):4d}  후보 {len(X):7,d}", flush=True)

    print(f"\n완료: 연주 {done:,} → {out_dir}")
    print(f"미매핑 근처 라벨없는 후보 제외: {masked:,}개")
    print("최종 클래스별 양성 라벨 수")
    tot = sum(counts.values())
    for c in CLASS_ORDER:
        if counts[c]:
            print(f"  {c:16s}{counts[c]:9,d}  ({counts[c]/tot:5.1%})")
    if counts["ride"]:
        print(f"\n  closed-HH : ride = {counts['hihat_closed']/counts['ride']:.2f} : 1")
    (out_dir / "_summary.json").write_text(json.dumps(
        {"kits": len(assign), "perfs": done, "hours": total_dur / 3600,
         "edge_map": edge_map, "masked_candidates": masked,
         "unmapped_mask_sec": UNMAPPED_MASK_SEC,
         "counts": dict(counts)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
