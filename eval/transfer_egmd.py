"""전이 검증 — E-GMD로 학습한 모델이 우리 실제 녹음에 통하는가?

★ 이 하니스의 존재 이유: **E-GMD로 학습하고 E-GMD 홀드아웃으로 평가하면 아무것도
증명하지 못한다.** 질문은 "E-GMD 안에서 잘하는가"가 아니라 "실제 음악에 통하는가"다.
따라서 평가는 **항상 우리 7킷**으로 한다. E-GMD는 학습에만 쓰이므로 우리 7킷은
전부 out-of-sample이다.

같은 이유로 킷 수 곡선을 그릴 때도 y축은 우리 7킷 성능이지 E-GMD 내부 성능이 아니다.

판정은 ride 단독이 아니라 **전 클래스 비교표**로 한다. E-GMD가 ride를 살리면서
kick/snare/hihat을 망가뜨리면 순손실이다 — TD-17 전자킷 음색이 우리 어쿠스틱 킷과
다르므로 충분히 가능한 시나리오다.

기준선은 현행 7킷 LOSO(우리가 지금 배포 중인 모델의 정직한 성능)다.

사용법:
    python eval/transfer_egmd.py --egmd eval/_egmd_cache/k41
    python eval/transfer_egmd.py --egmd eval/_egmd_cache/k41 --no-context
    python eval/transfer_egmd.py --egmd eval/_egmd_cache/k41 --mix   # E-GMD + 우리 6킷
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "scripts"))

import train_drum_classifier as tdc  # noqa: E402
from drum_features import feature_names  # noqa: E402
from score_drums import CLASS_ORDER  # noqa: E402

OUR_SONGS = [f"drums_sample{i}" for i in (4, 5, 6, 7, 8, 9, 11)]

# v5에서 추가된 시간 문맥 피처. E-GMD는 고유 연주가 1,059개뿐이라 리듬 어휘가 좁고
# 특정 드러머·장르에 치우쳐 있을 수 있다 — "라이드는 이런 리듬 자리에 온다"를 외우면
# 우리 데이터에도 사용자 음악에도 전이되지 않는다. 22 vs 42 검증에서 시간 문맥만으로
# AUC 0.893이 나온 것이 근거. 어블레이션으로 확인한다.
CONTEXT_KEYS = ("ioi", "density", "spec_sim", "regular")


def context_idx():
    names = feature_names()
    return [i for i, n in enumerate(names) if any(k in n for k in CONTEXT_KEYS)]


def load_egmd(cache_dir: Path, kits=None):
    """킷별 npz를 읽어 (X, Y, kit_names). kits를 주면 그 킷만."""
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"E-GMD 피처가 없습니다: {cache_dir}")
    XS, YS, used = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        kit = str(d["kit"])
        if kits is not None and kit not in kits:
            continue
        XS.append(d["X"]); YS.append(d["Y"]); used.append(kit)
    if not XS:
        raise SystemExit("선택된 킷이 없습니다.")
    return np.vstack(XS), np.vstack(YS), used


def score_song(models, song, data, cols, threshold=0.5):
    """한 곡을 class별 F1로 채점. tdc.loso와 같은 후보 단위 채점."""
    times, X, Y, gt = data[song]
    X = X[:, cols]
    out = {}
    for ci, c in enumerate(CLASS_ORDER):
        if c not in models:
            out[c] = None
            continue
        p = models[c].predict_proba(X)[:, 1] >= threshold
        tp = int((p & (Y[:, ci] == 1)).sum())
        fp = int((p & (Y[:, ci] == 0)).sum())
        fn = int((~p & (Y[:, ci] == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "p": prec, "r": rec, "n_ref": int((Y[:, ci] == 1).sum()),
        }
    return out


def run_baseline(ours, cols, threshold):
    """현행 기준선: 우리 6킷으로 학습(LOSO) → 남은 1킷 평가. E-GMD 미사용.

    ★ 비교는 반드시 같은 채점 방식이어야 한다. CLAUDE.md의 LOSO 0.764는 MIDI 수준
    채점이라 이 하니스의 후보 수준 F1과 직접 비교하면 안 된다 — 그래서 기준선을
    여기서 같은 코드로 다시 계산한다.
    """
    out = {}
    for song in OUR_SONGS:
        tr = [s for s in OUR_SONGS if s != song]
        Xtr = np.vstack([ours[s][1] for s in tr])[:, cols]
        Ytr = np.vstack([ours[s][2] for s in tr])
        models = {}
        for ci, c in enumerate(CLASS_ORDER):
            if Ytr[:, ci].sum() < 5:
                continue
            models[c] = tdc.fit_class(Xtr, Ytr[:, ci])
        out[song] = score_song(models, song, ours, cols, threshold)
    return out


def class_mean(results, c):
    vals = [results[s][c] for s in OUR_SONGS
            if results[s].get(c) and results[s][c]["n_ref"] > 0]
    if not vals:
        return None, 0
    return float(np.mean([v["f1"] for v in vals])), sum(v["n_ref"] for v in vals)


def main():
    ap = argparse.ArgumentParser(description="E-GMD → 우리 7킷 전이 검증")
    ap.add_argument("--egmd", default="eval/_egmd_cache/k41")
    ap.add_argument("--no-context", action="store_true",
                    help="시간 문맥 피처 6종 제외(E-GMD 과적합 통로 어블레이션)")
    ap.add_argument("--mix", action="store_true",
                    help="E-GMD + 우리 6킷 혼합 학습(평가 곡은 항상 제외)")
    ap.add_argument("--kits", default=None, help="쉼표 구분 킷 이름(부분집합 실험)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="현행 7킷 LOSO 기준선을 같은 채점 방식으로 함께 계산해 비교표 출력")
    args = ap.parse_args()

    names = feature_names()
    ctx = set(context_idx())
    cols = [i for i in range(len(names)) if not (args.no_context and i in ctx)]

    kits = set(args.kits.split(",")) if args.kits else None
    Xe, Ye, used_kits = load_egmd(REPO_ROOT / args.egmd, kits)
    print(f"E-GMD 학습: 킷 {len(used_kits)}개, 후보 {len(Xe):,}개, "
          f"피처 {len(cols)}차원{' (시간문맥 제외)' if args.no_context else ''}")

    print("우리 7킷 로드(평가 전용)...")
    ours = tdc.load_all(OUR_SONGS)

    results = {}
    for song in OUR_SONGS:
        Xtr, Ytr = Xe, Ye
        if args.mix:
            add = [song2 for song2 in OUR_SONGS if song2 != song]
            Xtr = np.vstack([Xe] + [ours[s][1] for s in add])
            Ytr = np.vstack([Ye] + [ours[s][2] for s in add])
        Xtr = Xtr[:, cols]

        models = {}
        for ci, c in enumerate(CLASS_ORDER):
            if Ytr[:, ci].sum() < 5:
                continue
            models[c] = tdc.fit_class(Xtr, Ytr[:, ci])
        results[song] = score_song(models, song, ours, cols, args.threshold)
        f = {c: (v["f1"] if v else None) for c, v in results[song].items()}
        print(f"  {song:16s} ride {f.get('ride') or 0:.3f}  crash {f.get('crash') or 0:.3f}")

    base = run_baseline(ours, cols, args.threshold) if args.baseline else None

    # ---- 비교표: 전 클래스, 현행 기준선과 나란히 -------------------------------
    # ★ go/no-go는 ride 단독이 아니라 이 표 전체로 판단한다. E-GMD가 ride를 살리면서
    #   kick/snare/hihat을 망가뜨리면 순손실이다.
    print()
    print("=" * 92)
    print("E-GMD 학습 → 우리 7킷 평가 (전부 out-of-sample). 채점 방식 동일.")
    print("=" * 92)
    head = f"{'class':16s}{'GT노트':>8s}{'E-GMD F1':>10s}{'P':>7s}{'R':>7s}"
    if base:
        head += f"{'현행LOSO':>10s}{'차이':>9s}"
    print(head + "   곡별 F1(n)")
    if base:
        print("  ※ '현행LOSO' = 우리 6킷 학습 → 남은 1킷 평가(in-sample 아님). "
              "E-GMD 열과 동일한 후보 수준 채점 코드.")
    print("-" * 92)
    agg = {}
    for c in CLASS_ORDER:
        vals = [results[s][c] for s in OUR_SONGS if results[s].get(c) and results[s][c]["n_ref"] > 0]
        if not vals:
            continue
        f1 = float(np.mean([v["f1"] for v in vals]))
        agg[c] = f1
        line = (f"{c:16s}{sum(v['n_ref'] for v in vals):8d}{f1:10.3f}"
                f"{np.mean([v['p'] for v in vals]):7.3f}{np.mean([v['r'] for v in vals]):7.3f}")
        if base:
            b, _ = class_mean(base, c)
            line += f"{(b if b is not None else 0):10.3f}{f1-(b or 0):+9.3f}"
        print(line + "   " + " ".join(f"{v['f1']:.2f}({v['n_ref']})" for v in vals))

    # ★ ride는 s4/s8 두 킷에만 있어 풀링하면 한 킷만 되는 경우가 가려진다 — 분리 보고.
    print("-" * 86)
    for c in ("ride", "crash"):
        detail = [(s, results[s][c]) for s in OUR_SONGS
                  if results[s].get(c) and results[s][c]["n_ref"] > 0]
        if detail:
            print(f"{c} 곡별 상세: " + "  ".join(
                f"{s.replace('drums_sample','s')} F1 {v['f1']:.3f}(n={v['n_ref']})" for s, v in detail))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"egmd": args.egmd, "no_context": args.no_context, "mix": args.mix,
             "kits": used_kits, "per_song": results, "class_mean": agg},
            indent=2, ensure_ascii=False, default=float))
        print(f"\n저장: {args.json_out}")


if __name__ == "__main__":
    main()
