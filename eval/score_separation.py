"""스템 분리 3자 비교 하니스 — 분리가 downstream note F1에 실제로 도움이 되는가?

배경(핸드오프 문서 B-0): 스템 분리 품질을 **SDR로 판단하면 안 된다.** 우리 최종
산출물은 MIDI라서, 분리 SDR이 좋아져도 전사가 나빠질 수 있고(위상 아티팩트,
트랜지언트 뭉개짐) 그 반대도 가능하다. 판단 기준은 언제나 **전사 note F1**이다.

측정하는 3경로:

    A) mix      → 전사 → note F1    분리 없는 baseline
    B) sep stem → 전사 → note F1    실제 파이프라인 (Demucs 등)
    C) gt stem  → 전사 → note F1    상한선

세 번째가 핵심이다. **GT 스템으로도 F1이 낮으면 병목은 분리가 아니라 전사이고,
분리 모델을 아무리 바꿔도 소용없다.** 반대로 B와 C의 갭이 크면 그때 분리에
투자한다 — 어디에 돈을 써야 하는지를 첫날에 알려주는 측정.

채점 로직은 새로 만들지 않고 기존 하니스를 그대로 재사용한다(얇은 래퍼):
    - 음정 악기(bass 등): `eval/score.py`의 mir_eval note F1
    - 드럼:               `eval/score_drums.py`의 class별 onset F1(MICRO)

★ 집계 표준 (하나로 고정 — 여러 값이 유통되면 나중에 회귀로 오판한다):
    **정식**      곡별 micro F1을 낸 뒤 곡 평균. 드럼 LOSO가 쓰는 방식과 동일하고,
                 이 스크립트가 헤드라인으로 출력하는 값이다.
    diagnostic   전 곡 풀링 micro / 클래스별 F1 / 실제 매칭 노트 수. 원인 분석용으로만
                 쓰고, 인용할 때는 반드시 집계 방식을 함께 적을 것.

⚠ 경로 C를 "상한선"이라고 부를 때 주의: 채점에 쓰는 모델이 그 곡으로 학습된
   배포 모델이면 A/B/C 전부 **in-sample**이고, C는 진짜 상한이 아니라 낙관적인
   값이다(드럼 배포 모델 `drum_class_model.joblib`의 `trained_on`을 확인할 것).
   처음 보는 킷에 대한 정직한 값이 필요하면 LOSO 모델로 채점해야 한다.

사용법:
    python eval/score_separation.py --manifest eval/separation_manifest.json
    python eval/score_separation.py --manifest ... --paths A,B --songs drums_sample4,drums_sample6
    python eval/score_separation.py --manifest ... --model htdemucs_6s

매니페스트 형식(JSON):
    {"songs": [
      {"name": "drums_sample4",
       "instrument": "drums",                       # drums | bass
       "mix":     "eval/ground_truth/drums_sample4.wav",
       "gt_midi": "eval/ground_truth/drums_sample4.mid",
       "gt_stem": null,                             # 있으면 경로 C 측정
       "model": null}                               # LOSO 평가용 모델(없으면 배포 모델)
    ]}

경로별 입력이 없으면(예: gt_stem 미보유) 그 경로는 조용히 건너뛰고 나머지만
보고한다 — 부분 데이터로도 즉시 쓸 수 있게 하기 위함.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))

import numpy as np  # noqa: E402

import score as bass_score  # noqa: E402  (eval/score.py)
import score_drums  # noqa: E402  (eval/score_drums.py)


# 악기 → (전사 스크립트, Demucs 스템 이름)
# 스템 이름은 4-stem(htdemucs) 기준. 6-stem 모델은 guitar/piano가 추가된다.
INSTRUMENTS = {
    "drums": {
        "script": "backend/scripts/drum_audio_to_midi.py",
        "stem": "drums",
        "metric": "drums",
    },
    "bass": {
        "script": "backend/scripts/bass_basicpitch_audio_hybrid.py",
        "stem": "bass",
        "metric": "pitched",
    },
}

PATH_LABELS = {
    "A": "mix (분리 없음)",
    "B": "sep stem (실제 파이프라인)",
    "C": "gt stem (상한선)",
}


# ---------------------------------------------------------------- 전사 / 분리


def transcribe(instrument: str, audio: Path, out_midi: Path, python_exe: str,
               model_path: Path = None) -> None:
    """악기별 배포 파이프라인을 그대로 호출한다(하니스가 전사 로직을 재구현하지 않음).

    model_path를 주면 그 모델로 채점한다 — LOSO 평가용. 배포 모델은 정답 곡 전부로
    학습돼 있어 그대로 쓰면 A/B/C 세 경로가 전부 in-sample이 된다.
    """
    script = REPO_ROOT / INSTRUMENTS[instrument]["script"]
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, str(script), str(audio), str(out_midi)]
    if model_path is not None:
        cmd += ["--model", str(model_path)]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def separate(mix: Path, cache_dir: Path, model: str, device: str, python_exe: str) -> dict:
    """Demucs 분리. 이미 캐시에 있으면 재실행하지 않는다(분리가 가장 비싼 단계)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem: p for p in cache_dir.glob("*.wav")}
    if existing:
        return existing

    script = REPO_ROOT / "backend/scripts/demucs_separate.py"
    cmd = [
        python_exe, str(script), str(mix), str(cache_dir),
        "--model", model, "--device", device,
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return {p.stem: p for p in cache_dir.glob("*.wav")}


# ---------------------------------------------------------------- 채점 (재사용)


def score_pitched(est_midi: Path, ref_midi: Path) -> dict:
    """eval/score.py 재사용. score.py가 '실질적 채점 기준'으로 삼는 것과 같은 방식으로
    전역 옥타브 시프트 / 전역 시간 오프셋을 보정한 뒤의 F1을 함께 낸다."""
    ref = bass_score.load_notes(ref_midi)
    est = bass_score.load_notes(est_midi)

    p, r, f1, _ = bass_score.mir_eval_metrics(ref, est)

    shift, _, _ = bass_score.detect_global_octave_shift(ref, est)
    lag = bass_score.detect_global_time_offset(ref, est)

    ref_adj = ref
    if shift is not None:
        ref_adj = bass_score.shift_notes(ref_adj, shift)
    if abs(lag) > 0.015:
        ref_adj = bass_score.time_shift_notes(ref_adj, lag)

    if ref_adj is ref:
        p_adj, r_adj, f1_adj = p, r, f1
    else:
        p_adj, r_adj, f1_adj, _ = bass_score.mir_eval_metrics(ref_adj, est)

    diag, extra = bass_score.greedy_octave_diagnostic(ref_adj, est)
    counts = {"correct": 0, "octave_error": 0, "other_error": 0, "missed": 0}
    for category, *_ in diag:
        counts[category] += 1

    return {
        "f1": f1, "precision": p, "recall": r,
        "f1_adjusted": f1_adj, "precision_adjusted": p_adj, "recall_adjusted": r_adj,
        "global_octave_shift": shift, "global_time_lag_ms": round(lag * 1000, 1),
        "n_est": len(est), "n_ref": len(ref),
        "missed": counts["missed"], "extra": len(extra),
    }


def score_drums_micro(est_midi: Path, ref_midi: Path, tol: float = 0.05) -> dict:
    """eval/score_drums.py 재사용. 헤드라인 지표는 MICRO F1, 참고로 class 무시한
    ONSET-ONLY(검출력만)도 함께 낸다 — 분리가 검출을 돕는지 분류를 돕는지 분리해서 보기 위함."""
    est_by, _ = score_drums.load_by_class(est_midi)
    ref_by, _ = score_drums.load_by_class(ref_midi)

    tot_m = tot_eo = tot_ro = 0
    per_class = {}
    for cls in score_drums.CLASS_ORDER:
        est = est_by.get(cls, np.array([]))
        ref = ref_by.get(cls, np.array([]))
        if len(est) == 0 and len(ref) == 0:
            continue
        m, eo, ro = score_drums.match_onsets(est, ref, tol)
        _, _, f = score_drums.prf(m, eo, ro)
        per_class[cls] = round(f, 4)
        tot_m += m; tot_eo += eo; tot_ro += ro

    p, r, f1 = score_drums.prf(tot_m, tot_eo, tot_ro)

    est_all = np.array(sorted(t for v in est_by.values() for t in v))
    ref_all = np.array(sorted(t for v in ref_by.values() for t in v))
    m, eo, ro = score_drums.match_onsets(est_all, ref_all, tol)
    _, _, f_onset = score_drums.prf(m, eo, ro)

    return {
        "f1": f1, "precision": p, "recall": r,
        "f1_adjusted": f1,          # 드럼 GT는 align_drum_gt.py로 이미 정렬됨
        "onset_only_f1": f_onset,
        "per_class": per_class,
        "n_est": int(len(est_all)), "n_ref": int(len(ref_all)),
    }


def separable_ratio(mix: Path, stem: Path) -> float:
    """믹스에서 해당 스템을 뺀 나머지(= 분리가 제거해야 할 다른 악기)의 상대 에너지.

    ★ 이 하니스에서 가장 중요한 안전장치다. 입력이 사실상 그 악기 단독 트랙이면
    분리기는 항등함수가 되고 B−A는 필연적으로 0이 나오는데, 이걸 "분리가 도움이
    안 된다"로 읽으면 완전히 틀린 결론이 된다. (실제로 drums_sample4~11은 믹스로
    알려져 있었으나 실측 결과 비드럼 성분이 1% 미만인 드럼 단독 스템이었다.)

    반환값이 작으면 그 곡의 B−A는 분리 성능에 대해 아무것도 말해주지 않는다.
    """
    import librosa

    y, sr = librosa.load(str(mix), sr=22050, mono=True)
    d, _ = librosa.load(str(stem), sr=22050, mono=True)
    n = min(len(y), len(d))
    if n == 0:
        return 0.0
    y, d = y[:n], d[:n]
    rms_mix = float(np.sqrt((y ** 2).mean()))
    if rms_mix == 0:
        return 0.0
    return float(np.sqrt(((y - d) ** 2).mean()) / rms_mix)


# 이 값은 "결론을 가르는 임계값"이 아니라 **경고를 띄우는 눈금**일 뿐이다.
# 잔차 6%인 파일이 4%인 파일과 본질적으로 다르지 않으므로, 리포트는 임계값으로
# 곡을 버리는 대신 모든 곡의 separable_ratio를 항상 출력하고 그 값으로 정렬해서
# 보여준다 — 숫자 하나가 결론을 좌우하지 않게 하기 위함.
# (이 프로젝트는 매직 넘버로 데인 적이 있다: 드럼 임계값 0.5는 전역 스윕 후에야
#  최적으로 확인됐고, onset 허용오차도 마찬가지였다.)
SEPARABLE_WARN = 0.05


def score_one(instrument: str, est_midi: Path, ref_midi: Path) -> dict:
    if INSTRUMENTS[instrument]["metric"] == "drums":
        return score_drums_micro(est_midi, ref_midi)
    return score_pitched(est_midi, ref_midi)


# ---------------------------------------------------------------- 실행


def resolve(p) -> Path:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def run_song(song: dict, paths: list, out_root: Path, model: str, device: str,
             python_exe: str) -> dict:
    name = song["name"]
    instrument = song["instrument"]
    if instrument not in INSTRUMENTS:
        raise SystemExit(
            f"[{name}] 지원하지 않는 악기 '{instrument}'. 지원: {list(INSTRUMENTS)}. "
            "새 악기를 재려면 INSTRUMENTS에 전사 스크립트 + 스템 이름을 추가할 것."
        )

    mix = resolve(song.get("mix"))
    gt_midi = resolve(song.get("gt_midi"))
    gt_stem = resolve(song.get("gt_stem"))

    if gt_midi is None or not gt_midi.exists():
        raise SystemExit(f"[{name}] gt_midi가 없습니다: {gt_midi}")

    song_out = out_root / name
    results = {}

    # 경로별 입력 오디오 결정
    inputs = {}
    if "A" in paths:
        if mix and mix.exists():
            inputs["A"] = mix
        else:
            print(f"  [{name}] A 건너뜀 — mix 없음")
    if "B" in paths:
        if mix and mix.exists():
            stem_name = INSTRUMENTS[instrument]["stem"]
            # ★ 캐시 키에 믹스 경로 해시를 포함한다. 곡 이름만으로 키를 잡으면 매니페스트가
            # 달라도(예: 유령 편향 측정용 클린 베드 믹스) 이름이 같으면 **다른 오디오의
            # 분리 결과를 조용히 재사용**해서 측정이 통째로 무효가 된다.
            mix_key = hashlib.sha1(str(mix.resolve()).encode()).hexdigest()[:8]
            cache = REPO_ROOT / "eval" / "_sep_cache" / model / f"{name}_{mix_key}"
            t0 = time.time()
            stems = separate(mix, cache, model, device, python_exe)
            sep_sec = time.time() - t0
            if stem_name in stems:
                inputs["B"] = stems[stem_name]
                meta = results.setdefault("_meta", {})
                meta["separation_sec"] = round(sep_sec, 1)
                ratio = separable_ratio(mix, stems[stem_name])
                meta["separable_ratio"] = round(ratio, 4)
                if ratio < SEPARABLE_WARN:
                    print(f"  [{name}] ⚠ 분리할 내용 없음 (비{stem_name} 성분 {ratio:.1%}) "
                          f"— 이 곡의 B−A는 분리 성능을 말해주지 않음")
            else:
                print(f"  [{name}] B 건너뜀 — '{stem_name}' 스템이 없음 (생성: {sorted(stems)})")
        else:
            print(f"  [{name}] B 건너뜀 — mix 없음")
    if "C" in paths:
        if gt_stem and gt_stem.exists():
            inputs["C"] = gt_stem
        else:
            print(f"  [{name}] C 건너뜀 — gt_stem 없음 (상한선 미측정)")

    for path_key, audio in inputs.items():
        out_midi = song_out / f"{path_key}.mid"
        t0 = time.time()
        transcribe(instrument, audio, out_midi, python_exe, resolve(song.get("model")))
        elapsed = time.time() - t0
        res = score_one(instrument, out_midi, gt_midi)
        res["transcribe_sec"] = round(elapsed, 1)
        res["audio"] = str(audio.relative_to(REPO_ROOT)) if audio.is_relative_to(REPO_ROOT) else str(audio)
        results[path_key] = res
        print(f"  [{name}] {path_key} {PATH_LABELS[path_key]:28s} F1 {res['f1']:.4f}  ({elapsed:.0f}s)")

    return results


def print_table(all_results: dict) -> None:
    print()
    print("=" * 78)
    print("3자 비교 — downstream note F1 (지표는 SDR이 아니다)")
    print("★ 정식 집계 = 곡별 micro F1 → 곡 평균 (드럼 LOSO와 같은 방식)")
    print("=" * 78)
    print(f"{'곡':22s} {'A mix':>9s} {'B sep':>9s} {'C gt':>9s} {'B-A':>8s} {'C-B':>8s} {'분리대상':>8s}")
    print("-" * 78)

    # 분리 대상 비율 내림차순으로 정렬해서 보여준다. B−A는 "분리할 내용이 얼마나
    # 있었는가"에 종속적이므로, 그 값을 옆에 두고 함께 읽어야 오독하지 않는다.
    def ratio_of(item):
        r = item[1].get("_meta", {}).get("separable_ratio")
        return -1.0 if r is None else r

    rows = sorted(all_results.items(), key=ratio_of, reverse=True)

    deltas_ba, deltas_cb, ratios = [], [], []
    for name, res in rows:
        a = res.get("A", {}).get("f1")
        b = res.get("B", {}).get("f1")
        c = res.get("C", {}).get("f1")
        ratio = res.get("_meta", {}).get("separable_ratio")

        def fmt(v):
            return f"{v:9.4f}" if v is not None else f"{'—':>9s}"

        ba = f"{b - a:+8.4f}" if (a is not None and b is not None) else f"{'—':>8s}"
        cb = f"{c - b:+8.4f}" if (b is not None and c is not None) else f"{'—':>8s}"
        if a is not None and b is not None:
            deltas_ba.append(b - a)
            ratios.append(ratio if ratio is not None else float("nan"))
        if b is not None and c is not None:
            deltas_cb.append(c - b)

        low = ratio is not None and ratio < SEPARABLE_WARN
        rtxt = f"{ratio:7.1%}{'⚠' if low else ' '}" if ratio is not None else f"{'—':>8s}"
        print(f"{name:22s} {fmt(a)} {fmt(b)} {fmt(c)} {ba} {cb} {rtxt}")

    print("-" * 78)
    if deltas_ba:
        mean_ba = float(np.mean(deltas_ba))
        print(f"{'평균 B−A (분리 효과)':22s} {mean_ba:+.4f}  (n={len(deltas_ba)})")
        # 분리 대상이 많은 곡 절반과 적은 곡 절반을 갈라 보여준다. 두 값이 크게
        # 다르면 평균이 '분리할 내용의 양'에 끌려간 것이므로 평균을 믿으면 안 된다.
        if len(deltas_ba) >= 4 and not np.isnan(ratios).any():
            half = len(deltas_ba) // 2
            top, bottom = deltas_ba[:half], deltas_ba[len(deltas_ba) - half:]
            print(f"{'  분리대상 상위 절반':22s} {np.mean(top):+.4f}   "
                  f"(분리대상 {min(ratios[:half]):.0%}~{max(ratios[:half]):.0%})")
            print(f"{'  분리대상 하위 절반':22s} {np.mean(bottom):+.4f}   "
                  f"(분리대상 {min(ratios[len(ratios)-half:]):.0%}~{max(ratios[len(ratios)-half:]):.0%})")
        if ratios and np.nanmedian(ratios) < SEPARABLE_WARN:
            print()
            print(f"  ⚠ 분리 대상 중앙값이 {np.nanmedian(ratios):.1%}에 불과하다 — 입력이 사실상")
            print("    해당 악기 단독 트랙이라 분리기가 항등함수로 동작한다. 위 B−A는 분리")
            print("    성능에 대해 아무것도 말해주지 않으니 결론에 쓰지 말 것.")
    if deltas_cb:
        mean_cb = float(np.mean(deltas_cb))
        print(f"{'평균 C−B (분리 개선 여지)':22s} {mean_cb:+.4f}   (n={len(deltas_cb)})")
        print()
        print("  해석: C−B가 작으면 분리 모델을 바꿔도 얻을 게 없다(병목은 전사).")
        print("        C−B가 크면 그때 분리에 투자한다.")
    else:
        print()
        print("  ⚠ 경로 C(GT 스템) 미측정 — 상한선을 모르면 '분리에 투자할 가치'를")
        print("    판단할 수 없다. stem+MIDI 정답이 한 세트인 데이터(Slakh2100 등)가 필요.")


def main():
    ap = argparse.ArgumentParser(description="스템 분리 3자 비교 (downstream note F1)")
    ap.add_argument("--manifest", default="eval/separation_manifest.json")
    ap.add_argument("--paths", default="A,B,C", help="측정할 경로, 기본 A,B,C")
    ap.add_argument("--songs", default=None, help="쉼표 구분 곡 이름 필터")
    ap.add_argument("--model", default="htdemucs", help="Demucs 모델 (htdemucs | htdemucs_6s)")
    ap.add_argument("--device", default="cpu", help="cpu | mps")
    ap.add_argument("--out", default=None, help="출력 폴더, 기본 eval/runs/separation/<model>")
    ap.add_argument("--python", default=sys.executable, help="전사/분리 하위프로세스용 파이썬")
    args = ap.parse_args()

    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    songs = manifest["songs"]

    if args.songs:
        wanted = {s.strip() for s in args.songs.split(",")}
        songs = [s for s in songs if s["name"] in wanted]
        missing = wanted - {s["name"] for s in songs}
        if missing:
            raise SystemExit(f"매니페스트에 없는 곡: {sorted(missing)}")

    paths = [p.strip().upper() for p in args.paths.split(",")]
    for p in paths:
        if p not in PATH_LABELS:
            raise SystemExit(f"알 수 없는 경로 '{p}'. 가능: A, B, C")

    out_root = resolve(args.out) if args.out else (REPO_ROOT / "eval" / "runs" / "separation" / args.model)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"매니페스트: {manifest_path}")
    print(f"곡 {len(songs)}개, 경로 {paths}, 분리 모델 {args.model} ({args.device})")
    print(f"출력: {out_root}")
    print()

    all_results = {}
    for song in songs:
        print(f"[{song['name']}] {song['instrument']}")
        all_results[song["name"]] = run_song(
            song, paths, out_root, args.model, args.device, args.python
        )

    print_table(all_results)

    summary = out_root / "results.json"
    summary.write_text(json.dumps(
        {"model": args.model, "device": args.device, "paths": paths, "results": all_results},
        indent=2, ensure_ascii=False,
    ))
    print()
    print(f"저장: {summary.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
