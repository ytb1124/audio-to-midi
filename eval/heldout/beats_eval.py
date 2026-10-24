"""R5 비트 추적 평가 — Slakh test 20곡(30~90초) 비트·다운비트 F-measure.

정답: all_src.mid의 템포 맵·박자표(pretty_midi get_beats/get_downbeats). Slakh 오디오는 MIDI를 그대로
렌더링하므로 시각 정답이 정확하다. 라벨 점검 결과 템포 이벤트는 20곡 모두 0번 트랙에 있다(pretty_midi가
읽는 위치). Track01883·01903은 박자표가 없어 4/4 가정 → 다운비트 정답이 불확실하므로 다운비트 집계에서
따로 표시한다.
지표: mir_eval.beat.f_measure(±70 ms, 앞 5초 제외 기본값). 곡별 평균, 곡 부트스트랩 95% CI, paired 차이.
추적기 출력: runs/_beats/<tracker>/<song>.json (beats_alt.py, .venv-beat). librosa는 여기서 직접 만든다.

사용법: .venv-backend/bin/python eval/heldout/beats_eval.py [--trackers beat_this madmom allin1 librosa]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_heldout import bootstrap_ci  # noqa: E402

RUNS = HERE / "runs" / "_beats"
WIN = (30.0, 90.0)
NO_TS = {"Track01883", "Track01903"}
# 라벨 오류(추적기와 무관한 점검): 정답 비트 위에 정답 드럼 타격이 하나도 없고(0%),
# 격자를 0.35/0.70박 밀면 96~99%가 맞는다 → MIDI 템포 격자가 음악과 위상이 어긋난 파일.
# 판정 규칙(사전 고정): "정답 비트 ±30 ms 드럼 타격 비율 < 0.1 이고, 어떤 위상 이동에서 ≥ 0.9".
LABEL_BAD = {"Track01895", "Track01903"}


def gt(song: str):
    pm = pretty_midi.PrettyMIDI(str(HERE / "data" / "_raw" / "slakh" / song / "all_src.mid"))
    b = pm.get_beats()
    d = pm.get_downbeats()
    sel = lambda x: x[(x >= WIN[0]) & (x < WIN[1])] - WIN[0]  # noqa: E731
    return sel(b), sel(d)


def librosa_beats(song: str) -> dict:
    out = RUNS / "librosa" / f"{song}.json"
    if out.exists():
        return json.loads(out.read_text())
    import librosa

    y, sr = librosa.load(str(HERE / "data" / "slakh" / song / "mix.wav"), sr=22050, mono=True)
    _, fr = librosa.beat.beat_track(y=y, sr=sr)
    b = librosa.frames_to_time(fr, sr=sr).tolist()
    res = {"beats": b, "downbeats": b[::4]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", nargs="+", default=["beat_this", "madmom", "allin1", "librosa"])
    ap.add_argument("--baseline", default="librosa")
    args = ap.parse_args()
    songs = [i["song"] for i in json.loads((HERE / "data" / "manifest.json").read_text())["slakh"]]
    scores = {}
    for tr in args.trackers:
        scores[tr] = {}
        for s in songs:
            if tr == "librosa":
                est = librosa_beats(s)
            else:
                p = RUNS / tr / f"{s}.json"
                if not p.exists():
                    continue
                est = json.loads(p.read_text())
            gb, gd = gt(s)
            eb = np.asarray(est["beats"], float)
            ed = np.asarray(est.get("downbeats", []), float)
            scores[tr][s] = {"beat_f": mir_eval.beat.f_measure(gb, eb),
                             "downbeat_f": mir_eval.beat.f_measure(gd, ed)}
    out = {}
    for metric, excl in (("beat_f", set()), ("beat_f", LABEL_BAD), ("downbeat_f", LABEL_BAD)):
        print(f"\n[{metric}, 제외: {sorted(excl) or '없음'}]")
        print(f"\n| 추적기 | {metric} 평균 [95% CI] | {args.baseline} 대비 [95% CI] | n |")
        print("|---|---|---|---|")
        for tr, sc in scores.items():
            ks = [k for k in sc if k not in excl and not (metric == "downbeat_f" and k in NO_TS)]
            v = [sc[k][metric] for k in ks]
            if not v:
                continue
            lo, hi = bootstrap_ci(v)
            base = scores.get(args.baseline, {})
            d = [sc[k][metric] - base[k][metric] for k in ks if k in base]
            dlo, dhi = bootstrap_ci(d)
            diff = "—" if tr == args.baseline else f"{np.mean(d):+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
            print(f"| {tr} | {np.mean(v):.3f} [{lo:.3f}, {hi:.3f}] | {diff} | {len(v)} |")
            out.setdefault(tr, {})[metric + ("_clean" if excl else "_all")] = float(np.mean(v))
    (RUNS / "scores.json").write_text(json.dumps({"summary": out, "per_song": scores}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
