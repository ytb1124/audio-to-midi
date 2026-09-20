"""실사용 테스트 중 "이 구간이 이상하다"를 조사하는 도구.

사용자가 웹에서 곡을 변환한 뒤 특정 구간을 지목하면, 그 구간의 생성 노트 /
클래스별 확신도 / 후보 검출 상황을 한 번에 덤프한다. 정성 평가를 다음 라운드의
정량 작업으로 넘기기 위한 연결 고리다.

산출물은 job 폴더에 이미 다 있다:
    outputs/<jobId>/drums.mid            생성 MIDI
    outputs/<jobId>/drum_adt_audit.csv   class/time/velocity/confidence
    outputs/<jobId>/drums.wav            분리된 드럼 스템(separate=true인 경우)
    uploads/<jobId>.<ext>                업로드 원본 오디오

사용법:
    python eval/inspect_job.py <jobId>                 # 요약
    python eval/inspect_job.py <jobId> 42.0 48.0       # 42~48초 구간 상세
    python eval/inspect_job.py <jobId> 42.0 48.0 --all # EXOTIC 포함 전 클래스 확신도
"""

import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
UPLOADS = REPO_ROOT / "uploads"


def find_job(job_id):
    d = OUTPUTS / job_id
    if not d.exists():
        matches = [p for p in OUTPUTS.iterdir() if p.name.startswith(job_id)]
        if len(matches) == 1:
            d = matches[0]
        else:
            raise SystemExit(f"job을 찾을 수 없습니다: {job_id}")
    src = next((p for p in UPLOADS.glob(f"{d.name}.*")), None)
    return d, src


def load_audit(d):
    f = d / "drum_adt_audit.csv"
    if not f.exists():
        return []
    with open(f) as fh:
        return [
            {"class": r["class"], "t": float(r["time_s"]),
             "vel": int(r["velocity"]), "conf": float(r["confidence"])}
            for r in csv.DictReader(fh)
        ]


def main():
    ap = argparse.ArgumentParser(description="실사용 job 구간 진단")
    ap.add_argument("job_id")
    ap.add_argument("start", nargs="?", type=float)
    ap.add_argument("end", nargs="?", type=float)
    ap.add_argument("--all", action="store_true", help="EXOTIC 포함 전 클래스 재계산")
    args = ap.parse_args()

    d, src = find_job(args.job_id)
    rows = load_audit(d)
    print(f"job      {d.name}")
    print(f"원본     {src if src else '(없음)'}")
    print(f"MIDI     {d / 'drums.mid'}")
    if (d / "drums.wav").exists():
        print(f"드럼스템 {d / 'drums.wav'}   ← 원본과 나란히 들어볼 것")
    print()

    if not rows:
        print("감사 CSV가 없습니다.")
        return

    if args.start is None:
        c = Counter(r["class"] for r in rows)
        print(f"전체 {len(rows)}노트 / {max(r['t'] for r in rows):.1f}초")
        for cls, n in c.most_common():
            confs = [r["conf"] for r in rows if r["class"] == cls]
            print(f"  {cls:14s}{n:6d}  확신도 중앙값 {sorted(confs)[len(confs)//2]:.3f}")
        print("\n구간을 지목하려면: python eval/inspect_job.py <jobId> <시작초> <끝초>")
        return

    end = args.end if args.end is not None else args.start + 5.0
    sel = [r for r in rows if args.start <= r["t"] <= end]
    print(f"=== {args.start:.2f}s ~ {end:.2f}s — 생성 노트 {len(sel)}개 ===")
    print(f"{'시각':>9s}  {'class':14s}{'vel':>5s}{'확신도':>8s}")
    for r in sorted(sel, key=lambda x: x["t"]):
        print(f"{r['t']:9.3f}  {r['class']:14s}{r['vel']:5d}{r['conf']:8.3f}")

    if not sel:
        print("(이 구간에 생성된 노트 없음 — 검출 실패이거나 EXOTIC으로 제외된 클래스)")

    if args.all:
        print()
        print("=== EXOTIC 포함 재계산 (crash/ride/triangle/shaker 확신도 확인) ===")
        out = d / "_inspect_exotic.mid"
        audit = d / "_inspect_exotic.csv"
        adt_input = d / "drums.wav"
        if not adt_input.exists():
            adt_input = src
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "backend/scripts/drum_audio_to_midi.py"),
             str(adt_input), str(out), "--audit", str(audit), "--exotic"],
            check=True, cwd=str(REPO_ROOT), capture_output=True,
        )
        with open(audit) as fh:
            ex = [r for r in csv.DictReader(fh)
                  if args.start <= float(r["time_s"]) <= end]
        by = defaultdict(list)
        for r in ex:
            by[r["class"]].append((float(r["time_s"]), float(r["confidence"])))
        for cls in ("crash", "ride", "triangle", "shaker"):
            if cls in by:
                print(f"  {cls}: " + ", ".join(f"{t:.2f}s({c:.2f})" for t, c in by[cls][:12]))
        if not any(c in by for c in ("crash", "ride", "triangle", "shaker")):
            print("  (이 구간에 EXOTIC 클래스 발화 없음)")


if __name__ == "__main__":
    main()
