"""3자 비교(B-0)용 합성 믹스 생성 — stem + MIDI 정답이 한 세트인 데이터를 로컬에서 만든다.

문제: `eval/score_separation.py`의 경로 C(GT 스템 → 전사)는 "스템 정답 + MIDI 정답"이
한 세트로 있어야 잰다. 우리가 가진 drums_sample4~11은 **MIDI 정답은 있지만 이미 드럼
단독 스템**이라(실측 비드럼 성분 <1.5%) 믹스가 없고, 반대로 실제 음악 파일은 믹스지만
MIDI 정답이 없다.

해법(Slakh2100과 같은 발상 — 렌더링해서 정답을 만든다):

    1. 실제 음악에서 Demucs로 **드럼을 제거**한 베드를 만든다(vocals+bass+other 합).
       → 이 베드에는 드럼 온셋이 없으므로 GT 드럼 MIDI가 그대로 유효하다.
    2. GT 드럼 스템 + 베드 = 합성 믹스.
    3. 결과적으로 mix / gt_stem / gt_midi 세 개가 전부 갖춰진다.

⚠ 한계(반드시 인지): 실제 프로덕션 믹스가 아니라 두 소스를 더한 것이라 룸/버스 컴프레션/
마스터링이 없고, 베드 자체도 Demucs 산물이라 이미 아티팩트가 있다. 즉 **절대 수치가 아니라
경로 간 상대 비교(A vs B vs C)** 용도다. 진짜 수치는 Slakh2100/MoisesDB로 재야 한다.

사용법:
    python eval/build_synthetic_mixes.py --out eval/synthetic_mixes \
        --manifest eval/separation_manifest_synth.json
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ROOT = REPO_ROOT.parent / "sample audio"

SR = 44100

# 드럼을 뺀 나머지로 베드를 만든다.
BED_STEMS = ["vocals", "bass", "other"]

# 베드 RMS를 드럼 RMS의 몇 배로 맞출지. 1.0이면 드럼과 반주가 대등한 일반적인 밸런스.
DEFAULT_BED_GAIN = 1.0


def demucs(input_audio: Path, out_dir: Path, device: str) -> dict:
    """Demucs 분리(캐시). 반환: {stem_name: path}"""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.stem: p for p in out_dir.glob("*.wav")}
    if existing:
        return existing
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "backend/scripts/demucs_separate.py"),
         str(input_audio), str(out_dir), "--device", device],
        check=True, cwd=str(REPO_ROOT),
    )
    return {p.stem: p for p in out_dir.glob("*.wav")}


def _sum_stems(stems: dict) -> np.ndarray:
    bed = None
    for name in BED_STEMS:
        y, _ = librosa.load(str(stems[name]), sr=SR, mono=True)
        bed = y if bed is None else bed[: min(len(bed), len(y))] + y[: min(len(bed), len(y))]
    return bed


def build_bed(music: Path, work: Path, device: str, strip_drums: bool = False) -> np.ndarray:
    """실제 음악에서 드럼을 제거한 베드(모노, SR)를 만든다.

    strip_drums=True면 1차 베드를 Demucs에 한 번 더 통과시켜 잔향 드럼까지 제거한다.

    ★ 왜 필요한가: Demucs의 드럼 제거는 완벽하지 않아 베드에 원곡 드럼의 잔향이 남는다.
    거기에 다른 곡의 GT 드럼을 얹으면 **템포가 무관한 두 개의 드럼 연주**가 든 믹스가
    되는데, 실제 음악엔 그런 게 없다. 이 유령이 경로 A에 false positive로 들어가면
    B−A가 부풀려진다. strip_drums로 만든 믹스와 비교하면 그 편향의 크기가 그대로 나온다.
    """
    stems = demucs(music, work / "stems" / music.stem, device)
    missing = [s for s in BED_STEMS if s not in stems]
    if missing:
        raise SystemExit(f"{music.name}: 스템 누락 {missing} (있는 것: {sorted(stems)})")

    bed = _sum_stems(stems)
    if not strip_drums:
        return bed

    pass2_dir = work / "bed_clean" / music.stem
    pass2_dir.mkdir(parents=True, exist_ok=True)
    bed_wav = pass2_dir / "bed.wav"
    if not bed_wav.exists():
        sf.write(bed_wav, bed, SR)
    stems2 = demucs(bed_wav, pass2_dir / "stems", device)
    missing2 = [s for s in BED_STEMS if s not in stems2]
    if missing2:
        raise SystemExit(f"{music.name}: 2차 스템 누락 {missing2}")
    return _sum_stems(stems2)


def tile_to(bed: np.ndarray, n: int) -> np.ndarray:
    """베드가 짧으면 이어붙이고, 길면 자른다."""
    if len(bed) >= n:
        return bed[:n]
    reps = int(np.ceil(n / len(bed)))
    return np.tile(bed, reps)[:n]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


def main():
    ap = argparse.ArgumentParser(description="B-0용 합성 믹스 생성")
    ap.add_argument("--out", default="eval/synthetic_mixes")
    ap.add_argument("--manifest", default="eval/separation_manifest_synth.json")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--bed-gain", type=float, default=DEFAULT_BED_GAIN,
                    help="베드 RMS / 드럼 RMS 비율, 기본 1.0")
    ap.add_argument("--strip-bed-drums", action="store_true",
                    help="베드를 Demucs에 한 번 더 통과시켜 잔향 드럼 제거(유령 편향 측정용)")
    ap.add_argument("--songs", default="4,5,6,7,8,9,11",
                    help="쓸 drums_sampleN (기본: 학습 7킷, sample10은 라벨 노이즈라 제외)")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    work = REPO_ROOT / "eval" / "_synth_work"
    out_dir.mkdir(parents=True, exist_ok=True)

    music_files = sorted((SAMPLE_ROOT / "music sample").glob("*.wav"))
    if not music_files:
        raise SystemExit(f"베드로 쓸 음악이 없습니다: {SAMPLE_ROOT / 'music sample'}")
    print(f"베드 소스 {len(music_files)}곡: {[m.stem for m in music_files]}")

    beds = []
    for m in music_files:
        print(f"  드럼 제거 중: {m.name}" + ("  (+잔향 제거 2차)" if args.strip_bed_drums else ""))
        beds.append((m.stem, build_bed(m, work, args.device, args.strip_bed_drums)))

    gt = REPO_ROOT / "eval" / "ground_truth"
    songs = []
    for i, n in enumerate(s.strip() for s in args.songs.split(",")):
        stem_wav = gt / f"drums_sample{n}.wav"
        gt_midi = gt / f"drums_sample{n}.mid"
        if not (stem_wav.exists() and gt_midi.exists()):
            print(f"  건너뜀: drums_sample{n} (파일 없음)")
            continue

        drums, _ = librosa.load(str(stem_wav), sr=SR, mono=True)
        bed_name, bed_full = beds[i % len(beds)]          # 곡마다 다른 베드를 순환 배정
        bed = tile_to(bed_full, len(drums))

        r_d, r_b = rms(drums), rms(bed)
        if r_b > 0:
            bed = bed * (args.bed_gain * r_d / r_b)

        mix = drums + bed
        peak = np.max(np.abs(mix))
        if peak > 0.99:                                    # 클리핑 방지(두 경로 모두 동일 스케일)
            scale = 0.99 / peak
            mix, drums_scaled = mix * scale, drums * scale
        else:
            drums_scaled = drums

        mix_path = out_dir / f"drums_sample{n}_mix.wav"
        stem_path = out_dir / f"drums_sample{n}_gtstem.wav"
        sf.write(mix_path, mix, SR)
        sf.write(stem_path, drums_scaled, SR)              # 믹스와 같은 스케일의 GT 스템

        ratio = rms(mix - drums_scaled) / rms(mix) if rms(mix) else 0.0
        print(f"  drums_sample{n}: 베드={bed_name}  분리대상 {ratio:.1%}")

        songs.append({
            "name": f"drums_sample{n}_synth",
            "instrument": "drums",
            "mix": str(mix_path.relative_to(REPO_ROOT)),
            "gt_midi": str(gt_midi.relative_to(REPO_ROOT)),
            "gt_stem": str(stem_path.relative_to(REPO_ROOT)),
            "bed_source": bed_name,
        })

    manifest = {
        "_note": "합성 믹스(build_synthetic_mixes.py). 실제 프로덕션 믹스가 아니므로 "
                 "절대 수치가 아니라 A/B/C 경로 간 상대 비교용으로만 읽을 것.",
        "bed_gain": args.bed_gain,
        "strip_bed_drums": args.strip_bed_drums,
        "songs": songs,
    }
    (REPO_ROOT / args.manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n{len(songs)}곡 생성 → {args.manifest}")


if __name__ == "__main__":
    main()
