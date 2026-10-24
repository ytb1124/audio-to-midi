"""YourMT3+ 전체 악기 출력(*_all.mid)에서 베이스 노트를 고르는 정책을 비교한다.

베이스 단독 오디오에서도 YourMT3+는 노트를 piano/guitar/bass 등 여러 프로그램으로
흩어 놓는다. 재추론 없이 정책별 F1(run_bench.score_bass와 동일 지표)을 낸다.
  bass_only : GM 32~39 프로그램만
  all       : 드럼 제외 전체
  mono      : 전체에서 동시(30ms) 온셋은 하나만 — 베이스 프로그램 우선, 다음 최저음
"""

from __future__ import annotations

import pretty_midi

from run_bench import BASS_SONGS, GT_DIR, ROOT, score_bass

ONSET_GROUP_SEC = 0.03


def notes_of(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    return [(n, int(i.program)) for i in pm.instruments if not i.is_drum for n in i.notes]


def mono(items):
    items = sorted(items, key=lambda x: x[0].start)
    out, group = [], []
    for it in items + [None]:
        if it is not None and group and it[0].start - group[0][0].start <= ONSET_GROUP_SEC:
            group.append(it)
            continue
        if group:
            out.append(min(group, key=lambda x: (0 if 32 <= x[1] <= 39 else 1, x[0].pitch)))
        group = [it] if it is not None else []
    return out


def write(items, path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=33)
    inst.notes = [n for n, _ in items]
    pm.instruments.append(inst)
    pm.write(str(path))


def main() -> int:
    policies = {
        "bass_only": lambda xs: [x for x in xs if 32 <= x[1] <= 39],
        "all": lambda xs: xs,
        "mono": mono,
    }
    tmp = ROOT / "eval" / "runs" / "yourmt3" / "_policy"
    tmp.mkdir(parents=True, exist_ok=True)
    totals = {k: [] for k in policies}
    print(f"{'song':14s} " + " ".join(f"{k:>10s}" for k in policies))
    for song in BASS_SONGS:
        src = ROOT / "eval" / "runs" / "yourmt3" / "bass" / song / "bass_yourmt3_all.mid"
        if not src.exists():
            continue
        items = notes_of(src)
        cells = []
        for name, fn in policies.items():
            out = tmp / f"{song}_{name}.mid"
            write(fn(items), out)
            f1 = score_bass(out, GT_DIR / f"{song}.mid")["f1"]
            totals[name].append(f1)
            cells.append(f"{f1:10.4f}")
        print(f"{song:14s} " + " ".join(cells))
    print(f"{'MEAN':14s} " + " ".join(f"{sum(v)/max(len(v),1):10.4f}" for v in totals.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
