import sys, numpy as np, pretty_midi
sys.path.insert(0, "eval/heldout")  # 실행: .venv-backend/bin/python eval/heldout/larsnet_metal4_cv.py (dev 기각 기록)
import larsnet_detect as L
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

LAB = {42: 0, 44: 0, 46: 1, 49: 2, 52: 2, 55: 2, 57: 2, 51: 3, 53: 3, 59: 3}

def mel_db(y):
    import librosa
    M = librosa.feature.melspectrogram(y=y, sr=L.SR, n_fft=1024, hop_length=L.HOP, n_mels=16, fmin=300, fmax=11000)
    return 10 * np.log10(M + 1e-10)

def feats(d, times):
    out = []
    mats = [mel_db(L.load(d / f"{s}.wav")) for s in ("hihat", "cymbals")]
    n = mats[0].shape[1]
    fr = lambda t: int(np.clip(round(t * L.SR / L.HOP), 0, n - 1))
    ts = np.sort(times)
    for t in times:
        k = np.searchsorted(ts, t, side="right"); ioi = min(1.0, ts[k] - t) if k < len(ts) else 1.0
        row = [ioi]
        for M in mats:
            a = M[:, fr(t):fr(t + 0.03) + 1].max(axis=1)
            row += list(a)
            for lo, hi in ((0.08, 0.15), (0.15, 0.3), (0.3, 0.6)):
                row += list(M[:, fr(t + lo):fr(t + hi) + 1].mean(axis=1) - a)
        # 두 스템의 어택 에너지 차이(어느 스템이 이 타격을 가져갔나)
        row.append(mats[0][:, fr(t):fr(t + 0.03) + 1].max() - mats[1][:, fr(t):fr(t + 0.03) + 1].max())
        out.append(row)
    return np.array(out)

def dataset(songs):
    X, Y, G = [], [], []
    for gi, s in enumerate(songs):
        pm = pretty_midi.PrettyMIDI(str(L.DATA / "slakh_train" / s / "drums.mid"))
        lab = {}
        for i in pm.instruments:
            for nn in i.notes:
                if nn.pitch in LAB: lab.setdefault(round(nn.start, 3), LAB[nn.pitch])
        if not lab: continue
        ts = np.array(sorted(lab))
        X.append(feats(L.stem_dir("slakh_train", s), ts)); Y.append([lab[t] for t in ts]); G += [gi] * len(ts)
    return np.vstack(X), np.concatenate(Y), np.array(G)

if __name__ == "__main__":
    songs = sorted(p.name for p in (L.RUNS / "_sep" / "slakh_train").iterdir() if (p / "stems.json").exists() and (L.DATA / "slakh_train" / p.name / "drums.mid").exists())[:40]
    X, Y, G = dataset(songs)
    print("class counts", np.bincount(Y))
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, G):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], Y[tr])
        pred[te] = clf.predict(X[te])
    from sklearn.metrics import classification_report
    print(classification_report(Y, pred, target_names=["closed", "open", "crash", "ride"], digits=3))
    print("majority acc", np.bincount(Y).max() / len(Y))
