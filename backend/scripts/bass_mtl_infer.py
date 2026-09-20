#!/usr/bin/env python3
"""
Joint MTL 모델(bass_joint_note_model.pt) 추론 래퍼.
(bass-note-verifier 브랜치 — gated shadow integration용, 2026-07-13)

학습 스크립트(eval/train_joint_note_model.py)와 **동일한** 아키텍처/표준화로
로드해서, 파이프라인 스테이지가 octave/offset/validity 점수를 얻게 한다.
octave/offset 헤드만 실제 배포 후보로 쓰고, validity는 audit/UI metadata용.

주의: 피처 순서와 표준화(mean/std)는 학습 때 저장한 그대로여야 결과가
재현된다. 체크포인트에 저장된 *_cols/*_mean/*_std를 그대로 사용.
"""
import sys
from pathlib import Path

import numpy as np


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


class _JointNet:
    """torch 없이도 import 되도록 지연 로딩. 실제 모델은 load()에서 구성."""

    def __init__(self, ckpt_path):
        import torch
        import torch.nn as nn

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hidden = ckpt["hidden"]

        n_oct = len(ckpt["octave_cols"])
        n_frag = len(ckpt["frag_cols"])
        n_prop = len(ckpt["prop_cols"])

        class JointNoteModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.octave_encoder = nn.Sequential(nn.Linear(n_oct, hidden), nn.ReLU())
                self.frag_encoder = nn.Sequential(nn.Linear(n_frag, hidden), nn.ReLU())
                self.prop_encoder = nn.Sequential(nn.Linear(n_prop, hidden), nn.ReLU())
                self.trunk = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                )
                self.octave_head = nn.Linear(hidden, 1)
                self.frag_head = nn.Linear(hidden, 1)
                self.validity_head = nn.Linear(hidden, 1)
                self.offset_head = nn.Linear(hidden, 1)
                self.onset_head = nn.Linear(hidden, 1)
                self.velocity_head = nn.Linear(hidden, 1)

            def octave_logits(self, x):
                return self.octave_head(self.trunk(self.octave_encoder(x))).squeeze(-1)

            def _prop(self, x):
                return self.trunk(self.prop_encoder(x))

            def validity_logit(self, x):
                return self.validity_head(self._prop(x)).squeeze(-1)

            def offset_logdur(self, x):
                return self.offset_head(self._prop(x)).squeeze(-1)

        self.model = JointNoteModel()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.torch = torch

        self.octave_cols = ckpt["octave_cols"]
        self.prop_cols = ckpt["prop_cols"]
        self._oct_mean = np.array([ckpt["octave_mean"][c] for c in self.octave_cols], dtype=np.float32)
        self._oct_std = np.array([ckpt["octave_std"][c] for c in self.octave_cols], dtype=np.float32)
        self._prop_mean = np.array([ckpt["prop_mean"][c] for c in self.prop_cols], dtype=np.float32)
        self._prop_std = np.array([ckpt["prop_std"][c] for c in self.prop_cols], dtype=np.float32)
        self._oct_std[self._oct_std == 0] = 1.0
        self._prop_std[self._prop_std == 0] = 1.0

    def octave_scores(self, feat_dicts):
        """feat_dicts: octave_cols 키를 가진 dict의 리스트. 반환: 후보별 확률."""
        X = np.array([[fd[c] for c in self.octave_cols] for fd in feat_dicts], dtype=np.float32)
        Xs = (X - self._oct_mean) / self._oct_std
        with self.torch.no_grad():
            logits = self.model.octave_logits(self.torch.tensor(Xs)).numpy()
        return logits

    def offset_durations(self, feat_dicts):
        X = np.array([[fd[c] for c in self.prop_cols] for fd in feat_dicts], dtype=np.float32)
        Xs = (X - self._prop_mean) / self._prop_std
        with self.torch.no_grad():
            logdur = self.model.offset_logdur(self.torch.tensor(Xs)).numpy()
        return np.exp(logdur)

    def validity_probs(self, feat_dicts):
        X = np.array([[fd[c] for c in self.prop_cols] for fd in feat_dicts], dtype=np.float32)
        Xs = (X - self._prop_mean) / self._prop_std
        with self.torch.no_grad():
            p = self.torch.sigmoid(self.model.validity_logit(self.torch.tensor(Xs))).numpy()
        return p


_CACHE = {}


def load_model(path=None):
    if path is None:
        path = Path(__file__).resolve().parent / "bass_joint_note_model.pt"
    path = str(path)
    if path not in _CACHE:
        _CACHE[path] = _JointNet(path)
    return _CACHE[path]


def softmax(logits):
    m = np.max(logits)
    e = np.exp(logits - m)
    return e / np.sum(e)
