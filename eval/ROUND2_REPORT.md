# Round 2 report — 2026-10-25 07:45 ~ (브랜치 `round2/2026-10-25`)

순서: 사전 점검 → 실험(N1~N5) → 판정 → 개발·배포(기본값이 바뀐 경우만). 하니스와 규칙은 지난 라운드와 같다
(채점 v2, 곡별 평균, paired 95% 부트스트랩 CI 10,000회 seed 0, 파라미터는 개발셋에서만, 테스트는 후보마다 1회).
수치의 원자료는 `eval/round2/results/`(점수 JSON·리포트 텍스트)에 있다.

**결론: 기본값·opt-in 변경 없음 → 배포하지 않음.** 차점은 N2 `adtof_split`(MDB 8클래스 +0.055, Slakh 보조에서 −0.004 유의 하락으로 근소 미달).

## 0. 사전 점검

| 후보 | 코드 | 가중치 | 라이선스 | 학습 데이터 | 오염 판정 |
|---|---|---|---|---|---|
| N1 MuScriptor (medium 307M) | GitHub muscriptor/muscriptor | HF `MuScriptor/*` — **게이트(로그인 + 라이선스 동의 필요)** | 코드 MIT / 가중치 CC BY-NC 4.0 | 합성 사전학습: Lakh MIDI 등 MIDI 145만 곡 × 사운드폰트 250+ / 실녹음 파인튜닝: 내부 17만 곡(1.1만 시간, 공개 데이터셋 이름 미기재) (arXiv 2607.08168) | **Slakh 계열 3세트 오염**(Slakh는 Lakh MIDI에서 뽑은 곡 → test MIDI가 사전학습에 들어갔을 가능성 높음; 실베이스 세트도 같은 MIDI). MDB: 내부 데이터 구성 불명 → "불명" |
| N2 LarsNet | GitHub polimi-ispl/larsnet (LICENSE 파일 없음) | 저자 Google Drive 562 MB | 가중치 CC BY-NC 4.0 | StemGMD(Groove MIDI × 어쿠스틱 킷 10종 합성) | 깨끗(Slakh·MDB 미사용). 검출에 쓴 ADTOF도 MDB·Slakh 미사용 |
| N3 ADT_STR | GitHub pier-maker92/ADT_STR, HF `Pierfrancesco/adt-str`(공개) | 공개(τ 0.4/0.6/0.8, 69M) | CC BY-SA 4.0 | 학습 설정(`configs/train/setting-tau-*.yaml`): `lakh_matched` 드럼 MIDI를 원샷 샘플로 합성. MDB·ENST는 평가에만 | **Slakh test 오염**(Lakh MIDI). MDB 깨끗 |
| N4a Explore This! (arXiv 2610.14817) | GitHub CPJKU/explore_this (MIT) | **미공개**(릴리스·체크포인트 없음, 학습 스크립트만) | MIT | Beat This!와 같은 16개 데이터셋(Slakh 없음) | — (건너뜀) |
| N4b CASM (arXiv 2610.09318) | GitHub zhanh-he/casm-beat-tracking (LICENSE 파일 없음) | 학습 가중치 없음. SMC fold 1–7로 고정한 스칼라 설정(7F) 동봉 | 미표기 | 백본 = Beat This! final0(MIT, Slakh 미사용) | 깨끗 |
| N5a SwiftF0 0.3.0 | GitHub lars76/swift-f0, PyPI `swift-f0` | 패키지 동봉(ONNX, 9.6만 파라미터) | MIT (학습 데이터 묶음 pitch-benchmark는 CC BY-NC-SA) | pitch-benchmark 코퍼스(Slakh 언급 없음). 우리 벤드 세트는 CC0 업라이트 실샘플 렌더라 오디오가 학습에 있을 수 없음 | 깨끗 |
| N5b arXiv 2601.11768 (Bitra & Beigi) | **없음**(논문·HTML에 코드/가중치 링크 없음) | 없음 | — | MedleyDB 스템(자기지도) | — (건너뜀) |

## 1. 후보별 결과

### N1. MuScriptor — 건너뜀(데이터 접근 차단)
- 가중치가 HF 게이트 저장소라 로그인 + 라이선스 동의가 필요하다(`muscriptor/utils/download.py`가 401 시 같은 안내를 낸다). 이 환경에는 HF 토큰이 없고, 계정 로그인·약관 동의는 대리할 수 없다. 대체 공식 배포처는 Replicate(유료 외부 API, 오디오 업로드)뿐이다.
- 목표의 "데이터 신청제 → 사유 기록 후 건너뜀"에 해당. 사용자가 `hf auth login` 후 https://huggingface.co/MuScriptor/muscriptor-medium 에서 동의하면 같은 하니스로 바로 돌릴 수 있다.
- 돌렸더라도 베이스 세 세트(slakh·slakh_realbass·slakh_realbass_up)는 모두 Slakh test MIDI라 **오염**으로 판정에서 빠진다. 판정 가능한 것은 MDB 드럼뿐(학습 데이터 불명 단서 포함).

### N2. LarsNet 피스 분리 → 8클래스

사전 등록(`eval/heldout/larsnet_detect.py` 머리말, 08:10): 세 시스템을 개발 분할(Slakh train songs[40:], 16곡)에서 8클래스 F1로 하나 고르고 MDB·Slakh test에 1회.
하이햇 오픈/클로즈·크래시/라이드 세분은 LarsNet 스템 특징 + 로지스틱 회귀(Slakh train songs[:40] 정답 온셋으로 학습, 문턱 0.5).

| 시스템 | dev 5클래스 | dev 8클래스 (현행 대비) | 비고 |
|---|---|---|---|
| 현행 stem_adtof | 0.695 | 0.605 | — |
| **adtof_split** (현행 출력 + 세분) | 0.695 (동일) | 0.599 (−0.006 [−0.014, −0.001]) | **dev 선택** |
| larsnet_onset (스템별 단순 온셋, delta dev 튜닝) | 0.609 | 0.536 (−0.068) | |
| larsnet_adtof (스템별 ADTOF, 해당 클래스만) | 0.399 | 0.351 (−0.253) | ADTOF가 고립 스템에서 약하다(하이햇 스템에서 189타 vs 전체 스템 1,812타). 첫 채점의 kick 0.000은 내 매핑 버그(ADTOF 킥=36, 톰=45)였고 고친 뒤 값이다 |

테스트(1회):

| 세트 | 지표 | 현행 | adtof_split | paired 차이 [95% CI] |
|---|---|---|---|---|
| **MDB 23곡(대표)** | 8클래스 F1 | 0.759 | **0.815** | **+0.055 [+0.006, +0.118]** |
| MDB | 5클래스 F1 | 0.859 | 0.859 | 0 (구조상 동일) |
| Slakh test 20곡(보조) | 8클래스 F1 | 0.648 | 0.643 | **−0.0042 [−0.0108, −0.00005]** |

- MDB 클래스별: ride 0.000 → 0.549, 오픈 하이햇 0.000 → 0.200, crash 0.481 → 0.440, 클로즈 하이햇 0.732 → 0.733.
- **판정: 기본값·opt-in 모두 아님(근소 미달).** 대표 세트 하한 > 0은 만족했지만 보조 세트에서 유의한 하락(상한 −0.00005)이 있다.
- **메커니즘(라벨 먼저 의심)**: 심벌 분류기가 MDB의 ADTOF 심벌 타격 93%를 ride로 낸다. 사후 진단으로 "심벌을 전부 ride로" 바꾸기만 해도 MDB +0.050 [−0.001, +0.115]다(`n2_allride_diagnostic.json`, 판정에 쓰지 않음).
  즉 이득의 대부분은 학습한 판별력이 아니라 Slakh train의 ride 우세 사전확률이고, crash 위주 곡에서는 손해다. 킷 간 교차검증(곡 단위 GroupKFold)에서 4클래스 금속 분류기는 다수 클래스 비율(0.85)보다 낮은 0.82였다(ride 재현율 0.04, `larsnet_metal4_cv.py`).
  CLAUDE.md의 "킷 간 심벌 일반화는 렌더 데이터만으로 안 된다"와 같은 결과다.

### N3. ADT_STR

| 단계 | 결과 |
|---|---|
| dev 선택(5클래스, 16곡) | τ0.4 스템 0.486 / τ0.6 스템 0.490 / **τ0.8 스템 0.546** / τ0.8 믹스 0.352 (현행 ADTOF 0.695). 믹스 입력은 곡당 약 2분이라 τ0.8에서만 확인(사전 결정) |
| **MDB(대표, 깨끗)** | 5클래스 **0.709 vs 0.859, −0.150 [−0.192, −0.108]**. 8클래스 0.668 vs 0.759, −0.091 [−0.143, −0.038] |
| Slakh test (**오염**, 참고) | 5클래스 0.573 vs 0.725 |

- **판정: 기각.** 하이햇을 크게 놓친다(dev 정답 3,003타 중 출력 492타). 출력 pitch는 저장소 고유 26클래스 코드라 GM으로 되돌려 채점했다(`adtstr_transcribe.py`의 CUSTOM_TO_GM; 매핑 점검 후 하이햇 부족은 모델 출력 자체임을 확인).
- 참고: 현행이 못 내는 클래스는 낸다 — MDB 오픈 하이햇 0.314, 탬버린 등 기타 0.720. 8클래스 체계를 다시 시도할 때 앙상블 후보.

### N4. 비트 — Explore This! 건너뜀, CASM 기각

Slakh test 18곡(라벨 오류 2곡 제외), mir_eval ±70 ms. 같은 Beat This! final0 로짓에서 기본 후처리(minimal)를 다시 내 R5 출력과 20곡 모두 일치함을 확인한 뒤 CASM만 바꿨다.

| 추적기 | 비트 F (18곡) | 다운비트 F (17곡) | CMLt | AMLt |
|---|---|---|---|---|
| **Beat This! (현행)** | 0.927 | 0.878 | 0.883 | 0.918 |
| Beat This! + CASM | 0.931 (+0.005 [−0.002, +0.012]) | 0.859 (−0.020 [−0.061, +0.006]) | 0.890 (+0.007 [−0.013, +0.027]) | 0.908 (−0.010 [−0.036, +0.011]) |

- **판정: 현행 유지.** 비트 F 하한 −0.002. 전체 20곡도 +0.004 [−0.002, +0.011]. 디코딩은 곡당 0.01초로 매우 빠르다.
- Explore This!: 공개 체크포인트가 없다. 학습하려면 Beat This!의 16개 데이터셋(상당수 오디오 비공개)으로 8-fold × 100 epoch(논문 설정)을 GPU에서 돌려야 해서 이 맥에서는 불가 → 건너뜀.

### N5. f0 — SwiftF0 기각, 2601.11768 건너뜀

R7과 같은 설정: dev_bends_up에서 파라미터 격자 162개를 "무벤드 노트 오차 ≤ 2 cent" 아래 F1 최대로 고르고 bendtest_up 1회.

| 추출기 | dev F1 | test F1 | test P / R | 벤드 곡선 오차 | 무벤드 오차 |
|---|---|---|---|---|---|
| **pYIN (현행)** | 0.781 | **0.813** | 0.965 / 0.703 | 31.7 c | 0.17 c |
| CREPE (R7) | 0.568 | 0.748 | 0.960 / 0.613 | 35.6 c | 0.11 c |
| SwiftF0 | 0.608 | 0.695 | 0.653 / 0.742 | 37.8 c | 0.81 c (조건 통과) |

- 곡별 paired: SwiftF0 − pYIN F1 **−0.184 [−0.320, −0.038]**, 무벤드 오차 +0.45 c [−0.10, +0.97].
- 실녹음(Bass_2·Mosquito): 안정 노트 6.03 → 5.79 c(−0.24 [−0.46, −0.08], n=476), 슬라이드 노트 166.8 → 87.9 c(−78.9 [−104.5, −54.2], **n=36**).
  SwiftF0은 기준 f0 집합(나머지 4개 추출기 중앙값)과 노트 표본이 pYIN(n=188)과 달라 직접 비교할 수 없고 n이 작다. 판정은 테스트 F1로 한다.
- **판정: pYIN 유지.** SwiftF0은 정밀도가 낮다(0.65). 모델 음역 하한이 46.875 Hz라 E1(41.2 Hz)~F1을 추적하지 못한다.
- 2601.11768: 코드·가중치 미공개 → 건너뜀.

## 2. 판정 요약

| 경로 | 바꾼 것 | 유지 | 차점 |
|---|---|---|---|
| 베이스 | 없음 | Basic Pitch + 하이브리드 | (N1 미실행) |
| 드럼 | 없음 | htdemucs → ADTOF 5클래스 | adtof_split(8클래스, MDB +0.055이나 Slakh −0.004 유의) |
| 비트 | 없음 | Beat This! minimal | CASM(비트 +0.005, 하한 −0.002) |
| 피치 벤드 | 없음 | pYIN | SwiftF0(−0.184) |

## 3. 다음 후보
1. **N1 MuScriptor MDB 드럼**: HF 동의만 되면 바로 실행 가능. 베이스는 오염이라 판정 불가 — 베이스는 Slakh가 아닌 MIDI로 만든 실베이스 세트가 있어야 한다.
2. **8클래스 드럼**: 이번 결과로 "현행 ADTOF + 세분"이 5클래스 손실 없이 8클래스를 올리는 구조임은 확인됐다. 다만 세분기가 사전확률에 의존한다. 실제 녹음 킷 다양성(ADTOF 데이터셋 신청, ENST) 확보가 선행 조건이다. ADT_STR의 오픈 하이햇·퍼커션 출력과 앙상블하는 안도 있다.
3. 비트: CASM은 비트 연속성(CMLt)이 소폭 오르는 경향이 있다(+0.007, 유의하지 않음). 템포 2배·절반 혼동 곡이 더 많은 세트에서 다시 볼 가치가 있다.

## 4. 운영 메모
- 무거운 작업은 동시에 최대 2개, `nice -n 15`, `OMP_NUM_THREADS=2`로 돌렸다. 부하 평균이 15를 넘은 시점(07:49·07:51, 시스템 프로세스 mds·WindowServer 때문)에는 가벼운 작업만 돌렸다.
- 서브에이전트는 쓰지 않았다(설치·다운로드가 짧아 메인에서 직접 처리).
- 다운로드 합계 약 2.4 GB(ADT_STR HF 1.1 GB, LarsNet 0.54 GB, torch 2.8 venv 약 0.7 GB, CASM·SwiftF0 소량). 모두 저자 GitHub/HF/저자 Drive 링크에서 받았다.
- 채점 뒤 삭제: LarsNet 분리 스템 4.8 GB, ADT_STR 가중치 1.1 GB, LarsNet 가중치 0.59 GB, `.venv-adtstr`. 남긴 것: 점수 JSON, 재현 스크립트, 출력 MIDI(소용량), 세분 모델(joblib, gitignore).
- 사용자 메시지: 이번 라운드 중에는 없었다.

## 재현
```bash
# N4
.venv-beat/bin/python eval/heldout/beats_casm.py        # CASM 소스는 runs/_r2/src/casm (git clone zhanh-he/casm-beat-tracking)
.venv-backend/bin/python eval/heldout/beats_eval.py --trackers beat_this beat_this_casm madmom librosa --baseline beat_this
# N5 (.venv-f0에 pip install swift-f0)
.venv-f0/bin/python eval/heldout/pitchbend_eval.py f0 --set dev_bends_up --methods swiftf0   # bendtest_up도 같은 방식, real_f0 --methods swiftf0
.venv-f0/bin/python eval/heldout/pitchbend_eval.py tune --set dev_bends_up --methods swiftf0
.venv-f0/bin/python eval/heldout/pitchbend_eval.py test --set bendtest_up && .venv-f0/bin/python eval/heldout/pitchbend_eval.py real
# N2 / N3 (격리 venv .venv-adtstr: torch 2.8, transformers<5, safetensors, omegaconf, pretty_midi, soundfile, pyyaml, h5py, pedalboard)
.venv-backend/bin/python eval/heldout/larsnet_detect.py separate --sets slakh_train_all mdb slakh
.venv-backend/bin/python eval/heldout/larsnet_detect.py fit
.venv-backend/bin/python eval/heldout/drums_r2.py run --systems adtstr_stem_t08 adtstr_stem_t06 adtstr_stem_t04 adtstr_mix_t08 adtof_split larsnet_onset larsnet_adtof --sets slakh_train
.venv-backend/bin/python eval/heldout/drums_r2.py run --systems adtstr_stem_t08 adtof_split --sets mdb slakh
.venv-backend/bin/python eval/heldout/drums_r2.py report --runs stem_adtof adtstr_stem_t08 adtof_split --sets mdb slakh
```
