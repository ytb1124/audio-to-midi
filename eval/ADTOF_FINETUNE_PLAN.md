# ADTOF 분리 스템 파인튜닝 실험 계획

작성 2026-10-24. 상태: **계획(미착수)**. 연구용 실험이다. ADTOF는 CC BY-NC-SA 4.0이라 결과물도 비상업으로만 쓴다.

## 1. 질문

ADTOF(Frame_RNN, adtofAll fold 0)를 **Demucs로 분리한 드럼 스템**에 맞춰 파인튜닝하면 실제 녹음 드럼 전사가 좋아지는가?

## 2. 출발점 (홀드아웃 실측, `eval/heldout/`)

| 입력 | MDB Drums 23곡 (실제 녹음) | Slakh test 20곡 (렌더링) |
|---|---|---|
| htdemucs 드럼 스템 → ADTOF | **0.859** | 0.716 |
| 믹스 → ADTOF | 0.811 | 0.704 |
| (참고) 스템 → 현행 E-GMD 분류기 | 0.546 | 0.663 |

지표는 5클래스(kick/snare/hihat/tom/cymbal) micro F1, onset 허용오차 50 ms, 곡별 평균이다.

MDB 클래스별 F1(스템 → ADTOF): kick 0.90, snare 0.85, hihat 0.81, **tom 0.59**, **cymbal 0.68**.

ADTOF는 이미 믹스보다 분리 스템에서 더 잘한다(+0.05). 그래서 이 실험의 여지는 "분리 아티팩트에 익숙해지는 것"과 "약한 클래스(tom, cymbal)"에 있다. **예상 이득은 작다(+0.01~0.04).** 크게 기대하지 않는다.

## 3. 가설과 위험

- **H1:** 학습 분포를 추론 분포(분리 스템)와 맞추면 F1이 오른다. CLAUDE.md B-0의 "학습/프로덕션 분포 불일치"와 같은 논리다.
- **위험 R1 — 렌더링 데이터로 인한 망각.** CLAUDE.md의 E-GMD 단독 학습은 순손실(−0.096)이었다. 렌더링 스템만으로 파인튜닝하면 ADTOF가 실제 음악에서 배운 것을 잊을 수 있다. 원래 도메인 데이터를 섞어 되새김(rehearsal)하는 것으로 막는다.
- **위험 R2 — 평가셋 누출.** MDB는 유일한 실제 녹음 테스트셋이다. 학습·검증·임계값 튜닝에 절대 쓰지 않는다.

## 4. 데이터

| 이름 | 내용 | 역할 | 비고 |
|---|---|---|---|
| **S-slakh** | Slakh2100 **train**의 믹스 → htdemucs 드럼 스템, 정답은 드럼 MIDI | 학습(렌더링 도메인) | HF 미러에서 필요한 곡만 받는다. test 곡과 겹치지 않음을 트랙 ID로 확인 |
| **S-enst** | ENST-Drums의 반주 포함 믹스 → htdemucs 드럼 스템 | 드러머 1·2는 학습, **드러머 3은 검증**(조기 종료·임계값) | 실제 녹음. ADTOF adtofAll 학습에 없다(`dataLoader.py`의 `adtof-all` = adtof_yt + adtof_F70). 라이선스·다운로드 조건 확인 필요 |
| **R-adtof** | ADTOF 공개 데이터셋(멜 스펙트로그램, 분리 안 됨) | 되새김용. 배치의 30~50% | Zenodo "upon request"(10.5281/zenodo.10084510). **오디오가 아니라 스펙트로그램이라 분리를 할 수 없다** — 원래 도메인 유지용으로만 쓴다 |
| MDB Drums 23곡 | 실제 녹음 | **최종 테스트 전용** | 이미 `eval/heldout/data/mdb` |
| Slakh test 20곡 | 렌더링 | 보조 테스트 | 이미 `eval/heldout/data/slakh` |

전처리 원칙:
- 학습 스펙트로그램은 **추론 경로(`adtof_transcribe.py` → `predictFolder`)와 같은 함수**로 만든다. 학습 시작 전에 같은 오디오를 두 경로로 넣어 입력 텐서가 일치하는지 확인한다. 피처 불일치는 CLAUDE.md v5에서 "조용히 이상한 예측"을 낸 전례가 있다.
- 분리는 heldout 벤치마크와 같은 설정(htdemucs, `DEMUCS_DEVICE=mps`)으로 한다.
- 대용량은 외장 SSD에 환경변수 경로로 둔다(`EGMD_ROOT`와 같은 방식, 절대경로 하드코딩 금지).

## 5. 실험 조건 (사전 등록)

| 조건 | 학습 데이터 | 설정 |
|---|---|---|
| **A0** | 없음(사전학습 그대로) | 기준선 |
| A1 | S-slakh | 전체 파인튜닝 |
| A2 | S-enst(드러머 1·2) | 전체 파인튜닝 |
| A3 | S-slakh + S-enst + R-adtof 되새김 | 전체 파인튜닝 |
| A4 | A3와 같음 | 합성곱 층 고정, GRU와 출력층만 학습 |

공통 설정:
- `Frame_RNN_adtofAll_0` 가중치에서 시작한다.
- 학습률 1e-4(원래 5e-4의 1/5).
- S-enst 드러머 3의 F1로 조기 종료한다(patience 3).
- 클래스별 피크 임계값(`peakThreshold`)은 **검증셋에서만** 다시 맞춘다. 원래 값은 adtofAll용 튜닝이다.
- 시드를 고정하고 기록한다.

먼저 처리량을 잰다. 맥 CPU(또는 tensorflow-metal)로 1 에폭이 몇 분인지 확인한 뒤 조건 수를 확정한다. Frame_RNN은 GRU 60×3의 작은 모델이라 로컬 학습이 가능할 것으로 보지만, 아직 측정하지 않았다.

## 6. 판정 기준 (미리 고정)

조건 선택은 검증셋(S-enst 드러머 3)으로만 한다. 선택된 **한 조건만** MDB에서 한 번 채점한다.

채택 조건(모두 만족):
1. MDB 23곡에서 A0 대비 곡별 paired 차이의 95% 부트스트랩 CI 하한 > 0
2. MDB 클래스별 평균 F1이 A0 대비 0.05 넘게 떨어지는 클래스가 없다
3. Slakh test가 A0 대비 0.02 넘게 떨어지지 않는다(보조)

중단 규칙: A2·A3·A4가 모두 검증셋에서 A0를 넘지 못하면 MDB 채점 없이 실험을 끝내고 A0를 유지한다.

## 7. 산출물

- `eval/adtof_finetune/`: 데이터 준비 스크립트, 학습 스크립트, 설정(시드 포함)
- 체크포인트: gitignore 대상. 경로와 sha256을 결과 문서에 기록한다
- `eval/adtof_finetune/RESULTS.md`: 검증 곡선, 선택 근거, MDB 1회 채점표(곡별·클래스별·CI)
- 채택 시: `adtof_transcribe.py`에 `--weights` 옵션을 추가하고, 홀드아웃 하니스(`run_heldout.py`)에 `stem_adtof_ft` 시스템을 추가한다

## 8. 범위 밖 (별도 실험)

- **클래스 확장**(open/closed 하이햇, crash/ride 분리): ADTOF는 5클래스라 현행 E-GMD 분류기(9클래스)보다 표현력이 낮다. 출력층을 7클래스로 늘리는 실험은 라벨이 있는 데이터(Slakh, ENST, E-GMD)로 따로 한다.
- 다른 분리기(RoFormer 계열)와의 조합.
- velocity 학습. 지금은 어택 dB 매핑(`backend/scripts/drum_velocity.py`)을 쓴다.
