# 피아노 외 악기 전사 개선 — 결과 (2026-10-24, 브랜치 `improve/non-piano-amt`)

## 요약

| 경로 | 기준선 평균 F1 | 새 방식 평균 F1 | 판정 |
|---|---:|---:|---|
| 베이스 6곡 | **0.8864** (Basic Pitch + 하이브리드 후처리) | 0.6710 (YourMT3+ 단독) / 0.8129 (YourMT3+ → 하이브리드 후처리) | 기준선보다 낮음 → **기본값 유지, 옵션으로만 추가** |
| 드럼 8곡 | **0.7755** (현행 E-GMD 분류기) | 0.6106 (ADTOF) | 기준선보다 낮음 → **기본값 유지, 옵션으로만 추가** |

- 피아노 회귀: 같은 고정 샘플의 MIDI sha256이 작업 전후 동일(`24a3790c…`). 피아노 경로 코드(Transkun, 서스테인 보정, `run_piano_mode`)는 바뀌지 않았다.
- 베이스 옥타브 오류: 일정한 **+12 편향은 없다.** 원본 GT 대비 일정한 차이는 −12(GT 라벨이 한 옥타브 높게 저장된 문제)이고, 모델 쪽 +12는 Basic Pitch 원출력에서만 곡별 38.6~84.0%로 **불규칙하게** 나오며 기존 후처리가 거의 0으로 줄인다(아래 6절).

## 1. 베이스 — 곡별 F1 (`eval/score.py`, onset 50ms)

대표값은 score.py가 출력하는 마지막 보정 점수다: 전역 시간 오프셋(|lag|>15ms)이 감지되면 시간+옥타브 보정 F1, 아니면 옥타브 보정 F1. 같은 규칙을 모든 방식에 똑같이 적용했다.

| 곡 | 기준선 (basicpitch) | yourmt3 | yourmt3_hybrid |
|---|---:|---:|---:|
| bass_sample | 0.6957 | 0.6087 | 0.6667 |
| bass_sample2 | 0.9989 | 0.9003 | 0.9507 |
| bass_sample3 | 0.8626 | 0.4958 | 0.8342 |
| bass_sample4 | 0.9651 | 0.6346 | 0.8978 |
| bass_sample5 | 0.8157 | 0.6501 | 0.7116 |
| bass_test | 0.9802 | 0.7367 | 0.8166 |
| **평균** | **0.8864** | **0.6710** | **0.8129** |

다른 집계 방식으로 본 평균(순위는 같다):

| 집계 | 기준선 | yourmt3 | yourmt3_hybrid |
|---|---:|---:|---:|
| 옥타브 보정만 (시간 보정 없음) | 0.7939 | 0.6706 | 0.7308 |
| 원본 GT 그대로 (보정 없음) | 0.1664 | 0.1686 | 0.1365 |

원본 GT 점수가 낮은 이유는 5곡의 GT가 한 옥타브 높게 저장돼 있기 때문이다(6절 참고).

### YourMT3+ 베이스 노트 선택 정책 (`eval/yourmt3_select_policy.py`)
YourMT3+는 베이스 단독 오디오에서도 노트를 piano/guitar/bass 등으로 나눠 낸다(bass_sample2: bass 162, piano 268, guitar 154). 그래서 베이스 프로그램만 고르면 노트 대부분을 버리게 된다.

| 정책 | 평균 F1 |
|---|---:|
| bass_only (GM 32~39만) | 0.4538 |
| all (드럼 제외 전체) | 0.6210 |
| **mono** (동시 30ms 온셋마다 1개, 베이스 프로그램 우선 → 최저음) — 채택 | **0.6710** |

## 2. 드럼 — 곡별 MICRO F1 (`eval/score_drums.py`, onset 50ms)

| 곡 | 기준선 (배포 E-GMD 분류기) | egmd_loso (참고) | adtof |
|---|---:|---:|---:|
| drums_sample4 | 0.7873 | 0.5701 | 0.6544 |
| drums_sample5 | 0.7278 | 0.6390 | 0.2736 |
| drums_sample6 | 0.9920 | 0.9803 | 0.9598 |
| drums_sample7 | 0.9206 | 0.8458 | 0.5853 |
| drums_sample8 | 0.8275 | 0.5941 | 0.5261 |
| drums_sample9 | 0.9221 | 0.9071 | 0.9173 |
| drums_sample10 | 0.0892 | 0.0892 | 0.1343 |
| drums_sample11 | 0.9373 | 0.8132 | 0.8338 |
| **평균** | **0.7755** | **0.6798** | **0.6106** |
| ONSET-ONLY 평균 (클래스 무시) | 0.8134 | 0.7808 | **0.8244** |

이 비교를 읽을 때 주의할 점:
- **기준선은 in-sample이다.** 배포 분류기는 sample10을 뺀 7곡으로 학습됐다. 처음 보는 킷 기준의 공정한 참고값은 `egmd_loso`(각 곡을 뺀 LOSO 폴드 모델)이며 평균 0.6798이다. ADTOF는 LOSO 참고값보다도 낮다.
- **ADTOF는 클래스가 5개뿐이다**(BD/SD/HH/TT/CY+RD). 그래서 오픈 하이햇, 라이드, 셰이커를 구분할 수 없다. 반면 타격 검출력(ONSET-ONLY)은 세 방식 중 가장 높아서, 손실은 대부분 클래스 체계에서 나온다. 셰이커 1113개가 있는 sample5에서 가장 크게 떨어진다.
- 클래스별 평균 F1 (LOSO vs ADTOF): kick 0.870 / **0.877**, snare 0.681 / **0.695**, crash 0.000 / **0.204**, hihat_closed **0.727** / 0.528, hihat_open **0.384** / 0.000, tom **0.419** / 0.413.

## 6. 베이스 옥타브 오류 진단 (`eval/diagnose_octave_bias.py`, 원자료 `eval/runs/octave_bias.json`)

onset으로 매칭한 쌍(pitch는 보지 않음)에서 생성−정답 pitch 차이의 비율이다. orig = 원본 GT, octfix = `*_octcorrected.mid`.

| 곡 | GT | 기준선 +12 | 기준선 −12 | BP 원출력 +12 | BP 원출력 −12 | yourmt3 +12 | yourmt3 −12 | yourmt3_hybrid +12 | yourmt3_hybrid −12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bass_sample | orig | 0.0% | 88.2% | 11.1% | 44.4% | 0.0% | 70.0% | 0.0% | 88.2% |
| | octfix | 0.0% | 5.9% | 38.9% | 0.0% | 5.0% | 5.0% | 0.0% | 5.9% |
| bass_sample2 | orig | 0.0% | 100.0% | 0.0% | 23.0% | 0.0% | 92.4% | 0.0% | 95.2% |
| | octfix | 0.0% | 0.0% | 75.4% | 0.0% | 2.3% | 0.0% | 0.0% | 0.0% |
| bass_sample3 | orig | 0.0% | 84.0% | 0.0% | 61.7% | 0.0% | 76.1% | 0.0% | 85.8% |
| | octfix | 1.4% | 0.7% | 38.3% | 0.0% | 17.5% | 0.0% | 0.6% | 0.6% |
| bass_sample4 | orig | 0.0% | 99.6% | 0.0% | 15.3% | 0.0% | 82.2% | 0.0% | 94.9% |
| | octfix | 0.0% | 0.6% | 84.0% | 0.0% | 10.9% | 1.3% | 0.0% | 0.6% |
| bass_sample5 | orig | 0.0% | 89.3% | 0.0% | 32.7% | 0.0% | 71.6% | 0.0% | 74.0% |
| | octfix | 0.0% | 0.0% | 61.3% | 0.0% | 0.7% | 0.0% | 0.0% | 0.0% |
| bass_test | orig | 0.0% | 0.0% | 38.6% | 0.0% | 3.6% | 0.7% | 0.0% | 0.0% |

결론:
1. **일정한 +12 편향은 없다.** 일정한 차이는 원본 GT 대비 −12(84~100%)이고, 5곡(bass_sample~5)의 원본 GT가 한 옥타브 높게 익스포트된 라벨 문제와 맞아떨어진다. 보정 GT(octfix)로 보면 기준선의 ±12는 0~6%다. 따로 만든 GT인 bass_test는 99.6%가 일치한다.
2. 모델 쪽 +12는 **Basic Pitch 원출력**에서 나온다. 보정 GT 대비 곡별 38.6~84.0%로 곡마다 다르다. 즉 상수 편향이 아니라 노트 단위로 흔들리는 옥타브 오류이고, 기존 후처리(octave_final_v2)가 거의 0으로 줄인다. 그래서 일괄 −12 보정 같은 상수 보정은 답이 아니다.
3. YourMT3+ 원출력도 일부 곡에서 +12가 남는다(sample3 17.5%, sample4 10.9%). 하이브리드 후처리를 거치면 0~0.6%로 줄어든다.

## 7. 기본값 결정

| 경로 | 조건 (새 방식 평균 F1 > 기준선) | 결정 |
|---|---|---|
| 베이스/일반 | 0.6710, 0.8129 < 0.8864 | `basicpitch` 기본값 유지. `yourmt3`, `yourmt3_hybrid`는 옵션 |
| 드럼 | 0.6106 < 0.7755 (LOSO 0.6798보다도 낮음) | `egmd` 기본값 유지. `adtof`는 옵션 |

## 사용법 (API 계약 추가 — 기존 요청은 그대로 동작)

- `POST /api/transcribe` multipart에 선택 필드 `backend` 추가
  - `mode=bass`: `basicpitch`(기본) | `yourmt3` | `yourmt3_hybrid`
  - `mode=general`: `basicpitch`(기본) | `yourmt3`
  - `mode=drums`: `egmd`(기본) | `adtof`
- `POST /api/drums`에도 같은 `backend` 필드(`egmd`|`adtof`)를 추가했다.
- 응답 `result`에 `backend`(실제로 쓴 엔진)와 `requestedBackend`가 들어간다. 새 엔진이 설치돼 있지 않거나 실패하면 Basic Pitch / E-GMD로 자동 폴백하고, 이때 `backend`는 폴백 엔진 이름이 된다.
- 환경변수 기본값: `BASS_BACKEND`, `GENERAL_BACKEND`, `DRUMS_BACKEND`. CLI: `backend/cli.py transcribe --backend ...`.

## 모델 출처와 설치 (공식 저장소만 사용)

- YourMT3+: HuggingFace Space `mimbres/YourMT3`(저자 공식). 체크포인트 `YPTF.MoE+Multi (noPS)`(`mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/last.ckpt`).
- ADTOF: GitHub `MZehren/ADTOF`. 모델 `Frame_RNN`, adtofAll fold 0.
- 메인 venv의 torch/numpy가 깨지지 않도록 각 모델은 격리 venv(`.venv-yourmt3`, `.venv-adtof`)에서 subprocess로 호출한다. 설치 스크립트: `backend/scripts/setup_isolated_models.sh`. `third_party/`와 venv는 gitignore 대상이다.
- 라이선스 주의: ADTOF 저장소는 CC BY-NC-SA 4.0(비상업)이다.

## 재현

```bash
.venv-backend/bin/python eval/run_bench.py bass  --backend basicpitch --tag baseline
.venv-backend/bin/python eval/run_bench.py bass  --backend yourmt3    --tag yourmt3
.venv-backend/bin/python eval/run_bench.py bass  --backend yourmt3_hybrid --tag yourmt3_hybrid
.venv-backend/bin/python eval/run_bench.py drums --backend egmd       --tag baseline
.venv-backend/bin/python eval/run_bench.py drums --backend egmd_loso  --tag egmd_loso
.venv-backend/bin/python eval/run_bench.py drums --backend adtof      --tag adtof
cd eval && ../.venv-backend/bin/python make_results.py --bass baseline yourmt3 yourmt3_hybrid --drums baseline egmd_loso adtof
```

## 피아노 회귀 기준

- 고정 샘플: `eval/fixtures/piano_regression_30s.wav` = `sample audio/piano smaple/original.wav`의 30.0~60.0초(48kHz 스테레오 PCM16). wav sha256 `68f137b9d9470a05e60c2abbe3915d02a37f8508e3506492c61361802fe8b09c`.
- 변환 MIDI(`piano_transkun_sustain.mid`) sha256 `24a3790cb7e8b30162350dac85b8f08e17c2e7567901bd03edf572c72e4159b0`
  - 작업 전(2회 실행, 결정적) = 작업 후 CLI = 작업 후 `/api/transcribe` 결과 → **세 값이 모두 일치**.

## 한계 / 남은 일

- 드럼 기준선이 in-sample이라 ADTOF와의 비교는 기준선에 유리하다. LOSO 참고값과 비교해도 결론은 같다.
- ADTOF 출력은 kick/snare/crash(LOSO 대비)에서 앞선다. 클래스별로 라우팅하는 하이브리드는 시험하지 않았다.
- `score.py` CLI는 음수 시간 lag로 GT가 0초 앞으로 밀리면 mir_eval 예외로 죽는다(bass_sample4 + yourmt3). 하니스(`run_bench.py`)에서는 0초로 클램프해서 채점했고, score.py 자체는 수정하지 않았다.
- YourMT3+는 CPU 추론으로 5분짜리 곡 하나에 약 1~4분 걸린다.
