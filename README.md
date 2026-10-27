<div align="center">

# Finger Counter

**스마트폰 카메라 영상에서 펴고 있는 손가락 개수(0~5)를 실시간으로 세는 OpenCV 웹 검사 시스템**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-MJPEG%20Streaming-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![pytest](https://img.shields.io/badge/tests-41%20passed-2EA44F?logo=pytest&logoColor=white)](#품질-관리)
[![Accuracy](https://img.shields.io/badge/real%20photos-92%25-1957BF)](#성능-평가)

**한국어** · [English](README.en.md)

<img src="docs/images/demo.gif" alt="웹 대시보드에서 실시간으로 손가락 개수를 세고 목표 개수와 비교해 OK/NG를 판정하는 화면" width="860">

</div>

## 한눈에 보기

| | |
|---|---|
| **무엇을** | 스마트폰(IP Webcam) 영상 속 손가락 개수를 실시간으로 세고, 목표 개수와 비교해 **OK / NG** 판정 |
| **어떻게** | 학습 데이터 없는 **규칙 기반 OpenCV 파이프라인** — 피부색 이진화 · Contour · Distance Transform · 원 샘플링 |
| **결과** | 실제 스마트폰 사진 25장 **Accuracy 0.48 → 0.92** |
| **품질** | 자동 테스트 41개, 실제 버그를 재현한 회귀 테스트, 카메라 없이 돌리는 가짜 IP Webcam |
| **배경** | 비솔 Vision AI 교육과정 머신비전 실습 「웹 실행 병뚜껑 검사」를 응용한 미니 프로젝트 |

<details>
<summary><b>수업 실습(병뚜껑 검사)과 비교</b></summary>

같은 실시간 구조(IP Webcam → FastAPI → OpenCV → 웹)를 유지하면서, 모양이 매번 달라지는 대상(손)에서 **개수**를 세도록 문제를 확장했습니다.

| | 병뚜껑 검사 (수업) | 손가락 검사 (이 프로젝트) |
|---|---|---|
| 이진화 | Gray → Otsu | **YCrCb의 Cr 채널** → Otsu (피부색, 조명에 강함) |
| 특징 | 가장 큰 Contour의 Area Ratio 1개 | 손바닥 중심·반지름(**Distance Transform**) + **원 샘플링** |
| 기준값 | 기준 이미지 2장의 중간값 | 기준 이미지 없음 (손바닥 크기 대비 비율 규칙) |
| 출력 | OK / NG | **0~5개** + 목표 개수 지정 시 OK / NG |
| 실시간 안정화 | 없음 | 최근 5프레임 중 3번 일치해야 결과 변경 |

</details>

## 시스템 구조

<img src="docs/images/architecture_ko.png" alt="스마트폰 IP Webcam에서 FastAPI, OpenCV, HTML 대시보드로 이어지는 데이터 흐름도" width="100%">

- **카메라 스레드 1개**가 영상을 읽고 검사합니다. 브라우저가 여러 개 열려도 카메라 연결은 하나입니다.
- 영상(`/stream.mjpg`)과 상태(`/api/status`, 0.4초 주기)를 따로 받아, 영상이 끊겨도 상태 표시는 유지됩니다.
- 설정이 바뀌면 처리 중이던 프레임 결과를 버려 **이전 설정의 결과가 섞이지 않게** 했습니다.

## 알고리즘

<img src="docs/images/pipeline_ko.jpg" alt="입력, Cr 채널, 이진화 마스크, 손바닥 찾기, 원 샘플링, 결과로 이어지는 6단계와 실제 사진 결과" width="100%">

손바닥 중심에서 원을 그려 **손이 원을 가로지르는 구간 수 = 손가락 수**로 셉니다. 모든 기준을 손바닥 반지름 r 대비 비율로 정해, 손이 카메라에서 멀든 가깝든 같은 규칙이 적용됩니다.

<details>
<summary><b>단계별 처리와 설계 이유</b></summary>

| 단계 | 처리 | 이렇게 한 이유 |
|---|---|---|
| ①&nbsp;이진화 | BGR → YCrCb, **Cr 채널에 Otsu** (기준값 최소 135) | 피부는 밝기(Y)와 무관하게 Cr이 높아 조명 변화에 강함. 손이 없을 때 Otsu가 배경을 억지로 나누지 않도록 하한을 둠 |
| ②&nbsp;정리 | Opening → Closing, 커널 크기는 이미지 크기에 비례 | 해상도가 달라도 같은 규칙이 적용되도록 |
| ③&nbsp;손&nbsp;찾기 | 가장 큰 외곽 Contour = 손 | 작은 피부색 잡음 제거 |
| ④&nbsp;손바닥 | **Distance Transform 최댓값** = 손바닥 중심, 그 값 = 반지름 r | 손가락·손목은 손바닥보다 가늘어서 중심이 자연스럽게 손바닥에 잡힘 |
| ⑤&nbsp;원&nbsp;샘플링 | 1.5r · 1.75r · 2.0r · 2.3r 원 위 360점에서 손 위에 있는 **연속 구간** 검출 | 구간 하나 = 원을 가로지르는 손가락 하나 |
| ⑥&nbsp;필터 | 이웃한 두 원에 **같은 방향(±15°)으로 모두 걸린** 구간만 인정, **아래쪽 45°~135°는 손목**으로 제외 | 주먹의 접힌 마디(짧음)와 소매에 잘린 손목(방향)을 걸러냄 |

참고값으로 **Convexity Defects**(손가락 사이 골짜기 수 + 1)도 계산해 화면에 보여 줍니다. 골짜기가 0개면 0개와 1개를 구분할 수 없어서 최종 판정에는 원 샘플링을 씁니다.

</details>

## 성능 평가

| 데이터 | 장수 | Accuracy | 1개 | 2개 | 3개 | 4개 | 5개 |
|---|---|---|---|---|---|---|---|
| 실제 스마트폰 사진 (개선 전) | 25 | 0.48 | 3/5 | 3/5 | 2/5 | 2/5 | 2/5 |
| **실제 스마트폰 사진 (개선 후)** | 25 | **0.92** | 5/5 | 4/5 | 4/5 | 5/5 | 5/5 |
| 가상 손 이미지 (0~5개 × 5) | 30 | 1.00 | | | | | |

결과 이미지를 한 장씩 분석해 **소매에 잘린 손목이 손가락으로 세어지는 문제**를 찾았고, 손목 판정 규칙을 폭에서 방향으로 바꿔 정확도를 0.48에서 0.92로 올렸습니다.

<details>
<summary><b>문제 해결 과정 자세히 보기</b></summary>

1. **관찰:** 이진화 마스크는 거의 완벽했는데 개수가 하나씩 많거나 적었습니다. 결과 이미지 25장을 모두 열어 원인을 분류했습니다.
2. **원인:**
   - 검은 소매가 손목을 비스듬히 자르며 생긴 **좁은 띠**가 "손목 = 폭이 넓은 구간" 규칙을 통과해 손가락으로 세어졌습니다.
   - 소매가 손바닥 아래를 가려 **반지름이 작게 측정**되자, 엄지·새끼가 손목으로 빠졌습니다.
3. **해결:** 손목을 **폭이 아니라 방향**(아래쪽 45°~135°)으로 구분하고, 폭 상한을 2.0r로 완화했습니다.
4. **재발 방지:** 같은 상황을 재현하는 회귀 테스트를 추가하고, 수정 전 코드에서 실패하는 것을 확인했습니다.

남은 오답 2장은 손을 기울여 손바닥 옆면이 잡힌 경우와, 손이 너무 가까워 손가락이 화면 밖으로 잘린 경우입니다. 25장에 과하게 맞추지 않으려고 여기서 멈췄습니다.

</details>

## 품질 관리

자동 테스트 41개(pytest)로 핵심 로직과 서버 API를 검증하고, 실제 버그를 재현한 회귀 테스트로 같은 문제가 다시 생기지 않게 막았습니다.

<details>
<summary><b>테스트 구성과 실행 명령</b></summary>

```bash
python -m pytest -q               # 41 passed
python -m src.finger_batch_test   # 폴더 이름(1~5)을 정답으로 정확도 측정 → CSV 보고서
```

| 구분 | 확인 내용 |
|---|---|
| 핵심 로직 | 가상 손 0~5개 × 촬영 조건 5가지(위치·크기·기울기·밝기), 손 없음, 잘라낸 이미지, 밝은 배경 모드, 잘못된 입력 |
| 회귀 테스트 | 소매에 잘린 손목 띠를 손가락으로 세지 않는지 (실제 버그 재현) |
| 서버 API | 화면·정적 파일, 목표 개수 OK/NG 흐름, 5프레임 중 3번 안정화, 입력값 검증(ROI 크기·목표 범위·사설 IP만 허용), 설정 변경 시 결과 초기화 |
| 카메라 없는 테스트 | `process_frame()`을 카메라 스레드와 분리해 이미지 한 장으로 서버 동작 검증 |
| 수동 테스트 | `fake_ipwebcam.py` — 이미지 폴더를 IP Webcam과 같은 MJPEG로 송출 |

</details>

## 실행 방법

<details>
<summary><b>설치와 실행 (스마트폰 IP Webcam)</b></summary>

```bash
git clone https://github.com/yonghoonchoi88/finger_counter.git && cd finger_counter
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.fastapi_finger_app:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 2
```

1. 스마트폰 IP Webcam 앱에서 서버를 시작합니다 (PC와 같은 Wi-Fi).
2. 브라우저 `http://localhost:8000`에서 주소(예: `http://192.168.0.25:8080/video`)를 입력하고 **카메라 연결**을 누릅니다.
3. 손 전체가 들어가도록 ROI를 드래그합니다. 목표 개수를 누르면 OK/NG로 판정합니다.

</details>

<details>
<summary><b>스마트폰 없이 실행 · 정확도 측정</b></summary>

```bash
python -m src.make_demo_hands                              # 가상 손 30장 생성
python -m src.fake_ipwebcam                                # 터미널 1 → http://127.0.0.1:8081/video
uvicorn src.fastapi_finger_app:app --port 8000             # 터미널 2

python -m src.finger_batch_test                            # samples/1~5 실제 사진
python -m src.finger_batch_test --dir samples/fingers_demo # 가상 손
```

결과: `reports/finger_batch_results.csv`, `outputs/finger_batch/*_result.jpg`

</details>

<details>
<summary><b>프로젝트 구조</b></summary>

```
finger_counter/
├── src/
│   ├── finger_counter.py       # 핵심 알고리즘 (이진화 · Contour · Distance Transform · 원 샘플링)
│   ├── fastapi_finger_app.py   # 실시간 서버 (카메라 스레드 · 안정화 · REST API · MJPEG)
│   ├── finger_batch_test.py    # 폴더 단위 정확도 측정 · CSV 보고서
│   ├── make_demo_hands.py      # 가상 손 이미지 생성기
│   └── fake_ipwebcam.py        # 테스트용 가짜 IP Webcam (MJPEG 송출)
├── web/
│   ├── finger_inspection.html  # 대시보드 (영상 · 결과 · 설정 · 동작 원리)
│   └── static/                 # app.js · style.css
├── tests/test_finger_counter.py
├── samples/1~5/                # 실제 손 사진 (정답 = 폴더 이름)
└── docs/images/                # 데모 GIF · 구조도 · 알고리즘 그림 (한국어/영어)
```

</details>

<details>
<summary><b>REST API</b></summary>

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/` | 대시보드 |
| `GET` | `/stream.mjpg` | 검사 결과를 그린 실시간 영상 (MJPEG) |
| `GET` | `/api/status` | 카메라 상태 · 손가락 수 · 판정 · 세부 수치 |
| `POST` | `/api/camera` | IP Webcam 주소 연결 (사설 IP · http(s)만 허용) |
| `POST` / `DELETE` | `/api/roi` | 검사 영역 설정 / 해제 (0~1 비율 좌표) |
| `POST` | `/api/target` | 목표 손가락 수 (0~5, `null` = 해제) |
| `POST` | `/api/settings` | 이진화 방식 `skin` · `dark_bg` · `light_bg`, 마스크 표시 |

</details>

## 한계와 다음 단계

손가락을 붙이면 1개로 셀 수 있고, 손이 옆으로 누우면 손목 방향 규칙이 맞지 않습니다. 얼굴·팔 등 다른 피부가 ROI에 크게 들어오면 손으로 오인할 수 있습니다.

<details>
<summary><b>다음 단계</b></summary>

- [ ] 주먹(0개) 사진과 다른 장소·조명 사진으로 테스트셋 확장 (현재 한 장소, 1~5개 × 5장)
- [ ] 손목 방향을 고정값(아래쪽) 대신 손의 주축(PCA)으로 추정해 기울어진 손 대응
- [ ] MediaPipe Hands(21개 관절 Keypoint)와 같은 테스트셋으로 정확도 · 속도 비교
- [ ] 직접 수집한 사진으로 경량 분류 모델(CNN) 학습, 규칙 기반을 Baseline으로 비교
- [ ] GitHub Actions로 push마다 pytest + Batch Test 자동 실행

</details>

---

<div align="center">

**최용훈 (Yonghoon Choi)** · [@yonghoonchoi88](https://github.com/yonghoonchoi88)<br>
SQA·테스트 자동화 5년+ → Vision AI 전환 · 비솔 「현장 데이터 기반 Vision AI & 산업분야 Vertical AI 융합 전문가 양성」 과정 미니 프로젝트

</div>
