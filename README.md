# 손가락 개수 실시간 검사 (Finger Counter)

교과 9 머신비전 실습의 **웹 실행 병뚜껑 검사**(IP Webcam → ROI → OpenCV → FastAPI)를 응용해,
카메라 앞 손이 **펴고 있는 손가락 개수(0~5)** 를 세는 프로그램입니다.
목표 개수를 지정하면 병뚜껑처럼 **OK / NG** 로도 판정합니다.

## 병뚜껑 검사와 무엇이 같고 무엇이 다른가

| 단계 | 병뚜껑 검사 | 손가락 검사 |
|---|---|---|
| 입력 | IP Webcam + 웹에서 ROI 드래그 | 동일 |
| 이진화 | Gray → Otsu (BINARY_INV) | **YCrCb 의 Cr(붉은 기) 채널 → Otsu** (피부색) |
| 정리 | Opening → Closing | 동일 (커널 크기는 ROI 크기에 맞춰 자동) |
| Contour | 가장 큰 Contour | 가장 큰 Contour = 손 |
| 특징 | Area Ratio 1개 | 손바닥 중심·반지름(**Distance Transform**) → **원 샘플링**으로 손가락 수 |
| 기준 | 기준 이미지 2장의 중간값 | 기준 이미지 없음 (규칙만으로 셈) |
| 판정 | OK / NG | 0~5개, 목표 개수 지정 시 OK / NG |
| 안정화 | 없음 | 최근 5프레임 중 3번 이상 같아야 숫자 변경 |

## 손가락을 세는 방법

1. **피부색 이진화** – BGR → YCrCb 변환 후 Cr 채널에 Otsu. 피부는 밝기와 상관없이 Cr 이 높아 조명 변화에 강합니다.
   손이 없을 때 Otsu 가 배경을 억지로 나누지 않도록 기준값은 최소 135 로 제한합니다.
2. **손 Contour** – 가장 큰 덩어리를 손으로 보고 그 영역만 채운 마스크를 만듭니다.
3. **손바닥 중심** – Distance Transform 의 최댓값 위치 = 손에서 가장 두꺼운 곳 = 손바닥 중심, 그 값 = 손바닥 반지름 `r`.
4. **원 샘플링 (판정)** – 중심에 반지름 1.5r / 1.75r / 2.0r / 2.3r 원을 그리고, 원 위 360개 점 중 손 위에 있는 **연속 구간**을 셉니다.
   - 너무 좁은 구간은 노이즈로 제외
   - 손바닥 중심에서 **아래쪽(45°~135°)을 향한 구간은 손목·팔**로 제외합니다. 소매에 비스듬히 잘린 손목이
     좁은 띠처럼 보여 손가락으로 세어지던 문제를 '폭'이 아니라 '방향'으로 구분해 해결했습니다.
   - 주먹의 접힌 마디가 작은 원에 걸리는 것을 막기 위해, **안쪽 원과 바로 바깥 원 모두에 같은 방향으로 걸린 구간**(= 길게 뻗은 손가락)만 셉니다.
5. **Convexity Defects (참고값)** – 손을 감싸는 Convex Hull 과 Contour 사이의 골짜기 중 각도 90° 이하·깊이 충분한 것을 세어 `골짜기 수 + 1` 을 함께 보여 줍니다.
   골짜기가 0개면 0개인지 1개인지 구분할 수 없어(`0~1`) 판정에는 원 샘플링을 사용합니다. 두 값을 비교해 보는 것도 공부가 됩니다.

## 폴더 구조

```
finger_counter/
├── src/
│   ├── finger_counter.py      # 핵심 로직 (cap_inspector.py 역할)
│   ├── fastapi_finger_app.py  # 실시간 웹 서버 (fastapi_cap_app.py 역할)
│   ├── finger_batch_test.py   # 폴더 이미지 정확도 측정 (22_cap_batch_test.py 역할)
│   ├── make_demo_hands.py     # 가상 손 이미지 생성 (실제 사진 없을 때)
│   └── fake_ipwebcam.py       # 스마트폰 없이 웹 화면 확인용 가짜 카메라
├── web/
│   ├── finger_inspection.html # 웹 화면 구조
│   └── static/                # style.css (디자인), app.js (동작)
├── tests/test_finger_counter.py
├── requirements.txt
└── pytest.ini
```

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 실행 1 – 스마트폰 IP Webcam 으로 실시간 검사

```bash
uvicorn src.fastapi_finger_app:app --host 0.0.0.0 --port 8000
```

1. 브라우저에서 `http://localhost:8000` 접속
2. 스마트폰 IP Webcam 앱의 주소 입력 (예: `http://192.168.0.25:8080/video`) → **카메라 연결**
3. 영상에서 **손 전체 + 손목 일부**가 들어가도록 드래그
4. 손가락 개수가 표시됩니다. 오른쪽 **목표 손가락 수** 버튼을 누르면 OK/NG 판정으로 바뀝니다.

잘 세지 못할 때
- 오른쪽 위 **이진화 마스크**에서 손만 흰색인지 확인합니다.
- 나무 책상·베이지 벽처럼 피부색과 비슷한 배경은 피하고, 단색 배경(흰 벽, 검은 천)에서 해 보세요.
- 단색 배경이라면 **이진화 방식**을 `어두운 배경 · 밝은 손` 또는 `밝은 배경 · 어두운 손`으로 바꿔 보세요. (병뚜껑과 같은 Gray + Otsu 방식)
- 손바닥이 카메라를 향하게, 손가락이 위를 향하게 세우고 손가락 사이를 벌려 주세요.

## 실행 2 – 스마트폰 없이 웹 화면 확인

```bash
python -m src.make_demo_hands          # 가상 손 30장 생성 (samples/fingers_demo)
python -m src.fake_ipwebcam            # 터미널 1 : 가짜 카메라 http://127.0.0.1:8081/video
uvicorn src.fastapi_finger_app:app     # 터미널 2
```

웹 화면에서 카메라 주소에 `http://127.0.0.1:8081/video` 를 입력하면 0 → 5개 손이 2초마다 바뀌며 나옵니다.

## 실행 3 – Batch 테스트 (정확도 측정)

```bash
python -m src.finger_batch_test --dir samples/fingers_demo   # 가상 손
python -m src.finger_batch_test                              # 내 사진: samples/0 ~ 5/*.jpg
```

- 폴더 이름이 정답 손가락 수입니다. 사진 전체를 ROI 로 쓰므로 **손 전체가 화면 안에** 들어오게 찍으세요.
- 스마트폰 원본 사진은 긴 변 1280px 이하로 줄여서 검사합니다 (`--max-side`, 0 이면 원본).
- 결과: `reports/finger_batch_results.csv`, `outputs/finger_batch/*_result.jpg`
- 가상 손 30장: Accuracy 1.0
- 실제 손 사진 25장 (1~5개 × 5장): **0.48 → 0.92** (손목 방향 규칙 추가 후)
  - 남은 오답 2장: 손을 기울여 손바닥 옆면이 넓게 잡힌 사진, 손이 카메라에 너무 가까워 손바닥 중심이 화면 밖에 걸린 사진

## 자동 테스트

```bash
python -m pytest -q
```

카메라 없이 가상 손으로 0~5개 판정, 손 없음, ROI 잘라내기, 밝은 배경 모드, 서버 API 흐름(목표 개수 OK/NG, 결과 안정화, 입력값 검증)을 확인합니다.

## 한계 (규칙 기반 방식)

- 손가락을 붙이면 원 위에서 한 덩어리로 보여 1개로 셀 수 있습니다.
- 손이 옆으로 누우면 손목이 아래쪽이 아니게 되어 손가락으로 오인할 수 있습니다.
- 얼굴·팔 등 다른 피부가 ROI 에 들어오면 가장 큰 덩어리가 손이 아닐 수 있습니다.

다음 단계로는 이 규칙 기반 결과를 기준선(Baseline)으로 두고, 손 Keypoint 모델(MediaPipe Hands 등)이나
직접 수집한 사진으로 학습한 분류 모델과 정확도를 비교해 볼 수 있습니다.
