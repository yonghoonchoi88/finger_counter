"""손가락 세기 로직과 FastAPI 서버 자동 테스트.

실행:  python -m pytest -q
카메라 없이 가상 손 이미지(src/make_demo_hands.py)로 검사합니다.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import fastapi_finger_app as appmod
from src.finger_counter import count_fingers, segment_hand
from src.make_demo_hands import POSES, SHOTS, draw_hand, render


def demo_hand(count: int, shot: int = 0) -> np.ndarray:
    cx, cy, r, tilt, gain = SHOTS[shot]
    return render(draw_hand(POSES[count], cx, cy, r, tilt), gain, seed=count * 10 + shot)


# ---------- 핵심 로직 ----------

@pytest.mark.parametrize("count", range(6))
@pytest.mark.parametrize("shot", range(len(SHOTS)))
def test_counts_demo_hands(count, shot):
    result = count_fingers(demo_hand(count, shot))
    assert result["status"] == "COUNTED"
    assert result["fingers"] == count


def test_empty_background_is_no_hand():
    background = np.full((480, 640, 3), (205, 200, 190), np.uint8)
    assert count_fingers(background)["status"] == "NO_HAND"


def test_light_bg_mode_matches_cap_style_binary():
    # 흰 배경 위 검은 손 : 병뚜껑과 같은 Gray + Otsu + BINARY_INV 방식
    mask = draw_hand(POSES[3], 320, 300, 60, 0)
    image = np.where(mask[..., None] > 0, 40, 230).astype(np.uint8).repeat(3, axis=2)
    assert count_fingers(image, "light_bg")["fingers"] == 3


def test_roi_crop_still_counts():
    frame = demo_hand(4)
    roi = frame[120:480, 180:460]
    assert count_fingers(roi)["fingers"] == 4


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        segment_hand(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        segment_hand(np.zeros((10, 10, 3), np.uint8), mode="unknown")


# ---------- 서버 (카메라 없이 process_frame 직접 호출) ----------

@pytest.fixture()
def client():
    appmod.state = appmod.LiveState()
    with TestClient(appmod.app) as c:
        yield c
    appmod.state.stop()


def feed(frames: int, image: np.ndarray) -> None:
    for _ in range(frames):
        appmod.state.process_frame(image)
    with appmod.state.lock:          # 카메라가 연결된 것처럼 표시
        appmod.state.camera_status = "ONLINE"
        appmod.state.last_frame_time = __import__("time").monotonic()


def test_status_flow_with_target(client):
    assert client.get("/").status_code == 200
    assert client.post("/api/roi", json={"x1": 0.2, "y1": 0.1, "x2": 0.8, "y2": 1.0}).status_code == 200

    feed(5, demo_hand(3))
    data = client.get("/api/status").json()
    assert data["decision"] == "COUNTED" and data["fingers"] == 3

    client.post("/api/target", json={"target": 3})
    assert client.get("/api/status").json()["decision"] == "OK"

    client.post("/api/target", json={"target": 5})
    assert client.get("/api/status").json()["decision"] == "NG"


def test_result_is_stabilized(client):
    client.post("/api/roi", json={"x1": 0.2, "y1": 0.1, "x2": 0.8, "y2": 1.0})
    feed(5, demo_hand(2))
    feed(2, demo_hand(5))            # 2프레임만 바뀌면 아직 그대로
    assert client.get("/api/status").json()["fingers"] == 2
    feed(2, demo_hand(5))            # 최근 5프레임 중 4번 → 바뀜
    assert client.get("/api/status").json()["fingers"] == 5


def test_input_validation(client):
    assert client.post("/api/roi", json={"x1": 0.1, "y1": 0.1, "x2": 0.15, "y2": 0.9}).status_code == 400
    assert client.post("/api/target", json={"target": 6}).status_code == 422
    assert client.post("/api/settings", json={"mode": "rainbow"}).status_code == 422
    assert client.post("/api/camera", json={"url": "http://8.8.8.8:8080/video"}).status_code == 400
    assert client.post("/api/camera", json={"url": "rtsp://192.168.0.2/video"}).status_code == 400


def test_mode_change_resets_result(client):
    client.post("/api/roi", json={"x1": 0.2, "y1": 0.1, "x2": 0.8, "y2": 1.0})
    feed(5, demo_hand(1))
    client.post("/api/settings", json={"mode": "dark_bg"})
    assert client.get("/api/status").json()["fingers"] is None


def test_overlay_frame_has_same_size(client):
    client.post("/api/roi", json={"x1": 0.2, "y1": 0.1, "x2": 0.8, "y2": 1.0})
    frame = demo_hand(5)
    out = appmod.state.process_frame(frame)
    assert out.shape == frame.shape
    assert cv2.imencode(".jpg", out)[0]
