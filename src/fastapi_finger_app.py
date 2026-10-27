"""스마트폰 IP Webcam → OpenCV 손가락 세기 → FastAPI 실시간 웹 화면.

병뚜껑 검사(fastapi_cap_app.py)와 같은 구조입니다.
- 바뀐 점 1 : 기준 이미지(cap_on/cap_off) 대신, 손 이진화 방식(피부색/어두운 배경/밝은 배경)을 고릅니다.
- 바뀐 점 2 : OK/NG 대신 손가락 개수(0~5)를 보여 줍니다.
              '목표 개수'를 지정하면 그 개수일 때 OK, 아니면 NG 입니다.
- 바뀐 점 3 : 손가락을 펴는 도중에 숫자가 흔들리지 않도록
              최근 5프레임 중 3번 이상 같은 개수가 나와야 화면 숫자를 바꿉니다.

실행:  uvicorn src.fastapi_finger_app:app --host 0.0.0.0 --port 8000
"""

# ① 라이브러리 · 경로
from collections import Counter, deque
from contextlib import asynccontextmanager
import ipaddress
from pathlib import Path
import threading
import time
from typing import Literal
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.finger_counter import count_fingers, draw_result

ROOT = Path(__file__).resolve().parents[1]
HTML_FILE = ROOT / "web" / "finger_inspection.html"
STATIC_DIR = ROOT / "web" / "static"      # style.css, app.js
IMAGE_DIR = ROOT / "images"                # 동작 원리 그림 (README 와 같은 파일)

HISTORY_SIZE = 5      # 최근 몇 프레임을 볼지
STABLE_COUNT = 3      # 그중 몇 번 같아야 화면 숫자를 바꿀지


# ② 대기 화면 이미지
def placeholder_jpg() -> bytes:
    frame = np.full((480, 640, 3), (38, 43, 50), dtype=np.uint8)
    cv2.putText(frame, "IP WEBCAM - WAITING", (65, 227),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2)
    return cv2.imencode(".jpg", frame)[1].tobytes()


WAITING_JPG = placeholder_jpg()

DECISION_COLORS = {"OK": (0, 190, 0), "NG": (0, 0, 225), "COUNTED": (0, 190, 0)}


# ③ 실시간 상태를 관리하는 클래스
class LiveState:
    """한 대의 카메라를 단일 스레드에서 읽고 여러 브라우저에 공유합니다."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.url: str | None = None
        self.roi: tuple[float, float, float, float] | None = None
        self.mode = "skin"
        self.target: int | None = None
        self.show_mask = True
        self.camera_status = "NOT_CONNECTED"
        self.message = "스마트폰 IP Webcam 주소를 연결하세요."
        self.result = self._empty_result("WAIT")
        self.history: deque = deque(maxlen=HISTORY_SIZE)
        self.stable_key: tuple = ("WAIT", None)
        self.latest_jpg: bytes = WAITING_JPG
        self.last_frame_time = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    @staticmethod
    def _empty_result(status: str) -> dict:
        return {"status": status, "fingers": None, "defect_fingers": None,
                "threshold": None, "area_ratio": None, "palm_radius": None}

    def _reset_result(self) -> None:
        self.result = self._empty_result("WAIT")
        self.history.clear()
        self.stable_key = ("WAIT", None)

    # ④ 시작·종료
    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)

    # ⑤ 카메라 · ROI · 설정 변경
    def set_camera(self, url: str) -> None:
        with self.lock:
            self.url = url
            self.roi = None
            self.camera_status = "CONNECTING"
            self.message = "카메라 연결 중입니다."
            self.latest_jpg = WAITING_JPG
            self.last_frame_time = 0.0
            self._reset_result()

    def set_roi(self, roi: tuple[float, float, float, float] | None) -> None:
        with self.lock:
            self.roi = roi
            self._reset_result()

    def set_settings(self, mode: str | None, show_mask: bool | None) -> None:
        with self.lock:
            if mode is not None and mode != self.mode:
                self.mode = mode
                self._reset_result()
            if show_mask is not None:
                self.show_mask = show_mask

    def set_target(self, target: int | None) -> None:
        with self.lock:
            self.target = target

    # ⑥ 화면 판정 : 손가락 수 + 목표 개수 비교
    def _decision(self, status: str, fingers: int | None) -> str:
        if status != "COUNTED":
            return "CHECK" if status == "NO_HAND" else "WAIT"
        if self.target is None:
            return "COUNTED"
        return "OK" if fingers == self.target else "NG"

    def get_status(self) -> dict:
        with self.lock:
            elapsed = (time.monotonic() - self.last_frame_time
                       if self.last_frame_time else None)
            camera_status = self.camera_status
            if camera_status == "ONLINE" and elapsed is not None and elapsed > 5:
                camera_status = "DISCONNECTED"
            result = self.result if camera_status == "ONLINE" else self._empty_result("WAIT")
            return {
                "camera_status": camera_status,
                "message": self.message,
                "roi": self.roi,
                "mode": self.mode,
                "target": self.target,
                "show_mask": self.show_mask,
                "decision": self._decision(result["status"], result["fingers"]),
                **result,
            }

    # ⑦ 프레임 한 장 처리 (카메라 없이도 테스트할 수 있도록 분리)
    # - ROI Crop → count_fingers() → 결과 안정화 → 화면 표시용 이미지 반환
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            roi_coords = self.roi
            mode = self.mode
            show_mask = self.show_mask

        h, w = frame.shape[:2]
        image = frame.copy()
        result = self._empty_result("WAIT")
        found = None

        if roi_coords is None:
            cv2.putText(image, "SELECT ROI ON WEB PAGE", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
        else:
            nx1, ny1, nx2, ny2 = roi_coords
            x1, x2 = max(0, round(nx1 * w)), min(w, round(nx2 * w))
            y1, y2 = max(0, round(ny1 * h)), min(h, round(ny2 * h))
            if x2 > x1 and y2 > y1:
                try:
                    found = count_fingers(frame[y1:y2, x1:x2], mode)
                    result = {k: found[k] for k in self._empty_result("WAIT")}
                    draw_result(image, found, (x1, y1))
                except (ValueError, cv2.error):
                    result = self._empty_result("NO_HAND")

        with self.lock:
            # 처리하는 동안 ROI/방식이 바뀌었다면 이 프레임 결과는 버립니다.
            if self.roi != roi_coords or self.mode != mode:
                return image
            key = (result["status"], result["fingers"])
            self.history.append(key)
            top_key, count = Counter(self.history).most_common(1)[0]
            if count >= STABLE_COUNT and top_key == key:
                self.stable_key = key          # 3번 이상 같은 결과 → 화면 숫자 변경
                self.result = result
            elif key == self.stable_key:
                self.result = result           # 같은 개수면 세부 값만 갱신
            decision = self._decision(*self.stable_key)
            stable_fingers = self.stable_key[1]

        if roi_coords is not None:
            color = DECISION_COLORS.get(decision, (0, 200, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            # 이진화 결과(손 마스크)를 오른쪽 위에 작게 보여 줍니다. (2일차 Binary 확인용)
            if show_mask and found is not None and found["mask"] is not None:
                thumb_w = max(80, w // 4)
                mh, mw = found["mask"].shape
                thumb_h = max(1, int(mh * thumb_w / mw))
                if thumb_h < h - 20:
                    thumb = cv2.resize(found["mask"], (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
                    image[10:10 + thumb_h, w - thumb_w - 10:w - 10] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(image, (w - thumb_w - 10, 10), (w - 10, 10 + thumb_h), (0, 230, 255), 1)

        label = decision if stable_fingers is None else f"{decision}  FINGERS: {stable_fingers}"
        cv2.putText(image, label, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 245, 255), 2)
        return image

    # ⑧ 카메라 스레드 : 프레임 읽기 → process_frame → JPEG 저장
    def camera_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                url = self.url
            if not url:
                self.stop_event.wait(0.1)
                continue

            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cap.release()
                with self.lock:
                    if self.url == url:
                        self.camera_status = "DISCONNECTED"
                        self.message = "IP Webcam 연결 실패: URL과 Wi-Fi를 확인하세요."
                self.stop_event.wait(1.0)
                continue

            try:
                while not self.stop_event.is_set():
                    with self.lock:
                        if self.url != url:
                            break
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        with self.lock:
                            if self.url == url:
                                self.camera_status = "DISCONNECTED"
                                self.message = "프레임 수신 중단: 재연결을 시도합니다."
                                self.latest_jpg = WAITING_JPG
                                self._reset_result()
                        break

                    image = self.process_frame(frame)
                    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 77])
                    if success:
                        with self.lock:
                            if self.url == url:
                                self.latest_jpg = encoded.tobytes()
                                self.camera_status = "ONLINE"
                                self.message = "영상 수신 중"
                                self.last_frame_time = time.monotonic()
            finally:
                cap.release()
            self.stop_event.wait(0.1)


# ⑨ 서버 시작·종료 (Lifespan)
state = LiveState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.start()
    yield
    state.stop()


app = FastAPI(title="손가락 개수 실시간 검사", lifespan=lifespan)

# web/static 폴더의 CSS·JS 파일을 /static/... 주소로 제공합니다.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# images 폴더의 동작 원리 그림을 /assets/... 주소로 제공합니다. (웹 화면 ④ 섹션)
app.mount("/assets", StaticFiles(directory=IMAGE_DIR), name="assets")


# ⑩ 요청 Body 검증 (Pydantic)
class CameraRequest(BaseModel):
    url: str


class ROIRequest(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class SettingsRequest(BaseModel):
    mode: Literal["skin", "dark_bg", "light_bg"] | None = None
    show_mask: bool | None = None


class TargetRequest(BaseModel):
    target: int | None = Field(default=None, ge=0, le=5)


# ⑪ 화면 · 카메라 · ROI (병뚜껑 앱과 동일)
@app.get("/")
def index():
    return FileResponse(HTML_FILE, media_type="text/html")


@app.post("/api/camera")
def connect_camera(body: CameraRequest):
    url = body.url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise HTTPException(400, "http(s)://IP:PORT/video 형식으로 입력하세요.")
    if parts.username or parts.password:
        raise HTTPException(400, "사용자 인증이 없는 교육용 IP Webcam 주소만 사용하세요.")
    try:
        addr = ipaddress.ip_address(parts.hostname)
        if not (addr.is_private or addr.is_loopback):
            raise ValueError("사설 IP 주소가 아닙니다")
        _ = parts.port
    except ValueError:
        raise HTTPException(400, "스마트폰의 사설 IP 주소(예: 192.168.x.x)를 사용하세요.")
    state.set_camera(url)
    return {"ok": True, "message": "연결을 시도합니다."}


@app.post("/api/roi")
def set_roi(body: ROIRequest):
    if body.x2 - body.x1 < 0.1 or body.y2 - body.y1 < 0.1:
        raise HTTPException(400, "ROI가 너무 작습니다. 손 전체가 들어가도록 넓게 드래그하세요.")
    state.set_roi((body.x1, body.y1, body.x2, body.y2))
    return {"ok": True, "roi": body.model_dump()}


@app.delete("/api/roi")
def reset_roi():
    state.set_roi(None)
    return {"ok": True}


# ⑫ 이진화 방식 · 마스크 보기 · 목표 개수
@app.post("/api/settings")
def set_settings(body: SettingsRequest):
    state.set_settings(body.mode, body.show_mask)
    return {"ok": True}


@app.post("/api/target")
def set_target(body: TargetRequest):
    state.set_target(body.target)
    return {"ok": True, "target": body.target}


# ⑬ 상태 조회 · MJPEG 영상
@app.get("/api/status")
def get_status():
    return state.get_status()


@app.get("/stream.mjpg")
def stream_video():
    def frames():
        while True:
            with state.lock:
                image_bytes = state.latest_jpg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-cache\r\n\r\n"
                   + image_bytes + b"\r\n")
            time.sleep(0.10)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )
