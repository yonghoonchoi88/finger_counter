"""스마트폰 없이 전체 흐름(카메라 → 서버 → 웹 화면)을 확인하는 IP Webcam 시뮬레이터입니다.

samples/fingers_demo 의 가상 손 이미지를 0 → 1 → … → 5 개 순서로 돌려 가며
IP Webcam 앱과 같은 MJPEG 형식(/video)으로 보내 줍니다.

실행 (터미널 2개)
1) python -m src.make_demo_hands             # 처음 한 번
   python -m src.ipwebcam_simulator           # 시뮬레이터 : http://127.0.0.1:8081/video
2) uvicorn src.fastapi_finger_app:app --port 8000
   브라우저 http://localhost:8000 → 카메라 주소에 http://127.0.0.1:8081/video 입력
"""

import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def load_frames(sample_dir: Path) -> list[bytes]:
    frames = []
    for folder in sorted(p for p in sample_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        image = cv2.imread(str(sorted(folder.glob("*.jpg"))[0]))
        if image is not None:
            frames.append(cv2.imencode(".jpg", image)[1].tobytes())
    if not frames:
        raise SystemExit(f"이미지가 없습니다: {sample_dir}  →  python -m src.make_demo_hands 먼저 실행")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=ROOT / "samples" / "fingers_demo")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--seconds", type=float, default=2.0, help="이미지 한 장을 보여 줄 시간")
    args = parser.parse_args()
    frames = load_frames(args.dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/video":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    index = int(time.monotonic() / args.seconds) % len(frames)
                    jpg = frames[index]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    print(f"IP Webcam 시뮬레이터: http://127.0.0.1:{args.port}/video  (Ctrl+C 종료)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
