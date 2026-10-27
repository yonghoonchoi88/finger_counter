"""Stand-in for the Android IP Webcam app, for end-to-end runs without a phone.

Serves ``/video`` as an MJPEG stream that cycles through the first image of
each numbered folder. Because the server under test cannot tell it apart from
a real phone, the full path (capture thread -> counter -> dashboard) is
exercised with the same input every time.

    python -m src.ipwebcam_simulator                 # http://127.0.0.1:8081/video
    python -m src.ipwebcam_simulator --dir samples   # replay real photos
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
        raise SystemExit(f"no images in {sample_dir} (run: python -m src.make_demo_hands)")
    return frames


def main():
    parser = argparse.ArgumentParser(description="MJPEG stream that mimics the IP Webcam app")
    parser.add_argument("--dir", type=Path, default=ROOT / "samples" / "fingers_demo")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--seconds", type=float, default=2.0, help="seconds per image")
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

    print(f"IP Webcam simulator on http://127.0.0.1:{args.port}/video (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
