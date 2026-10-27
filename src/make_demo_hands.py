"""Synthetic hand generator for deterministic tests.

    python -m src.make_demo_hands   ->  samples/fingers_demo/<count>/01.jpg ... 05.jpg

Each pose (0-5 fingers) is rendered under five shot conditions that vary
position, scale, tilt and exposure. Folded fingers are drawn as short knuckle
bumps so a fist is not a clean ellipse; that bump is exactly what the two-ring
rule in finger_counter.py has to reject.
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "samples" / "fingers_demo"
FRAME_SIZE = (640, 480)

SKIN = (120, 155, 215)          # BGR
BACKGROUND = (205, 200, 190)    # BGR, light gray wall

# (base x, base y, angle deg with 0 = up, length, thickness), all relative to palm radius
# thumb, index, middle, ring, pinky
FINGERS = [
    (-0.85, 0.20, -65, 0.85, 0.22),
    (-0.55, -0.80, -22, 1.10, 0.19),
    (-0.15, -0.95, -6, 1.20, 0.20),
    (0.27, -0.90, 10, 1.10, 0.19),
    (0.65, -0.70, 27, 0.90, 0.17),
]

# Fingers raised for each count (0 = thumb ... 4 = pinky), using the common gestures
POSES = {
    0: [],
    1: [1],
    2: [1, 2],
    3: [1, 2, 3],
    4: [1, 2, 3, 4],
    5: [0, 1, 2, 3, 4],
}

# Shot conditions: (center x, center y, palm radius px, tilt deg, exposure gain)
SHOTS = [
    (320, 300, 62, 0, 1.00),
    (300, 310, 55, -10, 0.90),
    (340, 290, 68, 8, 1.10),
    (310, 320, 50, 15, 0.85),
    (330, 300, 58, -18, 1.05),
]


def draw_hand(fingers_up: list[int], cx: int, cy: int, r: int, tilt: float) -> np.ndarray:
    """Return a binary hand mask (255 = hand)."""
    w, h = FRAME_SIZE
    mask = np.zeros((h, w), np.uint8)
    rad = np.deg2rad(tilt)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    def to_px(dx, dy):
        x, y = rot @ np.array([dx, dy]) * r
        return int(round(cx + x)), int(round(cy + y))

    # Palm (slightly tall ellipse) and a wrist running off the bottom edge
    cv2.ellipse(mask, (cx, cy), (int(r * 1.05), int(r * 1.15)), tilt, 0, 360, 255, -1, cv2.LINE_AA)
    wrist_top = to_px(0, 0.8)
    wrist_bottom = to_px(0, 4.0)
    cv2.line(mask, wrist_top, wrist_bottom, 255, int(r * 1.35), cv2.LINE_AA)

    for idx, (sx, sy, angle, length, thick) in enumerate(FINGERS):
        a = np.deg2rad(angle)
        extended = idx in fingers_up
        ln = (length * 1.6 if extended else 0.45) if idx else (length * 1.5 if extended else 0.4)
        start = np.array([sx, sy])
        end = start + ln * np.array([np.sin(a), -np.cos(a)])
        if idx == 0 and not extended:
            end = start + np.array([0.35, -0.15])        # folded thumb tucks toward the palm
        cv2.line(mask, to_px(*start), to_px(*end), 255, max(3, int(r * thick * 2)), cv2.LINE_AA)
    return mask


def render(mask: np.ndarray, gain: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    bg = np.full((h, w, 3), BACKGROUND, np.float32)
    shade = np.linspace(0.92, 1.05, w, dtype=np.float32)[None, :, None]
    hand = np.full((h, w, 3), SKIN, np.float32) * shade
    alpha = (cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255)[..., None]
    scene = bg * (1 - alpha) + hand * alpha
    scene = scene * gain + rng.normal(0, 4, scene.shape)
    return np.clip(scene, 0, 255).astype(np.uint8)


def generate(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    written = []
    for count, fingers_up in POSES.items():
        folder = output_dir / str(count)
        folder.mkdir(parents=True, exist_ok=True)
        for i, (cx, cy, r, tilt, gain) in enumerate(SHOTS, start=1):
            mask = draw_hand(fingers_up, cx, cy, r, tilt)
            path = folder / f"{i:02d}.jpg"
            cv2.imwrite(str(path), render(mask, gain, count * 10 + i))
            written.append(path)
    return written


if __name__ == "__main__":
    files = generate()
    print(f"wrote {len(files)} images to", OUTPUT_DIR)
