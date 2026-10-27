"""Count raised fingers (0-5) in a single BGR image with classical OpenCV.

Pipeline
    1. Segment skin on the Cr channel of YCrCb (Otsu, with a floor).
    2. Keep the largest external contour as the hand.
    3. Find the palm with a distance transform: the peak is the palm center and
       its value is the palm radius ``r``. Every later rule is relative to ``r``,
       so the logic does not depend on how far the hand is from the camera.
    4. Sample rings at 1.5r-2.3r around the palm. Each contiguous on-hand run on a
       ring is a finger crossing it. A run must also appear on the next ring in
       the same direction (rejects folded knuckles), and runs pointing downward
       are treated as the wrist.

Convexity defects are computed as a second opinion. They cannot tell 0 from 1
(no valleys in either case), so the final count comes from ring sampling.
"""

import math

import cv2
import numpy as np

SEGMENT_MODES = ("skin", "dark_bg", "light_bg")

# Skin pixels sit well above this on the Cr channel. Without a floor, Otsu
# happily splits an empty background into two classes and invents a "hand".
SKIN_CR_MIN = 135
MIN_HAND_AREA_RATIO = 0.03

CIRCLE_SCALES = (1.5, 1.75, 2.0, 2.3)
CIRCLE_SAMPLES = 360
MATCH_ANGLE = 15.0          # max direction change of a finger between two rings (deg)

# Run width relative to r. The upper bound is loose on purpose: a sleeve covering
# the lower palm shrinks r, and real fingers then measure up to ~1.6r.
MIN_FINGER_WIDTH = 0.12
MAX_FINGER_WIDTH = 2.0

# Runs centered in this cone (image y grows downward, 90 deg = straight down) are
# the wrist or forearm. Width alone failed on photos where a sleeve cut the wrist
# diagonally and left a finger-sized strip; direction is the stable signal.
WRIST_CONE = (45.0, 135.0)

DEFECT_MAX_ANGLE = 90.0
DEFECT_MIN_DEPTH = 0.45     # relative to r


def _check_bgr(image: np.ndarray) -> None:
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a non-empty BGR image")


def _odd(value: float, minimum: int = 3) -> int:
    v = max(minimum, int(value))
    return v if v % 2 else v + 1


def _angle_gap(a: float, b: float) -> float:
    gap = abs(a - b) % 360
    return min(gap, 360 - gap)


def segment_hand(roi: np.ndarray, mode: str = "skin") -> tuple[np.ndarray, float]:
    """Return a binary hand mask and the threshold that produced it.

    ``skin`` thresholds the Cr channel, which stays high on skin regardless of
    brightness. ``dark_bg`` / ``light_bg`` fall back to grayscale Otsu for plain
    backgrounds where color is unreliable.
    """
    _check_bgr(roi)
    if mode not in SEGMENT_MODES:
        raise ValueError(f"mode must be one of {SEGMENT_MODES}")

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    if mode == "skin":
        cr = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)[:, :, 1]
        otsu_value, _ = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        value = max(otsu_value, SKIN_CR_MIN)
        _, binary = cv2.threshold(cr, value, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        flag = cv2.THRESH_BINARY if mode == "dark_bg" else cv2.THRESH_BINARY_INV
        value, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)

    # Kernel scales with the image so the same settings work from 480p to 4K.
    k = _odd(min(roi.shape[:2]) / 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned, float(value)


def find_hand(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the largest contour and a filled mask of it, or (None, None)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < mask.size * MIN_HAND_AREA_RATIO:
        return None, None
    hand_mask = np.zeros_like(mask)
    cv2.drawContours(hand_mask, [largest], -1, 255, cv2.FILLED)
    return largest, hand_mask


def palm_circle(hand_mask: np.ndarray) -> tuple[tuple[int, int], float]:
    """Largest inscribed circle of the hand. Fingers and wrist are thinner than
    the palm, so the distance-transform peak lands on the palm."""
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    _, radius, _, center = cv2.minMaxLoc(dist)
    return (int(center[0]), int(center[1])), float(radius)


def count_by_circle(hand_mask: np.ndarray, center: tuple[int, int],
                    palm_radius: float, scale: float) -> tuple[int, list[dict], float]:
    """Find finger-like runs on one ring of radius ``palm_radius * scale``."""
    h, w = hand_mask.shape
    radius = palm_radius * scale
    angles = np.arange(CIRCLE_SAMPLES) * (2 * math.pi / CIRCLE_SAMPLES)
    xs = np.round(center[0] + radius * np.cos(angles)).astype(int)
    ys = np.round(center[1] + radius * np.sin(angles)).astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    on_hand = np.zeros(CIRCLE_SAMPLES, dtype=bool)
    on_hand[inside] = hand_mask[ys[inside], xs[inside]] > 0

    if on_hand.all() or not on_hand.any():
        return 0, [], radius

    # The ring wraps around; start scanning from an off-hand sample so a run
    # crossing angle 0 is not split in two.
    shift = int(np.argmin(on_hand))
    rolled = np.roll(on_hand, -shift)

    arc_per_sample = 2 * math.pi * radius / CIRCLE_SAMPLES
    fingers = []
    i = 0
    while i < CIRCLE_SAMPLES:
        if not rolled[i]:
            i += 1
            continue
        start = i
        while i < CIRCLE_SAMPLES and rolled[i]:
            i += 1
        length = i - start
        width_ratio = length * arc_per_sample / palm_radius
        mid = (start + length / 2 + shift) % CIRCLE_SAMPLES
        mid_deg = mid * 360 / CIRCLE_SAMPLES
        is_wrist = WRIST_CONE[0] <= mid_deg <= WRIST_CONE[1]
        if MIN_FINGER_WIDTH <= width_ratio <= MAX_FINGER_WIDTH and not is_wrist:
            a = mid * 2 * math.pi / CIRCLE_SAMPLES
            fingers.append({
                "x": int(round(center[0] + radius * math.cos(a))),
                "y": int(round(center[1] + radius * math.sin(a))),
                "angle": round(mid_deg, 1),
                "width": round(width_ratio, 3),
            })
    return min(len(fingers), 5), fingers, radius


def count_by_circles(hand_mask: np.ndarray, center: tuple[int, int],
                     palm_radius: float) -> tuple[int, list[dict], float]:
    """Count fingers that cross two neighboring rings in the same direction.

    A single radius does not work for every hand: small rings see fingers
    merged at the base, large rings miss the thumb and pinky. Requiring a run on
    both ring i and ring i+1 also filters folded knuckles, which reach the inner
    ring but not the next one. The best ring pair wins (ties go to the inner one).
    """
    rings = [count_by_circle(hand_mask, center, palm_radius, s) for s in CIRCLE_SCALES]
    best = (0, [], rings[0][2])
    for inner, outer in zip(rings, rings[1:]):
        long_fingers = [p for p in inner[1]
                        if any(_angle_gap(p["angle"], q["angle"]) <= MATCH_ANGLE for q in outer[1])]
        if len(long_fingers) > best[0]:
            best = (min(len(long_fingers), 5), long_fingers, inner[2])
    return best


def count_by_defects(contour: np.ndarray, center: tuple[int, int],
                     palm_radius: float) -> tuple[int | None, list[tuple[int, int]]]:
    """Second opinion: valleys between fingers + 1. Returns None for 0 valleys,
    because a fist and a single finger both have none."""
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 4:
        return None, []
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return None, []
    if defects is None:
        return None, []

    valleys = []
    for s, e, f, depth in defects.reshape(-1, 4):   # (N,1,4) on OpenCV 4, (N,4) on 5
        start, end, far = contour[s][0], contour[e][0], contour[f][0]
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b == 0 or c == 0:
            continue
        cos_angle = np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1, 1)
        angle = math.degrees(math.acos(cos_angle))
        near_wrist = far[1] > center[1] + palm_radius * 0.5
        if angle <= DEFECT_MAX_ANGLE and depth / 256.0 >= DEFECT_MIN_DEPTH * palm_radius \
                and not near_wrist:
            valleys.append((int(far[0]), int(far[1])))

    count = min(len(valleys) + 1, 5) if valleys else None
    return count, valleys


def count_fingers(roi: np.ndarray, mode: str = "skin") -> dict:
    """Run the full pipeline on one image.

    ``status`` is ``COUNTED`` or ``NO_HAND``. ``fingers`` is the ring-sampling
    count used for decisions; ``defect_fingers`` is shown for comparison only.
    """
    mask, threshold = segment_hand(roi, mode)
    contour, hand_mask = find_hand(mask)
    result = {
        "status": "NO_HAND", "fingers": None, "defect_fingers": None,
        "threshold": round(threshold, 1), "area_ratio": 0.0,
        "center": None, "palm_radius": None, "circle_radius": None,
        "finger_points": [], "valleys": [], "contour": None, "mask": mask,
    }
    if contour is None:
        return result

    center, palm_radius = palm_circle(hand_mask)
    if palm_radius < 3:
        return result

    fingers, points, circle_radius = count_by_circles(hand_mask, center, palm_radius)
    defect_fingers, valleys = count_by_defects(contour, center, palm_radius)

    result.update(
        status="COUNTED", fingers=fingers, defect_fingers=defect_fingers,
        area_ratio=round(cv2.contourArea(contour) / mask.size, 4),
        center=center, palm_radius=round(palm_radius, 1),
        circle_radius=round(circle_radius, 1),
        finger_points=points, valleys=valleys, contour=contour, mask=hand_mask,
    )
    return result


def draw_result(image: np.ndarray, result: dict, offset: tuple[int, int] = (0, 0)) -> None:
    """Overlay contour (green), palm (blue), counting ring (yellow), fingers (red)
    and valleys (purple). ``offset`` is the ROI origin inside ``image``."""
    if result["status"] != "COUNTED":
        return
    ox, oy = offset
    contour = result["contour"] + np.array([ox, oy])
    cx, cy = result["center"][0] + ox, result["center"][1] + oy
    cv2.drawContours(image, [contour], -1, (0, 220, 0), 2)
    cv2.circle(image, (cx, cy), int(result["palm_radius"]), (255, 140, 0), 2)
    cv2.circle(image, (cx, cy), int(result["circle_radius"]), (0, 220, 255), 1, cv2.LINE_AA)
    for v in result["valleys"]:
        cv2.circle(image, (v[0] + ox, v[1] + oy), 5, (200, 0, 200), -1)
    for i, p in enumerate(result["finger_points"], start=1):
        pt = (p["x"] + ox, p["y"] + oy)
        cv2.circle(image, pt, 7, (0, 0, 255), -1)
        cv2.putText(image, str(i), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
