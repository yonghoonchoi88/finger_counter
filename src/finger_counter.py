"""병뚜껑 검사 구조를 재사용해 '펴고 있는 손가락 개수(0~5)'를 세는 핵심 로직입니다.

병뚜껑 검사(cap_inspector.py)와 비교
- 병뚜껑 : Gray → Otsu → Morphology → 가장 큰 Contour → Area Ratio → OK/NG
- 손가락 : 피부색(YCrCb) → Otsu → Morphology → 가장 큰 Contour(손)
           → 손바닥 중심(Distance Transform) → 손바닥 둘레 원과 손이 겹치는 구간 수 = 손가락 수
           (참고값으로 Convexity Defects '손가락 사이 골짜기 수'도 함께 계산)
"""


# ============================================================
# ① 필요한 라이브러리
# ============================================================

import math

import cv2
import numpy as np


# ============================================================
# ② 검사 기준값
# - SEGMENT_MODES         : 손을 흰색으로 만드는 이진화 방식
#     skin     : 피부색 (YCrCb 의 Cr 채널 + Otsu)  ← 기본값
#     dark_bg  : 어두운 배경 위 밝은 손 (Gray + Otsu, THRESH_BINARY)
#     light_bg : 밝은 배경 위 어두운 손 (Gray + Otsu, THRESH_BINARY_INV, 병뚜껑과 같은 방식)
# - SKIN_CR_MIN           : 피부의 Cr 최솟값. Otsu 기준값이 이보다 낮으면 이 값을 씁니다.
#                           (손이 없을 때 Otsu 가 배경을 억지로 둘로 나누는 것 방지)
# - MIN_HAND_AREA_RATIO   : ROI 면적의 3% 이상인 Contour 만 손으로 봅니다.
# - CIRCLE_SCALES         : 손바닥 반지름 × 이 배수 크기의 원들을 그려 손가락을 셉니다.
# - MIN/MAX_FINGER_WIDTH  : 원 위에서 손가락 하나로 인정할 폭 (손바닥 반지름 대비)
#                           너무 좁으면 노이즈, 너무 넓으면 손목으로 봅니다.
# - DEFECT_MAX_ANGLE      : 손가락 사이 골짜기의 최대 각도(도)
# - DEFECT_MIN_DEPTH      : 골짜기 최소 깊이 (손바닥 반지름 대비)
# ============================================================

SEGMENT_MODES = ("skin", "dark_bg", "light_bg")
SKIN_CR_MIN = 135
MIN_HAND_AREA_RATIO = 0.03

CIRCLE_SCALES = (1.5, 1.75, 2.0, 2.3)
CIRCLE_SAMPLES = 360
MIN_FINGER_WIDTH = 0.12
MAX_FINGER_WIDTH = 0.95

DEFECT_MAX_ANGLE = 90.0
DEFECT_MIN_DEPTH = 0.45


def _check_bgr(image: np.ndarray) -> None:
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ROI는 비어 있지 않은 BGR 이미지여야 합니다.")


def _odd(value: float, minimum: int = 3) -> int:
    v = max(minimum, int(value))
    return v if v % 2 else v + 1


# ============================================================
# ③ 손 영역 이진화 (2일차 Binary 응용)
# - skin : BGR → YCrCb 변환 후 Cr(붉은 기) 채널에 Otsu 를 적용합니다.
#          피부는 밝기(Y)와 상관없이 Cr 값이 높아서 조명 변화에 강합니다.
# - Morphology 커널 크기는 ROI 크기에 맞춰 자동으로 정합니다.
#   Opening : 작은 노이즈 제거 / Closing : 손 안쪽 구멍 메우기
# - 반환 : (이진 마스크, 사용한 기준값)
# ============================================================

def segment_hand(roi: np.ndarray, mode: str = "skin") -> tuple[np.ndarray, float]:
    _check_bgr(roi)
    if mode not in SEGMENT_MODES:
        raise ValueError(f"mode 는 {SEGMENT_MODES} 중 하나여야 합니다.")

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

    k = _odd(min(roi.shape[:2]) / 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned, float(value)


# ============================================================
# ④ 손 Contour 찾기 (3일차 Contour 응용)
# - 가장 큰 외곽 Contour 를 손으로 봅니다.
# - 그 Contour 만 채운 마스크를 따로 만들어 작은 잡음 덩어리를 없앱니다.
# ============================================================

def find_hand(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < mask.size * MIN_HAND_AREA_RATIO:
        return None, None
    hand_mask = np.zeros_like(mask)
    cv2.drawContours(hand_mask, [largest], -1, 255, cv2.FILLED)
    return largest, hand_mask


# ============================================================
# ⑤ 손바닥 중심과 반지름 (Distance Transform)
# - 손 마스크의 각 흰 픽셀에서 가장 가까운 검은 픽셀까지 거리를 계산합니다.
# - 거리가 가장 큰 점 = 손에서 가장 '두꺼운' 곳 = 손바닥 중심
# - 그 거리 = 손바닥에 꼭 맞게 들어가는 원의 반지름
# - 손가락·손목은 손바닥보다 가늘어서 중심이 손바닥에 잡힙니다.
# ============================================================

def palm_circle(hand_mask: np.ndarray) -> tuple[tuple[int, int], float]:
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    _, radius, _, center = cv2.minMaxLoc(dist)
    return (int(center[0]), int(center[1])), float(radius)


# ============================================================
# ⑥ 원 샘플링으로 손가락 세기 (주 판정 방법)
# - 손바닥 중심에 반지름 R = 손바닥 반지름 × scale 인 원을 그립니다.
# - 원 위의 점 360개가 손(흰색) 위에 있는지 확인합니다.
# - 연속으로 손 위에 있는 구간 하나 = 원을 가로지르는 손가락 하나
# - 구간 폭이 너무 좁으면 노이즈, 너무 넓으면(손바닥 반지름의 0.95배 초과) 손목으로 보고 제외합니다.
# - 아래쪽(60°~120°, 이미지 좌표는 y가 아래로 커짐)을 향한 넓은 구간도 손목으로 봅니다.
# ============================================================

def count_by_circle(hand_mask: np.ndarray, center: tuple[int, int],
                    palm_radius: float, scale: float) -> tuple[int, list[dict], float]:
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

    # 원은 처음과 끝이 이어져 있으므로, 손 밖인 점에서 시작하도록 회전합니다.
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
        points_down = 60 <= mid_deg <= 120
        is_wrist = width_ratio > MAX_FINGER_WIDTH or (points_down and width_ratio > 0.6)
        if MIN_FINGER_WIDTH <= width_ratio and not is_wrist:
            a = mid * 2 * math.pi / CIRCLE_SAMPLES
            fingers.append({
                "x": int(round(center[0] + radius * math.cos(a))),
                "y": int(round(center[1] + radius * math.sin(a))),
                "angle": round(mid_deg, 1),
                "width": round(width_ratio, 3),
            })
    return min(len(fingers), 5), fingers, radius


# ============================================================
# ⑥-2 원 크기를 바꿔 가며 세고, '길게 뻗은' 손가락만 인정
# - 원이 너무 작으면 손가락 뿌리에서 붙어 있어 1개로 합쳐지고,
#   원이 너무 크면 짧은 손가락(새끼·엄지)이 원에 닿지 않습니다.
#   그래서 1.5 / 1.75 / 2.0 / 2.3배 원을 모두 확인합니다.
# - 주먹을 쥐면 접힌 손가락 마디가 작은 원에 살짝 걸립니다.
#   펴진 손가락은 길어서 바로 바깥 원에도 같은 방향(±15°)으로 걸리므로,
#   '안쪽 원과 바로 바깥 원에 모두 걸린 구간'만 손가락으로 셉니다.
# - 원 쌍마다 센 개수 중 가장 큰 값을 사용합니다. (같으면 안쪽 원 우선)
# ============================================================

MATCH_ANGLE = 15.0


def _angle_gap(a: float, b: float) -> float:
    gap = abs(a - b) % 360
    return min(gap, 360 - gap)


def count_by_circles(hand_mask: np.ndarray, center: tuple[int, int],
                     palm_radius: float) -> tuple[int, list[dict], float]:
    rings = [count_by_circle(hand_mask, center, palm_radius, s) for s in CIRCLE_SCALES]
    best = (0, [], rings[0][2])
    for inner, outer in zip(rings, rings[1:]):
        long_fingers = [p for p in inner[1]
                        if any(_angle_gap(p["angle"], q["angle"]) <= MATCH_ANGLE for q in outer[1])]
        if len(long_fingers) > best[0]:
            best = (min(len(long_fingers), 5), long_fingers, inner[2])
    return best


# ============================================================
# ⑦ Convexity Defects 로 손가락 사이 골짜기 세기 (참고값)
# - Convex Hull(손을 감싸는 고무줄)과 손 Contour 사이의 움푹한 곳 = Defect
# - 각도가 90° 보다 좁고, 깊이가 충분한 골짜기만 손가락 사이로 인정합니다.
# - 손목 근처(손바닥 중심보다 아래) 골짜기는 제외합니다.
# - 골짜기 N개 → 손가락 N+1개. 골짜기가 0개면 0개인지 1개인지 구분할 수 없어 None.
# ============================================================

def count_by_defects(contour: np.ndarray, center: tuple[int, int],
                     palm_radius: float) -> tuple[int | None, list[tuple[int, int]]]:
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
    for s, e, f, depth in defects.reshape(-1, 4):
        start, end, far = contour[s][0], contour[e][0], contour[f][0]
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b == 0 or c == 0:
            continue
        cos_angle = np.clip((b ** 2 + c ** 2 - a ** 2) / (2 * b * c), -1, 1)
        angle = math.degrees(math.acos(cos_angle))
        if angle <= DEFECT_MAX_ANGLE and depth / 256.0 >= DEFECT_MIN_DEPTH * palm_radius \
                and far[1] <= center[1] + palm_radius * 0.5:
            valleys.append((int(far[0]), int(far[1])))

    count = min(len(valleys) + 1, 5) if valleys else None
    return count, valleys


# ============================================================
# ⑧ ROI 한 장을 검사하는 최종 함수
# - status
#   NO_HAND : ROI 안에서 손을 찾지 못함
#   COUNTED : 손가락 개수 계산 완료
# - fingers : 원 샘플링 결과(주 판정), defect_fingers : 참고값
# ============================================================

def count_fingers(roi: np.ndarray, mode: str = "skin") -> dict:
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


# ============================================================
# ⑨ 결과 그리기 (실시간 화면 · Batch 결과 이미지 공용)
# - 초록 : 손 Contour / 파랑 : 손바닥 원 / 노랑 : 손가락 세는 원
# - 빨간 점 : 센 손가락 / 보라 점 : 손가락 사이 골짜기
# - offset : ROI 를 원본 프레임에 그릴 때 ROI 왼쪽 위 좌표
# ============================================================

def draw_result(image: np.ndarray, result: dict, offset: tuple[int, int] = (0, 0)) -> None:
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
