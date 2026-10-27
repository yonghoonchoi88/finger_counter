"""폴더에 모아 둔 손 사진으로 손가락 세기 정확도를 측정합니다. (22_cap_batch_test.py 의 손가락 버전)

폴더 구조 (폴더 이름 = 정답 손가락 수)
samples/fingers/
├── 0/  01.jpg 02.jpg ...
├── 1/  ...
└── 5/  ...

- 병뚜껑과 달리 Reference 이미지가 필요 없습니다. (규칙만으로 개수를 셈)
- 사진 전체를 ROI 로 사용하므로 손이 화면 대부분을 차지하게 찍으세요.

실행 예
python -m src.finger_batch_test                            # samples/fingers
python -m src.finger_batch_test --dir samples/fingers_demo # 가상 데모 손
python -m src.finger_batch_test --mode light_bg            # 밝은 배경 · 어두운 손
"""

# ① 라이브러리
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from src.finger_counter import SEGMENT_MODES, count_fingers, draw_result

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="손가락 개수 Batch Test")
    parser.add_argument("--dir", type=Path, default=ROOT / "samples" / "fingers")
    parser.add_argument("--mode", choices=SEGMENT_MODES, default="skin")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "finger_batch")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "finger_batch_results.csv")
    return parser.parse_args()


# ② 정답 폴더(0~5)별 이미지 목록
def collect(sample_dir: Path) -> list[tuple[int, Path]]:
    items = []
    for folder in sorted(p for p in sample_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        expected = int(folder.name)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((expected, path))
    return items


def main():
    args = parse_args()
    if not args.dir.exists():
        raise SystemExit(f"샘플 폴더가 없습니다: {args.dir}\n"
                         "데모로 확인하려면: python -m src.make_demo_hands 후 --dir samples/fingers_demo")

    items = collect(args.dir)
    if not items:
        raise SystemExit("0~5 이름의 폴더에 이미지를 넣으세요.")

    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # ③ 모든 이미지에 같은 규칙 적용 · 정답 비교
    rows, total, correct = [], 0, 0
    per_class = defaultdict(lambda: [0, 0])          # 정답 개수별 [맞힌 수, 전체 수]
    for expected, path in items:
        image = cv2.imread(str(path))
        if image is None:
            print("읽기 실패:", path)
            continue

        result = count_fingers(image, args.mode)
        predicted = result["fingers"] if result["status"] == "COUNTED" else "NO_HAND"
        is_correct = predicted == expected
        total += 1
        correct += is_correct
        per_class[expected][0] += is_correct
        per_class[expected][1] += 1

        # ④ 결과 이미지 저장 (Contour · 손바닥 원 · 센 손가락 표시)
        view = image.copy()
        draw_result(view, result)
        cv2.putText(view, f"expected {expected} / predicted {predicted}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0) if is_correct else (0, 0, 255), 2)
        cv2.imwrite(str(args.out / f"{expected}_{path.stem}_result.jpg"), view)

        rows.append({
            "file": f"{expected}/{path.name}",
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "defect_fingers": result["defect_fingers"],
            "threshold": result["threshold"],
            "area_ratio": result["area_ratio"],
            "palm_radius": result["palm_radius"],
            "finger_widths": " ".join(str(p["width"]) for p in result["finger_points"]),
        })

    if total == 0:
        raise SystemExit("검사할 이미지가 없습니다.")

    # ⑤ CSV 보고서
    with args.report.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ⑥ 결과 출력 : 전체 정확도 · 개수별 정확도 · 틀린 이미지
    print("==============================")
    print("Finger Batch Test 결과")
    print("==============================")
    print("확인 이미지 수:", total)
    print("정답:", correct)
    print("오답:", total - correct)
    print("Accuracy:", round(correct / total, 3))
    print()
    print("손가락 수별 정확도")
    for n in sorted(per_class):
        hit, count = per_class[n]
        print(f"  {n}개: {hit}/{count}")
    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print()
        print("틀린 이미지")
        for r in wrong:
            print(f"  ✗ {r['file']}: {r['predicted']}개로 판정 (손가락 폭 {r['finger_widths'] or '-'})")
    print()
    print("CSV:", args.report)
    print("결과 이미지:", args.out)


if __name__ == "__main__":
    main()
