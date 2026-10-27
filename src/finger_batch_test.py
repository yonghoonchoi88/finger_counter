"""Offline evaluation on a labeled image folder.

Layout (folder name = ground-truth finger count)::

    samples/
    ├── 1/  a.jpg b.jpg ...
    ├── 2/
    └── 5/

The whole image is inspected (no ROI). Phone photos are downscaled so the long
side is at most ``--max-side`` px: full 12 MP frames are slow and do not resemble
the 720p-1080p frames the live system sees.

Examples::

    python -m src.finger_batch_test
    python -m src.finger_batch_test --dir samples/fingers_demo
    python -m src.finger_batch_test --mode light_bg
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src.finger_counter import SEGMENT_MODES, count_fingers, draw_result

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the finger counter on a labeled folder")
    parser.add_argument("--dir", type=Path, default=ROOT / "samples")
    parser.add_argument("--mode", choices=SEGMENT_MODES, default="skin")
    parser.add_argument("--max-side", type=int, default=1280,
                        help="downscale so the long side is at most this many px (0 = keep original)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "finger_batch")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "finger_batch_results.csv")
    return parser.parse_args()


def collect(sample_dir: Path) -> list[tuple[int, Path]]:
    items = []
    for folder in sorted(p for p in sample_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        expected = int(folder.name)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((expected, path))
    return items


def print_confusion(confusion: Counter, labels: list) -> None:
    preds = sorted({p for _, p in confusion}, key=lambda v: (isinstance(v, str), v))
    width = max(len(str(p)) for p in preds) + 2
    print("Confusion matrix (rows = expected, cols = predicted)")
    print(" " * 6 + "".join(f"{str(p):>{width}}" for p in preds))
    for e in labels:
        print(f"{e:>6}" + "".join(f"{confusion[(e, p)] or '.':>{width}}" for p in preds))


def main():
    args = parse_args()
    if not args.dir.exists():
        raise SystemExit(f"sample folder not found: {args.dir}\n"
                         "for synthetic data run: python -m src.make_demo_hands")

    items = collect(args.dir)
    if not items:
        raise SystemExit(f"no images under numbered folders (0-5) in {args.dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    per_class = defaultdict(lambda: [0, 0])     # expected -> [correct, total]
    confusion = Counter()
    for expected, path in items:
        image = cv2.imread(str(path))
        if image is None:
            print("unreadable:", path)
            continue

        longest = max(image.shape[:2])
        if args.max_side and longest > args.max_side:
            scale = args.max_side / longest
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        result = count_fingers(image, args.mode)
        predicted = result["fingers"] if result["status"] == "COUNTED" else "NO_HAND"
        is_correct = predicted == expected
        per_class[expected][0] += is_correct
        per_class[expected][1] += 1
        confusion[(expected, predicted)] += 1

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

    if not rows:
        raise SystemExit("no readable images")

    with args.report.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    print(f"Images:   {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {correct / total:.3f}")
    print()
    for n in sorted(per_class):
        hit, count = per_class[n]
        print(f"  {n}: {hit}/{count}")
    print()
    print_confusion(confusion, sorted(per_class))

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print()
        print("Misclassified")
        for r in wrong:
            print(f"  {r['file']}: predicted {r['predicted']} (run widths {r['finger_widths'] or '-'})")
    print()
    print("CSV:   ", args.report)
    print("Images:", args.out)


if __name__ == "__main__":
    main()
