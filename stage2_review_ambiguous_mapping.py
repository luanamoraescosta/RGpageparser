#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage2_review_ambiguous_mapping.py

Stage 2 of the pipeline. Manual review only -- no Qwen, no API calls.

For every entry flagged in AMBIGUOUS_MAPPING_TSV (stage 1 output), shows
the two candidate page scans side by side and lets you decide which page
the entry really belongs to, always looking at the actual images.

Updates MAPPING_TSV in place with your decisions.

Keys while reviewing:
    a   -> entry belongs to page A (left image)
    b   -> entry belongs to page B (right image)
    n   -> skip / not sure (logged, not applied)
    q   -> quit (saves everything decided so far)

Usage
-----
    python stage2_review_ambiguous_mapping.py
    python stage2_review_ambiguous_mapping.py --limit 20
    python stage2_review_ambiguous_mapping.py --start-at 10
"""

import argparse
import csv
import json
import textwrap
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline_config import (
    IMAGE_DIR,
    DEBUG_DIR,
    MAPPING_TSV,
    AMBIGUOUS_MAPPING_TSV,
    SCAN_REPORT_TSV,
    MAPPING_RESOLUTIONS_JSON,
    REVIEW_RESOLVED_TSV,
    REVIEW_UNSURE_TSV,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def read_scan_report_page_to_image() -> dict[int, str]:
    if not SCAN_REPORT_TSV.exists():
        raise FileNotFoundError(f"Missing scan report: {SCAN_REPORT_TSV}. Run stage1 first.")

    page_to_image: dict[int, str] = {}
    with open(SCAN_REPORT_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            filename = row.get("filename", "").strip()
            final_page = row.get("final_page", "").strip()
            if not filename or not final_page or final_page == "None":
                continue
            try:
                page = int(final_page)
            except Exception:
                continue
            page_to_image.setdefault(page, filename)
    return page_to_image


def read_ambiguous_entries() -> list[dict]:
    if not AMBIGUOUS_MAPPING_TSV.exists():
        raise FileNotFoundError(
            f"Missing ambiguous TSV: {AMBIGUOUS_MAPPING_TSV}. Run stage1 first."
        )

    rows = []
    with open(AMBIGUOUS_MAPPING_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows.append({
                    "missing_entry": int(row["missing_entry"]),
                    "left_entry": int(row["left_entry"]),
                    "left_page": int(row["left_page"]),
                    "right_entry": int(row["right_entry"]),
                    "right_page": int(row["right_page"]),
                })
            except Exception:
                continue
    return rows


def load_mapping_tsv(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry_raw = row.get("entry", "").strip()
            page_raw = row.get("page", "").strip()
            if not entry_raw or not page_raw:
                continue
            mapping[int(entry_raw)] = int(page_raw)
    return mapping


def save_mapping_tsv(path: Path, mapping: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("entry\tpage\n")
        for entry, page in sorted(mapping.items()):
            f.write(f"{entry}\t{page}\n")


def load_existing_resolutions() -> dict[int, int]:
    if not MAPPING_RESOLUTIONS_JSON.exists():
        return {}
    with open(MAPPING_RESOLUTIONS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def save_resolutions(resolutions: dict[int, int]) -> None:
    MAPPING_RESOLUTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_RESOLUTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(resolutions.items())}, f, indent=2, ensure_ascii=False)
    print(f"Saved resolutions -> {MAPPING_RESOLUTIONS_JSON}")


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_display_image(path: Path, max_h: int = 950) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    h, w = img.shape[:2]
    if h > max_h:
        scale = max_h / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def make_side_by_side(img_a: np.ndarray, img_b: np.ndarray, header_lines: list[str]) -> np.ndarray:
    h = max(img_a.shape[0], img_b.shape[0])
    w_a = img_a.shape[1]
    w_b = img_b.shape[1]

    header_h = 160
    canvas = np.full((h + header_h, w_a + w_b + 20, 3), 245, dtype=np.uint8)

    canvas[header_h:header_h + img_a.shape[0], 0:w_a] = img_a
    canvas[header_h:header_h + img_b.shape[0], w_a + 20:w_a + 20 + w_b] = img_b

    cv2.putText(canvas, "A", (20, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(canvas, "B", (w_a + 40, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)

    y = 26
    for line in header_lines:
        for part in textwrap.wrap(line, width=150):
            cv2.putText(canvas, part, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 0, 0), 2, cv2.LINE_AA)
            y += 24

    return canvas


# ── Logs ──────────────────────────────────────────────────────────────────────

def append_resolved_tsv(entry: int, page: int, answer: str,
                         image_a: str, image_b: str, left_page: int, right_page: int) -> None:
    exists = REVIEW_RESOLVED_TSV.exists()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_RESOLVED_TSV, "a", encoding="utf-8") as f:
        if not exists:
            f.write("entry\tpage\tanswer\timage_a\timage_b\tleft_page\tright_page\n")
        f.write(f"{entry}\t{page}\t{answer}\t{image_a}\t{image_b}\t{left_page}\t{right_page}\n")


def append_unsure_tsv(entry: int, image_a: str, image_b: str,
                       left_page: int, right_page: int) -> None:
    exists = REVIEW_UNSURE_TSV.exists()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_UNSURE_TSV, "a", encoding="utf-8") as f:
        if not exists:
            f.write("entry\timage_a\timage_b\tleft_page\tright_page\n")
        f.write(f"{entry}\t{image_a}\t{image_b}\t{left_page}\t{right_page}\n")


# ── Review loop ───────────────────────────────────────────────────────────────

def review_ambiguous_entries(limit: Optional[int] = None, start_at: int = 0) -> None:
    page_to_image = read_scan_report_page_to_image()
    ambiguous = read_ambiguous_entries()
    saved_resolutions = load_existing_resolutions()
    mapping = load_mapping_tsv(MAPPING_TSV)

    if start_at:
        ambiguous = ambiguous[start_at:]
    if limit:
        ambiguous = ambiguous[:limit]

    print(f"Ambiguous rows to review: {len(ambiguous)}")
    print(f"Already resolved entries: {len(saved_resolutions)}")

    cv2.namedWindow("ambiguous-review", cv2.WINDOW_NORMAL)

    for idx, row in enumerate(ambiguous, 1):
        entry = row["missing_entry"]

        if entry in saved_resolutions:
            print(f"[{idx}/{len(ambiguous)}] entry {entry} already resolved, skipping.")
            continue

        left_page = row["left_page"]
        right_page = row["right_page"]
        left_entry = row["left_entry"]
        right_entry = row["right_entry"]

        image_a_name = page_to_image.get(left_page)
        image_b_name = page_to_image.get(right_page)

        if not image_a_name or not image_b_name:
            print(f"[{idx}/{len(ambiguous)}] Missing image for entry {entry}")
            continue

        image_a = IMAGE_DIR / image_a_name
        image_b = IMAGE_DIR / image_b_name

        if not image_a.exists() or not image_b.exists():
            print(f"Missing image file for entry {entry}")
            continue

        print(f"\n[{idx}/{len(ambiguous)}] Entry {entry}")
        print(f"A: page {left_page} image {image_a_name}")
        print(f"B: page {right_page} image {image_b_name}")

        img_a = load_display_image(image_a)
        img_b = load_display_image(image_b)

        header_lines = [
            f"Entry {entry}  (ambiguous between page {left_page} and page {right_page})",
            f"A: page {left_page}, image {image_a_name}, nearby entry {left_entry}",
            f"B: page {right_page}, image {image_b_name}, nearby entry {right_entry}",
            "Keys: a = A | b = B | n = skip/unsure | q = quit",
        ]

        canvas = make_side_by_side(img_a, img_b, header_lines)

        while True:
            cv2.imshow("ambiguous-review", canvas)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("a"):
                mapping[entry] = left_page
                saved_resolutions[entry] = left_page
                save_resolutions(saved_resolutions)
                save_mapping_tsv(MAPPING_TSV, mapping)
                append_resolved_tsv(entry, left_page, "A", image_a_name, image_b_name, left_page, right_page)
                print(f"Accepted: entry {entry} -> page {left_page}")
                break

            elif key == ord("b"):
                mapping[entry] = right_page
                saved_resolutions[entry] = right_page
                save_resolutions(saved_resolutions)
                save_mapping_tsv(MAPPING_TSV, mapping)
                append_resolved_tsv(entry, right_page, "B", image_a_name, image_b_name, left_page, right_page)
                print(f"Accepted: entry {entry} -> page {right_page}")
                break

            elif key == ord("n"):
                append_unsure_tsv(entry, image_a_name, image_b_name, left_page, right_page)
                print(f"Skipped entry {entry}.")
                break

            elif key == ord("q"):
                print("Quitting review.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\nReview complete.")
    print(f"MAPPING_TSV updated -> {MAPPING_TSV}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0)
    args = parser.parse_args()

    review_ambiguous_entries(limit=args.limit, start_at=args.start_at)

    print("\nStage 2 done. Next: stage3_locate_sublemmas.py --review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
