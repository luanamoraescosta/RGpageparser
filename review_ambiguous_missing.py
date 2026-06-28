#!/usr/bin/env python3
"""
Review ambiguous missing entries between pages.

Can work in two modes:

1. Manual-only mode:
   Shows two page images side by side.
   You decide whether the missing entry is on page A or B.

2. Qwen-assisted mode:
   Sends both images to a vision LLM.
   Shows the model answer.
   You can accept it or override manually.

Inputs:
  debug/ambiguous_missing_between_pages.tsv
  debug/scan_report.tsv

Outputs:
  manual_overrides_review.json
  debug/review_resolved_missing.tsv
  debug/review_unsure_missing.tsv

Optional:
  updates XML with:
    <page source="manual-review">N</page>
  directly inside <sublemma>.

Environment variables for Qwen/OpenAI-compatible API:
  GWDG_API_KEY
  GWDG_BASE_URL
  GWDG_MODEL

Example usage:

  # Manual-only, no XML writing
  python3 review_ambiguous_missing.py --no-qwen --no-xml

  # Manual-only, write XML after accepted decisions
  python3 review_ambiguous_missing.py --no-qwen

  # Qwen-assisted, no XML writing
  python3 review_ambiguous_missing.py --use-qwen --no-xml

  # Qwen-assisted, review first 20
  python3 review_ambiguous_missing.py --use-qwen --limit 20
"""

import os
import re
import csv
import json
import base64
import shutil
import argparse
import textwrap
import xml.etree.ElementTree as ET

from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(r"/Users/luanamoraescosta/Downloads/RG-Utils/notebooks/RG8")
IMAGE_DIR = BASE_DIR / "H4 6481 (8,1"

XML_INPUT = BASE_DIR / "rg8_enriched.xml"
XML_OUTPUT = BASE_DIR / "rg8_enriched.xml"

DEBUG_DIR = BASE_DIR / "debug"

AMBIGUOUS_TSV = DEBUG_DIR / "ambiguous_missing_between_pages.tsv"
SCAN_REPORT_TSV = DEBUG_DIR / "scan_report.tsv"

RESOLVED_TSV = DEBUG_DIR / "review_resolved_missing.tsv"
UNSURE_TSV = DEBUG_DIR / "review_unsure_missing.tsv"

REVIEW_OVERRIDES_JSON = BASE_DIR / "manual_overrides_review.json"

PAGE_SOURCE_MANUAL = "manual-review"
PAGE_SOURCE_QWEN = "ocr-qwen-reviewed"


# =============================================================================
# Optional Qwen/OpenAI-compatible config
# =============================================================================

GWDG_API_KEY = os.environ.get("GWDG_API_KEY", "")
GWDG_BASE_URL = os.environ.get("GWDG_BASE_URL", "")
GWDG_MODEL = os.environ.get("GWDG_MODEL", "")


# =============================================================================
# Data loading
# =============================================================================

def read_scan_report_page_to_image() -> dict[int, str]:
    """
    Reads debug/scan_report.tsv and returns:
      {final_page_number: filename}
    """

    if not SCAN_REPORT_TSV.exists():
        raise FileNotFoundError(f"Missing scan report: {SCAN_REPORT_TSV}")

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
    """
    Reads ambiguous_missing_between_pages.tsv.

    Expected columns:
      missing_entry
      left_entry
      left_page
      right_entry
      right_page
      comment
    """

    if not AMBIGUOUS_TSV.exists():
        raise FileNotFoundError(f"Missing ambiguous TSV: {AMBIGUOUS_TSV}")

    rows = []

    with open(AMBIGUOUS_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            try:
                rows.append({
                    "missing_entry": int(row["missing_entry"]),
                    "left_entry": int(row["left_entry"]),
                    "left_page": int(row["left_page"]),
                    "right_entry": int(row["right_entry"]),
                    "right_page": int(row["right_page"]),
                    "comment": row.get("comment", ""),
                })
            except Exception:
                continue

    return rows


def load_existing_resolutions() -> dict[int, int]:
    if not REVIEW_OVERRIDES_JSON.exists():
        return {}

    with open(REVIEW_OVERRIDES_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {int(k): int(v) for k, v in raw.items()}


def save_resolutions(resolutions: dict[int, int]) -> None:
    with open(REVIEW_OVERRIDES_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in sorted(resolutions.items())},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved overrides → {REVIEW_OVERRIDES_JSON}")


# =============================================================================
# Image helpers
# =============================================================================

def image_to_data_url(path: Path, max_side: int = 1800, quality: int = 85) -> str:
    img = Image.open(path).convert("RGB")

    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)

    if scale < 1.0:
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS,
        )

    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64}"


def load_display_image(path: Path, max_h: int = 950) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    h, w = img.shape[:2]

    if h > max_h:
        scale = max_h / h
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return img


def make_side_by_side(
    img_a: np.ndarray,
    img_b: np.ndarray,
    header_lines: list[str],
) -> np.ndarray:
    h = max(img_a.shape[0], img_b.shape[0])
    w_a = img_a.shape[1]
    w_b = img_b.shape[1]

    header_h = 175

    canvas = np.full(
        (h + header_h, w_a + w_b + 20, 3),
        245,
        dtype=np.uint8,
    )

    canvas[header_h:header_h + img_a.shape[0], 0:w_a] = img_a
    canvas[header_h:header_h + img_b.shape[0], w_a + 20:w_a + 20 + w_b] = img_b

    cv2.putText(
        canvas,
        "A",
        (20, header_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "B",
        (w_a + 40, header_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )

    y = 26

    for line in header_lines:
        wrapped = textwrap.wrap(line, width=150)

        for part in wrapped:
            cv2.putText(
                canvas,
                part,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            y += 24

    return canvas


# =============================================================================
# Qwen / OpenAI-compatible client
# =============================================================================

def get_qwen_client():
    from openai import OpenAI

    if not GWDG_API_KEY:
        raise RuntimeError("GWDG_API_KEY is not set.")

    if not GWDG_BASE_URL:
        raise RuntimeError("GWDG_BASE_URL is not set.")

    if not GWDG_MODEL:
        raise RuntimeError("GWDG_MODEL is not set.")

    return OpenAI(
        api_key=GWDG_API_KEY,
        base_url=GWDG_BASE_URL,
    )


def parse_qwen_answer(text: str) -> tuple[str, str]:
    clean = text.strip()
    lines = [line.strip() for line in clean.splitlines() if line.strip()]

    if not lines:
        return "UNSURE", ""

    first = lines[0].upper()

    m = re.search(r"\b(A|B|BOTH|NONE|UNSURE)\b", first)

    if m:
        answer = m.group(1)
    else:
        m2 = re.search(r"\b(A|B|BOTH|NONE|UNSURE)\b", clean.upper())
        answer = m2.group(1) if m2 else "UNSURE"

    reason = " ".join(lines[1:]).strip()

    return answer, reason


def ask_qwen_entry_location(
    entry: int,
    image_a: Path,
    image_b: Path,
    left_page: int,
    right_page: int,
    left_entry: int,
    right_entry: int,
) -> tuple[str, str, str]:
    """
    Ask model whether entry appears in image A or B.
    """

    client = get_qwen_client()

    prompt = f"""
You are checking scanned pages of a historical printed book.

Task:
Find whether the entry number "{entry}" appears as a lemma/entry number on either image.

Important layout information:
- Each book page usually has two text columns.
- Entry numbers are printed at the start of a lemma.
- They usually appear in the left margin or at the left edge of a column.
- The entry number is followed by a personal/place name, often in bold.
- Ignore page numbers at the bottom.
- Ignore dates.
- Ignore archive references such as "L 547", "S 520", "V 498", "IE 443".
- Ignore numbers inside the running text.
- Only decide where the marginal/heading entry number "{entry}" appears.

Images:
- Image A corresponds to book page {left_page}, near known entry {left_entry}.
- Image B corresponds to book page {right_page}, near known entry {right_entry}.

Answer with exactly one of these on the first line:
A
B
BOTH
NONE
UNSURE

On the second line, give a very short reason.
""".strip()

    response = client.chat.completions.create(
        model=GWDG_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_a),
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_b),
                        },
                    },
                ],
            }
        ],
        temperature=0,
    )

    raw_text = response.choices[0].message.content.strip()

    answer, reason = parse_qwen_answer(raw_text)

    return answer, reason, raw_text


# =============================================================================
# Logs
# =============================================================================

def append_resolved_tsv(
    entry: int,
    page: int,
    method: str,
    answer: str,
    reason: str,
    image_a: str,
    image_b: str,
    left_page: int,
    right_page: int,
) -> None:
    exists = RESOLVED_TSV.exists()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESOLVED_TSV, "a", encoding="utf-8") as f:
        if not exists:
            f.write(
                "entry\tpage\tmethod\tanswer\treason\t"
                "image_a\timage_b\tleft_page\tright_page\n"
            )

        f.write(
            f"{entry}\t{page}\t{method}\t{answer}\t{reason}\t"
            f"{image_a}\t{image_b}\t{left_page}\t{right_page}\n"
        )


def append_unsure_tsv(
    entry: int,
    method: str,
    answer: str,
    reason: str,
    image_a: str,
    image_b: str,
    left_page: int,
    right_page: int,
) -> None:
    exists = UNSURE_TSV.exists()
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    with open(UNSURE_TSV, "a", encoding="utf-8") as f:
        if not exists:
            f.write(
                "entry\tmethod\tanswer\treason\t"
                "image_a\timage_b\tleft_page\tright_page\n"
            )

        f.write(
            f"{entry}\t{method}\t{answer}\t{reason}\t"
            f"{image_a}\t{image_b}\t{left_page}\t{right_page}\n"
        )


# =============================================================================
# XML update
# =============================================================================

def update_xml_with_resolutions(resolutions: dict[int, tuple[int, str]]) -> None:
    """
    resolutions:
      entry -> (page, source)
    """

    if not resolutions:
        print("No resolutions to write to XML.")
        return

    ET.register_namespace("xi", "http://www.w3.org/2001/XInclude")

    tree = ET.parse(str(XML_INPUT))
    root = tree.getroot()

    updated = 0
    no_sublemma = []

    for lemma in root.iter("lemma"):
        lemma_id = lemma.get("id", "")

        m = re.search(r"(\d{4})$", lemma_id)

        if not m:
            continue

        entry_num = int(m.group(1))

        if entry_num not in resolutions:
            continue

        page, source = resolutions[entry_num]

        sublemma = lemma.find("./reg/sublemma")

        if sublemma is None:
            sublemma = lemma.find(".//sublemma")

        if sublemma is None:
            no_sublemma.append(entry_num)
            continue

        page_el = sublemma.find("page")

        if page_el is None:
            page_el = ET.Element("page")
            sublemma.insert(0, page_el)

        page_el.text = str(page)
        page_el.set("source", source)

        reg = lemma.find("reg")

        if reg is not None:
            old_page = reg.find("page")

            if old_page is not None:
                reg.remove(old_page)

        updated += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if XML_OUTPUT.exists():
        backup = XML_OUTPUT.with_name(
            XML_OUTPUT.stem + f"_backup_before_review_{timestamp}" + XML_OUTPUT.suffix
        )
        shutil.copy2(XML_OUTPUT, backup)
        print(f"Backup created → {backup}")

    tmp = XML_OUTPUT.with_suffix(".tmp.xml")

    tree.write(
        str(tmp),
        encoding="utf-8",
        xml_declaration=True,
    )

    tmp.replace(XML_OUTPUT)

    print("\n=== XML UPDATED WITH REVIEW RESOLUTIONS ===")
    print(f"Updated entries : {updated}")
    print(f"No sublemma     : {len(no_sublemma)}")
    print(f"Output          : {XML_OUTPUT}")


# =============================================================================
# Review loop
# =============================================================================

def review_ambiguous_entries(
    use_qwen: bool = False,
    limit: Optional[int] = None,
    start_at: int = 0,
    apply_xml_at_end: bool = True,
) -> None:
    page_to_image = read_scan_report_page_to_image()
    ambiguous = read_ambiguous_entries()
    saved_resolutions = load_existing_resolutions()

    if start_at:
        ambiguous = ambiguous[start_at:]

    if limit:
        ambiguous = ambiguous[:limit]

    print(f"Ambiguous rows to review: {len(ambiguous)}")
    print(f"Already resolved entries: {len(saved_resolutions)}")
    print(f"Qwen enabled: {use_qwen}")

    session_xml_resolutions: dict[int, tuple[int, str]] = {}

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
            print(
                f"[{idx}/{len(ambiguous)}] Missing image for entry {entry}: "
                f"p.{left_page}={image_a_name}, p.{right_page}={image_b_name}"
            )
            continue

        image_a = IMAGE_DIR / image_a_name
        image_b = IMAGE_DIR / image_b_name

        if not image_a.exists() or not image_b.exists():
            print(f"Missing image file for entry {entry}")
            continue

        qwen_answer = "DISABLED"
        qwen_reason = ""

        if use_qwen:
            print(f"\n[{idx}/{len(ambiguous)}] Asking Qwen for entry {entry}...")

            try:
                qwen_answer, qwen_reason, raw = ask_qwen_entry_location(
                    entry=entry,
                    image_a=image_a,
                    image_b=image_b,
                    left_page=left_page,
                    right_page=right_page,
                    left_entry=left_entry,
                    right_entry=right_entry,
                )
            except Exception as exc:
                qwen_answer = "UNSURE"
                qwen_reason = f"API error: {exc}"
                print(f"Qwen error: {exc}")

        print(f"\n[{idx}/{len(ambiguous)}] Entry {entry}")
        print(f"A: page {left_page} image {image_a_name}")
        print(f"B: page {right_page} image {image_b_name}")
        print(f"Qwen: {qwen_answer} {qwen_reason}")

        img_a = load_display_image(image_a)
        img_b = load_display_image(image_b)

        header_lines = [
            f"Entry {entry}",
            f"A: page {left_page}, image {image_a_name}, nearby entry {left_entry}",
            f"B: page {right_page}, image {image_b_name}, nearby entry {right_entry}",
            f"Qwen: {qwen_answer} — {qwen_reason}",
            "Keys: a choose A | b choose B | y accept Qwen | n skip | q quit",
        ]

        canvas = make_side_by_side(img_a, img_b, header_lines)

        while True:
            cv2.imshow("ambiguous-review", canvas)

            key = cv2.waitKey(0) & 0xFF

            chosen_page = None
            method = None
            answer = None
            source = None

            if key == ord("a"):
                chosen_page = left_page
                method = "manual"
                answer = "A"
                source = PAGE_SOURCE_MANUAL

            elif key == ord("b"):
                chosen_page = right_page
                method = "manual"
                answer = "B"
                source = PAGE_SOURCE_MANUAL

            elif key == ord("y"):
                if not use_qwen:
                    print("Qwen disabled; use 'a' or 'b'.")
                    continue

                if qwen_answer == "A":
                    chosen_page = left_page
                    method = "qwen-reviewed"
                    answer = "A"
                    source = PAGE_SOURCE_QWEN

                elif qwen_answer == "B":
                    chosen_page = right_page
                    method = "qwen-reviewed"
                    answer = "B"
                    source = PAGE_SOURCE_QWEN

                else:
                    print(f"Cannot accept Qwen answer: {qwen_answer}")
                    append_unsure_tsv(
                        entry=entry,
                        method="qwen",
                        answer=qwen_answer,
                        reason=qwen_reason,
                        image_a=image_a_name,
                        image_b=image_b_name,
                        left_page=left_page,
                        right_page=right_page,
                    )
                    break

            elif key == ord("n"):
                append_unsure_tsv(
                    entry=entry,
                    method="manual",
                    answer=qwen_answer,
                    reason=qwen_reason,
                    image_a=image_a_name,
                    image_b=image_b_name,
                    left_page=left_page,
                    right_page=right_page,
                )
                print(f"Skipped entry {entry}.")
                break

            elif key == ord("q"):
                print("Quitting review.")
                cv2.destroyAllWindows()

                if saved_resolutions:
                    save_resolutions(saved_resolutions)

                if session_xml_resolutions and apply_xml_at_end:
                    update_xml_with_resolutions(session_xml_resolutions)

                return

            else:
                continue

            if chosen_page is not None:
                saved_resolutions[entry] = chosen_page
                save_resolutions(saved_resolutions)

                session_xml_resolutions[entry] = (chosen_page, source)

                append_resolved_tsv(
                    entry=entry,
                    page=chosen_page,
                    method=method,
                    answer=answer,
                    reason=qwen_reason,
                    image_a=image_a_name,
                    image_b=image_b_name,
                    left_page=left_page,
                    right_page=right_page,
                )

                print(f"Accepted: entry {entry} -> page {chosen_page}")
                break

    cv2.destroyAllWindows()

    if session_xml_resolutions and apply_xml_at_end:
        update_xml_with_resolutions(session_xml_resolutions)

    print("\nReview complete.")
    print(f"Session accepted: {len(session_xml_resolutions)}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use-qwen", action="store_true")
    parser.add_argument("--no-qwen", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0)

    parser.add_argument("--no-xml", action="store_true")

    args = parser.parse_args()

    use_qwen = bool(args.use_qwen and not args.no_qwen)

    review_ambiguous_entries(
        use_qwen=use_qwen,
        limit=args.limit,
        start_at=args.start_at,
        apply_xml_at_end=not args.no_xml,
    )


if __name__ == "__main__":
    main()