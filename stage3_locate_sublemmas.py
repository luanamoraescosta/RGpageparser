#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage3_locate_sublemmas.py

Stage 3 of the pipeline.

For each ordered page scan:
  1. OCR the page, split into left/right columns.
  2. Take the last N lines from the right column (default 2).
  3. Fuzzy-match those lines against candidate <head>/<sublemma> texts from
     XML_INPUT, restricted to a small window of entries around the last
     header known for this page (from MAPPING_TSV, stage 1+2 output).
  4. Record WHERE inside the winning candidate's text the OCR matched
     (text_position: "starts-here" / "ends-here" / "complete-on-page" /
     "continues"), which stage 4 uses to reconstruct page_start/page_end.

Manual review (--review): always shows the actual scan image (with the
matched lines highlighted) alongside the OCR-vs-XML text comparison, no
external viewer needed and no Qwen/API calls.

Usage
-----
    python stage3_locate_sublemmas.py --test100 --review
    python stage3_locate_sublemmas.py --review
    python stage3_locate_sublemmas.py --limit 100
"""

import argparse
import csv
import difflib
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from pipeline_config import (
    TESSERACT_CMD,
    IMAGE_DIR,
    XML_INPUT,
    MAPPING_TSV,
    DEBUG_DIR,
    DEBUG_IMG_DIR,
    MATCHES_TSV,
    MATCHES_REVIEWED_TSV,
    MANUAL_CORRECTIONS_TSV,
    FIRST_SCAN_NO,
    FIRST_LOGICAL_PAGE,
    IMAGE_EXTENSIONS,
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ── OCR/matching configuration ────────────────────────────────────────────────

MIN_CONF = 20

DEFAULT_LAST_LINES = 2
DEFAULT_BACK_ENTRIES = 1
DEFAULT_FORWARD_ENTRIES = 1
DEFAULT_THRESHOLD = 55.0

LINE_Y_TOLERANCE = 24
AMBIGUOUS_SCORE_DELTA = 0.5

TEXT_POSITION_START_THRESHOLD = 0.12
TEXT_POSITION_END_THRESHOLD = 0.90

REVIEW_IMG_MAX_H = 950


try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OcrToken:
    text: str
    conf: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class OcrLine:
    text: str
    x: int
    y: int
    w: int
    h: int
    col: int


@dataclass
class SublemmaDoc:
    entry: int
    sub_id: str
    xml_text: str
    norm_text: str
    is_head: bool = False


@dataclass
class PageMatch:
    page: int
    scan: str
    entry: Optional[int]
    sub_id: Optional[str]
    score: float
    status: str
    candidate_entries: list[int]
    ocr_last_lines: str
    xml_preview: str
    matched_is_head: bool = False
    detected_header_entry: Optional[int] = None
    reviewed: bool = False
    review_note: str = ""
    match_rel_start: float = 0.0
    match_rel_end: float = 0.0
    text_position: str = ""
    last_line_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    split_x: int = 0


# ── Text normalization ────────────────────────────────────────────────────────

def normalize_for_match(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ſ", "s").replace("°", "r")
    text = re.sub(r"[\"'`\u2018\u2019\u201c\u201d´]", "v", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_preview(text: str, max_len: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text


def sub_index(sub_id: str) -> int:
    m = re.search(r"-(\d+)$", sub_id or "")
    return int(m.group(1)) if m else 0


# ── Load mapping ──────────────────────────────────────────────────────────────

def load_mapping_tsv(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry_raw = row.get("entry", "").strip()
            page_raw = row.get("page", "").strip()
            if not entry_raw or not page_raw:
                continue
            mapping[int(entry_raw)] = int(page_raw)
    return mapping


# ── Load XML: heads + sublemmas ───────────────────────────────────────────────

def element_text_without_page_tags(el: ET.Element) -> str:
    skip_tags = {"page", "page_start", "page_end"}
    parts: list[str] = []

    def walk(node: ET.Element):
        if node.tag in skip_tags:
            return
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return " ".join(parts)


def load_sublemmas(xml_path: Path) -> list[SublemmaDoc]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    docs: list[SublemmaDoc] = []

    for lemma in root.iter("lemma"):
        m = re.search(r"(\d{4})$", lemma.get("id", ""))
        if not m:
            continue
        entry = int(m.group(1))

        head = lemma.find(".//head")
        if head is not None:
            head_id = head.get("id", "").strip() or f"{lemma.get('id', '')}-0"
            head_text = element_text_without_page_tags(head)
            norm_head = normalize_for_match(head_text)
            if norm_head:
                docs.append(SublemmaDoc(entry=entry, sub_id=head_id, xml_text=head_text,
                                         norm_text=norm_head, is_head=True))

        for sub in lemma.findall(".//sublemma"):
            sub_id = sub.get("id", "").strip()
            if not sub_id:
                continue
            xml_text = element_text_without_page_tags(sub)
            norm_text = normalize_for_match(xml_text)
            if not norm_text:
                continue
            docs.append(SublemmaDoc(entry=entry, sub_id=sub_id, xml_text=xml_text,
                                     norm_text=norm_text, is_head=False))

    return docs


# ── Ordered scans ─────────────────────────────────────────────────────────────

def scan_number(path: Path) -> Optional[int]:
    m = re.search(r"_(\d+)(?=\.[^.]+$)", path.name)
    return int(m.group(1)) if m else None


def ordered_scan_paths(image_dir: Path, limit: Optional[int] = None) -> list[Path]:
    paths: list[Path] = []
    for p in image_dir.iterdir():
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        n = scan_number(p)
        if n is None or n < FIRST_SCAN_NO:
            continue
        paths.append(p)
    paths.sort(key=lambda p: scan_number(p) or 0)
    if limit is not None:
        paths = paths[:limit]
    return paths


# ── Image/OCR helpers ─────────────────────────────────────────────────────────

def safe_conf(v) -> int:
    try:
        return int(float(v))
    except Exception:
        return -1


def load_image(path: Path) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def ocr_tokens(img: np.ndarray) -> list[OcrToken]:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, config="--psm 6 --oem 3", output_type=pytesseract.Output.DICT)
    tokens: list[OcrToken] = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf = safe_conf(data["conf"][i])
        if not text or conf < MIN_CONF:
            continue
        tokens.append(OcrToken(text=text, conf=conf, x=int(data["left"][i]), y=int(data["top"][i]),
                                w=int(data["width"][i]), h=int(data["height"][i])))
    return tokens


def find_column_split(tokens: list[OcrToken], img_w: int) -> int:
    if not tokens:
        return img_w // 2
    centers = sorted(t.x + t.w / 2 for t in tokens if img_w * 0.10 <= t.x + t.w / 2 <= img_w * 0.90)
    if len(centers) < 20:
        return img_w // 2
    best_gap, best_split = 0.0, img_w // 2
    for a, b in zip(centers, centers[1:]):
        gap = b - a
        mid = (a + b) / 2
        if img_w * 0.35 <= mid <= img_w * 0.65 and gap > best_gap:
            best_gap, best_split = gap, int(mid)
    return best_split if best_gap >= 80 else img_w // 2


def group_tokens_into_lines_manual(tokens: list[OcrToken], col: int, img_h: int,
                                    line_y_tolerance: int = LINE_Y_TOLERANCE) -> list[OcrLine]:
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: (t.y + t.h / 2, t.x))
    groups: list[list[OcrToken]] = []
    for tok in toks:
        cy = tok.y + tok.h / 2
        if not groups:
            groups.append([tok])
            continue
        last_cy = sum(t.y + t.h / 2 for t in groups[-1]) / len(groups[-1])
        if abs(cy - last_cy) <= line_y_tolerance:
            groups[-1].append(tok)
        else:
            groups.append([tok])

    lines: list[OcrLine] = []
    for group in groups:
        group = sorted(group, key=lambda t: t.x)
        text = " ".join(t.text for t in group).strip()
        if not text:
            continue
        x1 = min(t.x for t in group)
        y1 = min(t.y for t in group)
        x2 = max(t.x + t.w for t in group)
        y2 = max(t.y + t.h for t in group)
        if re.fullmatch(r"\d{1,4}", text) and y1 > img_h * 0.85:
            continue
        if len(normalize_for_match(text)) < 2:
            continue
        lines.append(OcrLine(text=text, x=x1, y=y1, w=x2 - x1, h=y2 - y1, col=col))
    lines.sort(key=lambda ln: (ln.y, ln.x))
    return lines


def tokens_to_lines(tokens: list[OcrToken], img_w: int, img_h: int) -> tuple[list[OcrLine], int]:
    if not tokens:
        return [], img_w // 2
    split_x = find_column_split(tokens, img_w)
    left_tokens = [t for t in tokens if t.x + t.w / 2 < split_x]
    right_tokens = [t for t in tokens if t.x + t.w / 2 >= split_x]
    left_lines = group_tokens_into_lines_manual(left_tokens, col=0, img_h=img_h)
    right_lines = group_tokens_into_lines_manual(right_tokens, col=1, img_h=img_h)
    return left_lines + right_lines, split_x


def entries_near_page(page: int, mapping: dict[int, int], window: int = 1) -> set[int]:
    return {entry for entry, p in mapping.items() if abs(p - page) <= window}


def detect_trailing_header_entry(lines: list[OcrLine], page: int, mapping: dict[int, int],
                                  max_lines_from_bottom: int = 2, page_window: int = 1) -> Optional[int]:
    if not lines:
        return None
    near_entries = entries_near_page(page, mapping, window=page_window)
    if not near_entries:
        return None
    start = max(0, len(lines) - max_lines_from_bottom)
    for idx in range(len(lines) - 1, start - 1, -1):
        text = lines[idx].text.strip()
        m = re.match(r"^(\d{1,5})\s+\S+", text)
        if not m:
            continue
        entry_num = int(m.group(1))
        if entry_num in near_entries:
            return entry_num
    return None


def get_last_ocr_lines(lines: list[OcrLine], n: int) -> list[OcrLine]:
    if not lines:
        return []
    right_lines = [ln for ln in lines if ln.col == 1]
    left_lines = [ln for ln in lines if ln.col == 0]
    source_lines = right_lines or left_lines or lines
    return source_lines[-n:]


# ── Candidate selection anchored on last header of the page ──────────────────

def sorted_entries_from_mapping(mapping: dict[int, int]) -> list[int]:
    return sorted(mapping.keys())


def last_header_entry_for_page(page: int, mapping: dict[int, int]) -> Optional[int]:
    entries_on_page = sorted(entry for entry, p in mapping.items() if p == page)
    if entries_on_page:
        return max(entries_on_page)
    previous_entries = sorted(entry for entry, p in mapping.items() if p < page)
    return max(previous_entries) if previous_entries else None


def candidate_entries_for_page(page: int, mapping: dict[int, int],
                                back_entries: int = DEFAULT_BACK_ENTRIES,
                                forward_entries: int = DEFAULT_FORWARD_ENTRIES) -> list[int]:
    entries = sorted_entries_from_mapping(mapping)
    if not entries:
        return []
    target = last_header_entry_for_page(page, mapping)
    if target is None:
        return []
    if target not in entries:
        return [target]
    idx = entries.index(target)
    start = max(0, idx - back_entries)
    end = min(len(entries), idx + forward_entries + 1)
    return entries[start:end]


def candidate_sublemmas_for_page(page: int, sublemmas: list[SublemmaDoc], mapping: dict[int, int],
                                  back_entries: int, forward_entries: int) -> tuple[list[int], list[SublemmaDoc]]:
    entries = candidate_entries_for_page(page, mapping, back_entries, forward_entries)
    entry_set = set(entries)
    candidates = [sub for sub in sublemmas if sub.entry in entry_set]
    return entries, candidates


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def fuzzy_score(ocr_text: str, xml_text: str) -> float:
    ocr_norm = normalize_for_match(ocr_text)
    xml_norm = normalize_for_match(xml_text)
    if not ocr_norm or not xml_norm:
        return 0.0
    if HAVE_RAPIDFUZZ:
        return float(fuzz.partial_ratio(ocr_norm, xml_norm))
    if len(xml_norm) <= len(ocr_norm):
        return 100.0 * difflib.SequenceMatcher(None, ocr_norm, xml_norm).ratio()
    win = max(len(ocr_norm), 50)
    step = max(10, win // 4)
    best = 0.0
    for start in range(0, max(1, len(xml_norm) - win + 1), step):
        chunk = xml_norm[start:start + win]
        score = 100.0 * difflib.SequenceMatcher(None, ocr_norm, chunk).ratio()
        if score > best:
            best = score
    return best


def find_match_position(ocr_text: str, xml_text: str) -> tuple[float, float]:
    """
    Find WHERE inside xml_text the OCR text best aligns, as fractions
    (0.0-1.0) of the normalized xml_text length.

    rel_end close to 1.0   -> OCR matched near the END of the candidate
                               text -> this page likely contains the block's
                               final part.
    rel_start close to 0.0 -> OCR matched near the START -> this page likely
                               contains the block's first part.
    Both low/high           -> whole block fits on this single page.
    Neither                 -> OCR matched a MIDDLE portion -> the block
                               spans pages before and after this one too.
    """
    ocr_norm = normalize_for_match(ocr_text)
    xml_norm = normalize_for_match(xml_text)

    if not ocr_norm or not xml_norm:
        return 0.0, 0.0

    xml_len = len(xml_norm)
    if xml_len == 0:
        return 0.0, 0.0

    if HAVE_RAPIDFUZZ:
        try:
            alignment = fuzz.partial_ratio_alignment(ocr_norm, xml_norm)
        except Exception:
            alignment = None

        if alignment is not None:
            if len(ocr_norm) <= len(xml_norm):
                rel_start = alignment.dest_start / xml_len
                rel_end = alignment.dest_end / xml_len
            else:
                rel_start, rel_end = 0.0, 1.0

            return max(0.0, min(1.0, rel_start)), max(0.0, min(1.0, rel_end))

    if xml_len <= len(ocr_norm):
        return 0.0, 1.0

    win = max(len(ocr_norm), 50)
    step = max(10, win // 4)

    best_score = -1.0
    best_start, best_end = 0, win

    for start in range(0, max(1, xml_len - win + 1), step):
        chunk = xml_norm[start:start + win]
        score = difflib.SequenceMatcher(None, ocr_norm, chunk).ratio()
        if score > best_score:
            best_score = score
            best_start, best_end = start, start + win

    return best_start / xml_len, min(1.0, best_end / xml_len)


def classify_text_position(rel_start: float, rel_end: float) -> str:
    starts_near_beginning = rel_start <= TEXT_POSITION_START_THRESHOLD
    ends_near_end = rel_end >= TEXT_POSITION_END_THRESHOLD

    if starts_near_beginning and ends_near_end:
        return "complete-on-page"
    if ends_near_end:
        return "ends-here"
    if starts_near_beginning:
        return "starts-here"
    return "continues"


def ranked_candidates(ocr_text: str, candidates: list[SublemmaDoc]) -> list[tuple[float, SublemmaDoc]]:
    scored: list[tuple[float, SublemmaDoc]] = []
    for sub in candidates:
        score = fuzzy_score(ocr_text, sub.xml_text)
        scored.append((score, sub))
    scored.sort(
        key=lambda item: (item[0], item[1].entry, sub_index(item[1].sub_id)),
        reverse=True,
    )
    return scored


def match_page_to_sublemma(
    page: int,
    scan_name: str,
    ocr_last_text: str,
    candidates: list[SublemmaDoc],
    candidate_entries: list[int],
    threshold: float,
    detected_header_entry: Optional[int],
) -> PageMatch:
    if not candidates:
        return PageMatch(page=page, scan=scan_name, entry=None, sub_id=None, score=0.0,
                          status="no-candidates", candidate_entries=candidate_entries,
                          ocr_last_lines=ocr_last_text, xml_preview="",
                          detected_header_entry=detected_header_entry)

    if len(normalize_for_match(ocr_last_text)) < 10:
        return PageMatch(page=page, scan=scan_name, entry=None, sub_id=None, score=0.0,
                          status="ocr-too-short", candidate_entries=candidate_entries,
                          ocr_last_lines=ocr_last_text, xml_preview="",
                          detected_header_entry=detected_header_entry)

    scored = ranked_candidates(ocr_last_text, candidates)
    if not scored:
        return PageMatch(page=page, scan=scan_name, entry=None, sub_id=None, score=0.0,
                          status="no-match", candidate_entries=candidate_entries,
                          ocr_last_lines=ocr_last_text, xml_preview="",
                          detected_header_entry=detected_header_entry)

    best_score, best_sub = scored[0]
    status = "ok" if best_score >= threshold else "low-score"
    review_note = ""

    if len(scored) > 1:
        second_score, second_sub = scored[1]
        if abs(best_score - second_score) <= AMBIGUOUS_SCORE_DELTA:
            if detected_header_entry is not None and {best_sub.entry, second_sub.entry} & {detected_header_entry}:
                status = f"ambiguous-header-{detected_header_entry}"
            else:
                status = "ambiguous"

    if detected_header_entry is not None:
        if best_sub.entry == detected_header_entry and best_sub.is_head:
            review_note = f"correctly matched head of entry {detected_header_entry}"
        elif best_sub.entry == detected_header_entry:
            review_note = f"matched sublemma of entry {detected_header_entry} (not head)"
        else:
            review_note = (
                f"header-like line for entry {detected_header_entry} detected, "
                f"but best match is entry {best_sub.entry}"
            )

    rel_start, rel_end = find_match_position(ocr_last_text, best_sub.xml_text)
    text_position = classify_text_position(rel_start, rel_end)

    return PageMatch(
        page=page, scan=scan_name, entry=best_sub.entry, sub_id=best_sub.sub_id,
        score=best_score, status=status, candidate_entries=candidate_entries,
        ocr_last_lines=ocr_last_text, xml_preview=clean_preview(best_sub.xml_text),
        matched_is_head=best_sub.is_head, detected_header_entry=detected_header_entry,
        review_note=review_note, match_rel_start=rel_start, match_rel_end=rel_end,
        text_position=text_position,
    )


# ── Debug images ──────────────────────────────────────────────────────────────

def save_debug_image(img: np.ndarray, page: int, scan_name: str, all_lines: list[OcrLine],
                      last_lines: list[OcrLine], match: PageMatch, split_x: int) -> Path:
    DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    draw = img.copy()
    h, _ = draw.shape[:2]

    cv2.line(draw, (split_x, 0), (split_x, h), (0, 180, 0), 2)

    for ln in all_lines:
        cv2.rectangle(draw, (ln.x, ln.y), (ln.x + ln.w, ln.y + ln.h), (255, 180, 80), 1)

    for ln in last_lines:
        cv2.rectangle(draw, (ln.x, ln.y), (ln.x + ln.w, ln.y + ln.h), (0, 0, 255), 3)

    head_tag = " [HEAD]" if match.matched_is_head else ""
    label = (
        f"page={page} score={match.score:.1f} entry={match.entry} sub={match.sub_id}{head_tag} "
        f"pos={match.text_position}({match.match_rel_start:.2f}-{match.match_rel_end:.2f}) "
        f"status={match.status}"
    )
    cv2.putText(draw, label[:160], (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)

    safe_scan = re.sub(r"[^\w.-]+", "_", scan_name)
    img_out = DEBUG_IMG_DIR / f"page_{page:04d}_{safe_scan}.jpg"
    cv2.imwrite(str(img_out), draw)
    return img_out


def make_review_image(img: np.ndarray, last_lines: list[OcrLine], split_x: int,
                       header_lines: list[str], max_h: int = REVIEW_IMG_MAX_H) -> np.ndarray:
    """
    Build the image shown during interactive review: the scan with the
    matched last lines highlighted, plus a text header on top with page,
    status, score and candidate info. This always shows the real scan, no
    external image viewer needed.
    """
    draw = img.copy()
    h, w = draw.shape[:2]

    cv2.line(draw, (split_x, 0), (split_x, h), (0, 180, 0), 2)
    for ln in last_lines:
        cv2.rectangle(draw, (ln.x, ln.y), (ln.x + ln.w, ln.y + ln.h), (0, 0, 255), 4)

    if h > max_h:
        scale = max_h / h
        draw = cv2.resize(draw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    header_h = 26 * (len(header_lines) + 1) + 20
    canvas = np.full((draw.shape[0] + header_h, draw.shape[1], 3), 245, dtype=np.uint8)
    canvas[header_h:, :] = draw

    y = 24
    for line in header_lines:
        import textwrap
        for part in textwrap.wrap(line, width=130):
            cv2.putText(canvas, part, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            y += 24

    return canvas


# ── Reports ──────────────────────────────────────────────────────────────────

def save_matches_report(matches: list[PageMatch], out_path: Path) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "page\tentry\tsub_id\tis_head\tscore\tstatus\tscan\t"
            "text_position\tmatch_rel_start\tmatch_rel_end\t"
            "candidate_entries\tdetected_header_entry\treviewed\treview_note\t"
            "ocr_last_lines\txml_sublemma_preview\n"
        )
        for m in matches:
            f.write(
                f"{m.page}\t{m.entry or ''}\t{m.sub_id or ''}\t{int(m.matched_is_head)}\t"
                f"{m.score:.2f}\t{m.status}\t{m.scan}\t{m.text_position}\t"
                f"{m.match_rel_start:.3f}\t{m.match_rel_end:.3f}\t"
                f"{'|'.join(map(str, m.candidate_entries))}\t{m.detected_header_entry or ''}\t"
                f"{int(m.reviewed)}\t{m.review_note}\t"
                f"{clean_preview(m.ocr_last_lines, 900)}\t{clean_preview(m.xml_preview, 900)}\n"
            )
    print(f"Report saved: {out_path}")
    return out_path


def append_manual_correction(match: PageMatch) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    exists = MANUAL_CORRECTIONS_TSV.exists()
    with open(MANUAL_CORRECTIONS_TSV, "a", encoding="utf-8") as f:
        if not exists:
            f.write(
                "page\tentry\tsub_id\tis_head\tscore\tstatus\tscan\t"
                "text_position\tmatch_rel_start\tmatch_rel_end\t"
                "candidate_entries\tdetected_header_entry\treview_note\tocr_last_lines\n"
            )
        f.write(
            f"{match.page}\t{match.entry or ''}\t{match.sub_id or ''}\t{int(match.matched_is_head)}\t"
            f"{match.score:.2f}\t{match.status}\t{match.scan}\t{match.text_position}\t"
            f"{match.match_rel_start:.3f}\t{match.match_rel_end:.3f}\t"
            f"{'|'.join(map(str, match.candidate_entries))}\t{match.detected_header_entry or ''}\t"
            f"{match.review_note}\t{clean_preview(match.ocr_last_lines, 900)}\n"
        )


# ── Interactive review (manual, always shows the image) ───────────────────────

def needs_review(match: PageMatch) -> bool:
    return match.status != "ok"


def interactive_review(matches: list[PageMatch], sublemmas: list[SublemmaDoc],
                        scan_to_path: dict[str, Path], top_n: int = 15) -> None:
    review_items = [m for m in matches if needs_review(m)]

    if not review_items:
        print("\nNo cases need review.")
        return

    print("\n" + "=" * 80)
    print("INTERACTIVE REVIEW (image window will open)")
    print("=" * 80)
    print("Terminal commands (type in terminal, image window stays open):")
    print("  number       choose candidate number")
    print("  k            keep automatic choice")
    print("  n            mark as no-sublemma/header-only")
    print("  m SUB_ID     manually set sublemma/head id")
    print("  s            skip for now")
    print("  q            quit review")
    print("=" * 80)

    sub_by_id = {s.sub_id: s for s in sublemmas}

    cv2.namedWindow("sublemma-review", cv2.WINDOW_NORMAL)

    for idx, match in enumerate(review_items, start=1):
        candidates = [s for s in sublemmas if s.entry in set(match.candidate_entries)]
        scored = ranked_candidates(match.ocr_last_lines, candidates)

        path = scan_to_path.get(match.scan)
        img = None
        last_lines: list[OcrLine] = []
        split_x = 0

        if path is not None:
            try:
                img = deskew(load_image(path))
                img_h, img_w = img.shape[:2]
                tokens = ocr_tokens(img)
                lines, split_x = tokens_to_lines(tokens, img_w, img_h)
                last_lines = get_last_ocr_lines(lines, len(match.ocr_last_lines.split()) or 2)
                # Fall back to re-detecting last N lines consistent with match text
                last_lines = [ln for ln in lines if ln.text in match.ocr_last_lines] or last_lines
            except Exception as exc:
                print(f"Could not reload image for review: {exc}")

        while True:
            head_tag = " [HEAD]" if match.matched_is_head else ""

            header_lines = [
                f"Case {idx}/{len(review_items)}   page={match.page}   scan={match.scan}",
                f"status={match.status}   score={match.score:.2f}",
                f"auto entry/sub: {match.entry} / {match.sub_id}{head_tag}",
                f"candidate entries: {'|'.join(map(str, match.candidate_entries))}",
                f"text position: {match.text_position} "
                f"({match.match_rel_start:.1%} - {match.match_rel_end:.1%} of candidate text)",
            ]
            if match.detected_header_entry is not None:
                header_lines.append(f"header-like line detected for entry: {match.detected_header_entry}")
            if match.review_note:
                header_lines.append(f"note: {match.review_note}")
            header_lines.append(f"OCR lines: {clean_preview(match.ocr_last_lines, 160)}")

            if img is not None:
                canvas = make_review_image(img, last_lines, split_x, header_lines)
                cv2.imshow("sublemma-review", canvas)
                cv2.waitKey(1)  # refresh window, non-blocking

            print("\n" + "-" * 80)
            for line in header_lines:
                print(line)

            print("\nTop candidates:")
            for n, (score, sub) in enumerate(scored[:top_n], start=1):
                marker = ""
                if sub.sub_id == match.sub_id:
                    marker += "  [AUTO]"
                if sub.is_head:
                    marker += "  [HEAD]"
                r_start, r_end = find_match_position(match.ocr_last_lines, sub.xml_text)
                pos_label = classify_text_position(r_start, r_end)
                print(f"{n:>2}. score={score:>6.2f} entry={sub.entry:<6} sub={sub.sub_id}{marker} "
                      f"pos={pos_label}({r_start:.1%}-{r_end:.1%})")
                print(f"    {clean_preview(sub.xml_text, 220)}")

            cmd = input("\nChoice [number/k/n/m SUB_ID/s/q]: ").strip()
            if not cmd:
                continue

            if cmd.lower() == "q":
                print("Stopping review.")
                cv2.destroyAllWindows()
                save_matches_report(matches, MATCHES_REVIEWED_TSV)
                return

            if cmd.lower() == "s":
                print("Skipped.")
                break

            if cmd.lower() == "k":
                match.reviewed = True
                match.review_note = (match.review_note + " | kept automatic").strip(" |")
                append_manual_correction(match)
                print("Kept automatic choice.")
                break

            if cmd.lower() == "n":
                match.entry = None
                match.sub_id = None
                match.score = 0.0
                match.status = "manual-no-sublemma-or-header-only"
                match.reviewed = True
                match.review_note = "marked no sublemma/header only"
                match.xml_preview = ""
                match.matched_is_head = False
                match.match_rel_start = 0.0
                match.match_rel_end = 0.0
                match.text_position = ""
                append_manual_correction(match)
                print("Marked as no sublemma/header only.")
                break

            if cmd.lower().startswith("m "):
                manual_sub_id = cmd.split(maxsplit=1)[1].strip()
                sub = sub_by_id.get(manual_sub_id)
                if not sub:
                    print(f"Unknown sub_id: {manual_sub_id}")
                    continue
                r_start, r_end = find_match_position(match.ocr_last_lines, sub.xml_text)
                match.entry = sub.entry
                match.sub_id = sub.sub_id
                match.score = fuzzy_score(match.ocr_last_lines, sub.xml_text)
                match.status = "manual"
                match.reviewed = True
                match.review_note = "manual sub_id"
                match.xml_preview = clean_preview(sub.xml_text)
                match.matched_is_head = sub.is_head
                match.match_rel_start = r_start
                match.match_rel_end = r_end
                match.text_position = classify_text_position(r_start, r_end)
                append_manual_correction(match)
                print(f"Set manually to {sub.sub_id}.")
                break

            if cmd.isdigit():
                choice = int(cmd)
                if not (1 <= choice <= min(top_n, len(scored))):
                    print("Invalid candidate number.")
                    continue
                score, sub = scored[choice - 1]
                r_start, r_end = find_match_position(match.ocr_last_lines, sub.xml_text)
                match.entry = sub.entry
                match.sub_id = sub.sub_id
                match.score = score
                match.status = "manual"
                match.reviewed = True
                match.review_note = f"chosen candidate {choice}"
                match.xml_preview = clean_preview(sub.xml_text)
                match.matched_is_head = sub.is_head
                match.match_rel_start = r_start
                match.match_rel_end = r_end
                match.text_position = classify_text_position(r_start, r_end)
                append_manual_correction(match)
                print(f"Chosen: {sub.sub_id}")
                break

            print("Unknown command.")

    cv2.destroyAllWindows()
    save_matches_report(matches, MATCHES_REVIEWED_TSV)
    print("\nInteractive review finished.")


# ── Main processing ───────────────────────────────────────────────────────────

def process_pages(sublemmas: list[SublemmaDoc], mapping: dict[int, int], limit: Optional[int],
                   debug_images: int, last_lines_count: int, back_entries: int,
                   forward_entries: int, threshold: float) -> tuple[list[PageMatch], dict[str, Path]]:

    paths = ordered_scan_paths(IMAGE_DIR, limit=limit)
    scan_to_path = {p.name: p for p in paths}

    if not paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR} with scan >= {FIRST_SCAN_NO}")

    matches: list[PageMatch] = []

    print(f"\nProcessing {len(paths)} pages")
    print(f"First scan number       : {FIRST_SCAN_NO}")
    print(f"First logical page      : {FIRST_LOGICAL_PAGE}")
    print(f"Last OCR lines per page : {last_lines_count}")
    print(f"Back entries            : {back_entries}")
    print(f"Forward entries         : {forward_entries}")
    print(f"Fuzzy threshold         : {threshold}")
    print(f"RapidFuzz available     : {HAVE_RAPIDFUZZ}")
    print("-" * 110)

    for i, path in enumerate(paths):
        page = FIRST_LOGICAL_PAGE + i
        scan_no = scan_number(path)

        try:
            img = deskew(load_image(path))
            img_h, img_w = img.shape[:2]
            tokens = ocr_tokens(img)
            lines, split_x = tokens_to_lines(tokens, img_w, img_h)
            last_lines = get_last_ocr_lines(lines, last_lines_count)
            ocr_last_text = " ".join(ln.text for ln in last_lines).strip()

            detected_header_entry = detect_trailing_header_entry(last_lines, page, mapping)

            candidate_entries, candidates = candidate_sublemmas_for_page(
                page, sublemmas, mapping, back_entries, forward_entries
            )

            match = match_page_to_sublemma(
                page, path.name, ocr_last_text, candidates, candidate_entries,
                threshold, detected_header_entry,
            )
            matches.append(match)

            header_note = f" header_like={detected_header_entry}" if detected_header_entry is not None else ""
            head_tag = " [HEAD]" if match.matched_is_head else ""
            pos_note = (
                f" pos={match.text_position}({match.match_rel_start:.0%}-{match.match_rel_end:.0%})"
                if match.text_position else ""
            )

            print(
                f"[{i + 1:>4}/{len(paths)}] page={page:<4} scan={scan_no:<4} "
                f"entries={','.join(map(str, candidate_entries)):<14} score={match.score:>6.1f} "
                f"entry={str(match.entry or ''):<6} sub={(match.sub_id or ''):<16}{head_tag} "
                f"{match.status}{pos_note}{header_note}"
            )

            if i < debug_images:
                save_debug_image(img, page, path.name, lines, last_lines, match, split_x)

        except Exception as exc:
            print(f"[{i + 1:>4}/{len(paths)}] page={page:<4} scan={scan_no:<4} ERROR: {exc}")
            matches.append(PageMatch(page=page, scan=path.name, entry=None, sub_id=None, score=0.0,
                                      status=f"error: {exc}", candidate_entries=[],
                                      ocr_last_lines="", xml_preview=""))

    return matches, scan_to_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug-images", type=int, default=0)
    parser.add_argument("--test100", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--last-lines", type=int, default=DEFAULT_LAST_LINES)
    parser.add_argument("--back-entries", type=int, default=DEFAULT_BACK_ENTRIES)
    parser.add_argument("--forward-entries", type=int, default=DEFAULT_FORWARD_ENTRIES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    args = parser.parse_args(argv[1:])

    if args.test100:
        args.limit = 100
        args.debug_images = 10

    if not XML_INPUT.exists():
        print(f"ERROR: XML not found: {XML_INPUT}", file=sys.stderr)
        return 1
    if not MAPPING_TSV.exists():
        print(f"ERROR: mapping TSV not found: {MAPPING_TSV}. Run stage1 first.", file=sys.stderr)
        return 1
    if not IMAGE_DIR.exists():
        print(f"ERROR: image directory not found: {IMAGE_DIR}", file=sys.stderr)
        return 1

    print(f"Loading mapping: {MAPPING_TSV}")
    mapping = load_mapping_tsv(MAPPING_TSV)
    print(f"Entries in mapping: {len(mapping)}")

    print(f"Loading XML sublemmas: {XML_INPUT}")
    sublemmas = load_sublemmas(XML_INPUT)
    n_heads = sum(1 for s in sublemmas if s.is_head)
    print(f"Candidate docs loaded: {len(sublemmas)} (heads: {n_heads}, sublemmas: {len(sublemmas) - n_heads})")

    matches, scan_to_path = process_pages(
        sublemmas=sublemmas, mapping=mapping, limit=args.limit, debug_images=args.debug_images,
        last_lines_count=args.last_lines, back_entries=args.back_entries,
        forward_entries=args.forward_entries, threshold=args.threshold,
    )

    save_matches_report(matches, MATCHES_TSV)

    if args.review:
        interactive_review(matches, sublemmas, scan_to_path, top_n=15)

    print("\nStage 3 done. Next: stage4_build_enriched_xml.py --write-xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
