#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage1_build_page_mapping.py

Stage 1 of the pipeline.

OCRs every scan to find:
  - the printed page number (top/bottom center strip)
  - the entry numbers printed in the margin (start of each lemma)

Builds a rough entry -> page mapping, cleans obvious OCR mistakes, infers
missing entries that sit between two known entries on the SAME page, and
flags entries that are ambiguous between two DIFFERENT pages (left for
stage2_review_ambiguous_mapping.py to resolve manually).

This script does NOT touch any XML. It only produces:

    MAPPING_TSV               entry -> page
    AMBIGUOUS_MAPPING_TSV     entries ambiguous between two pages
    SCAN_REPORT_TSV           per-scan OCR debug info (used by stage 2 to
                              find which image file corresponds to a page)

Usage
-----
    python stage1_build_page_mapping.py
    python stage1_build_page_mapping.py --rebuild
    python stage1_build_page_mapping.py --test 80
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from pipeline_config import (
    TESSERACT_CMD,
    BASE_DIR,
    IMAGE_DIR,
    XML_INPUT,
    DEBUG_DIR,
    MAPPING_TSV,
    AMBIGUOUS_MAPPING_TSV,
    SCAN_REPORT_TSV,
    IMAGE_EXTENSIONS,
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

MAPPING_CACHE = BASE_DIR / "page_mapping.json"

FALLBACK_MAX_ENTRY_NUMBER = 9999
VALID_ENTRY_NUMBERS: set[int] = set()
MAX_ENTRY_NUMBER: int = FALLBACK_MAX_ENTRY_NUMBER

TESS_CONFIGS = ["--psm 6 --oem 3", "--psm 11 --oem 3", "--psm 4 --oem 3", "--psm 13 --oem 3"]
TESS_PAGE_NR_CONFIGS = ["--psm 6 --oem 3", "--psm 7 --oem 3"]

OCR_CONF_MIN = 40
OCR_CONF_MIN_FALLBACK = 25

PAGE_STRIP_FRAC = 0.15
PAGE_CENTER_TOL = 0.35

ENTRY_MARGIN_PX = 70
MAX_FORWARD_JUMP = 80
MAX_INTERNAL_GAP = 50

ARCHIVE_SIGLA = {
    "L", "S", "V", "T", "A", "R", "E", "Q",
    "IE", "II", "III", "IV", "VI", "VII", "VIII", "IX", "X",
    "Arm", "Reg", "Sup", "Plut", "Chigi", "Misc",
}


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class PageInfo:
    filename: str
    page_number: Optional[int]
    entry_numbers: list[int] = field(default_factory=list)
    raw_page_number: Optional[int] = None
    raw_entry_numbers: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ── Utility ──────────────────────────────────────────────────────────────────

def safe_conf(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return -1


def clean_token(token: str) -> str:
    return token.strip().strip(".,;:!?\"'\u201c\u201d\u2018\u2019[]{}")


def load_valid_entry_numbers_from_xml(xml_path: Path) -> set[int]:
    if not xml_path.exists():
        print(f"XML not found, using fallback max entry: {FALLBACK_MAX_ENTRY_NUMBER}")
        return set()

    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    entries: set[int] = set()

    for lemma in root.iter("lemma"):
        m = re.search(r"(\d{4})$", lemma.get("id", ""))
        if m:
            entries.add(int(m.group(1)))

    return entries


def initialize_entry_domain() -> None:
    global VALID_ENTRY_NUMBERS, MAX_ENTRY_NUMBER

    VALID_ENTRY_NUMBERS = load_valid_entry_numbers_from_xml(XML_INPUT)

    if VALID_ENTRY_NUMBERS:
        MAX_ENTRY_NUMBER = max(VALID_ENTRY_NUMBERS)
        print("\n=== ENTRY DOMAIN FROM XML ===")
        print(f"Valid entries : {len(VALID_ENTRY_NUMBERS)}")
        print(f"Entry range   : {min(VALID_ENTRY_NUMBERS)} - {MAX_ENTRY_NUMBER}")
    else:
        MAX_ENTRY_NUMBER = FALLBACK_MAX_ENTRY_NUMBER
        print("\n=== ENTRY DOMAIN FALLBACK ===")
        print(f"Using max entry: {MAX_ENTRY_NUMBER}")


def entry_upper_bound() -> int:
    return MAX_ENTRY_NUMBER or FALLBACK_MAX_ENTRY_NUMBER


def entry_exists_or_fallback(n: int) -> bool:
    if VALID_ENTRY_NUMBERS:
        return n in VALID_ENTRY_NUMBERS
    return 1 <= n <= entry_upper_bound()


# ── Image loading / preprocessing ────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
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
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def split_columns(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    h, w = img.shape[:2]
    s, e = int(w * 0.30), int(w * 0.70)
    whiteness = np.mean(binary[:, s:e], axis=0)
    whiteness = np.convolve(whiteness, np.ones(20) / 20, mode="same")
    gutter_x = s + int(np.argmax(whiteness))
    gutter_x = max(int(w * 0.35), min(int(w * 0.65), gutter_x))
    return img[:, :gutter_x], img[:, gutter_x:]


# ── OCR helpers ──────────────────────────────────────────────────────────────

def ocr_to_data(img_bgr: np.ndarray, config: str = "--psm 6 --oem 3") -> dict:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return pytesseract.image_to_data(pil, config=config, output_type=pytesseract.Output.DICT)


def group_into_lines(data: dict, y_tolerance: int = 15) -> dict[int, list[dict]]:
    lines: dict[int, list[dict]] = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = safe_conf(data["conf"][i])
        if not word or conf < 20:
            continue
        y = data["top"][i]
        matched = next((ly for ly in lines if abs(ly - y) < y_tolerance), None)
        key = matched if matched is not None else y
        lines.setdefault(key, []).append({
            "text": word, "conf": conf,
            "x": data["left"][i], "y": y,
            "w": data["width"][i], "h": data["height"][i],
        })
    return lines


def ocr_tokens(data: dict, min_conf: int = 20) -> list[dict]:
    tokens = []
    for i in range(len(data["text"])):
        raw = data["text"][i].strip()
        conf = safe_conf(data["conf"][i])
        if not raw or conf < min_conf:
            continue
        tokens.append({"index": i, "text": raw, "conf": conf,
                        "x": data["left"][i], "y": data["top"][i],
                        "w": data["width"][i], "h": data["height"][i]})
    return tokens


def vertical_overlap_ratio(a: dict, b: dict) -> float:
    overlap = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if overlap <= 0:
        return 0.0
    return overlap / max(1, min(a["h"], b["h"]))


def same_visual_line(a: dict, b: dict) -> bool:
    if vertical_overlap_ratio(a, b) >= 0.35:
        return True
    tolerance = max(10, 0.75 * max(a["h"], b["h"]))
    return abs((a["y"] + a["h"] / 2) - (b["y"] + b["h"] / 2)) <= tolerance


def tokens_right_on_same_visual_line(candidate: dict, tokens: list[dict],
                                      max_tokens: int = 8) -> list[dict]:
    cand_right = candidate["x"] + candidate["w"]
    same_line = [t for t in tokens
                 if t["index"] != candidate["index"]
                 and t["x"] > cand_right
                 and same_visual_line(candidate, t)]
    same_line.sort(key=lambda t: t["x"])
    return same_line[:max_tokens]


def has_token_left_on_same_visual_line(candidate: dict, tokens: list[dict]) -> bool:
    for tok in tokens:
        if tok["index"] == candidate["index"]:
            continue
        if tok["x"] >= candidate["x"] - 3:
            continue
        if same_visual_line(candidate, tok):
            return True
    return False


# ── Page number extraction ───────────────────────────────────────────────────

def _try_extract_page_number(strip: np.ndarray, full_width: int, config: str) -> Optional[int]:
    data = ocr_to_data(strip, config=config)
    lines = group_into_lines(data)
    center_x = full_width / 2
    candidates = []
    for _, words in lines.items():
        if len(words) != 1:
            continue
        token = words[0]
        text = token["text"].strip()
        if not re.fullmatch(r"\d{1,4}", text):
            continue
        num = int(text)
        if not (1 <= num <= 2000):
            continue
        word_cx = token["x"] + token["w"] / 2
        dist = abs(word_cx - center_x) / center_x
        if dist <= PAGE_CENTER_TOL:
            candidates.append((dist, token["conf"], num))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return candidates[0][2]


def extract_page_number(img: np.ndarray) -> Optional[int]:
    h, w = img.shape[:2]
    strip_h = int(h * PAGE_STRIP_FRAC)
    strips = [img[h - strip_h:h, :], img[:strip_h, :]]
    for strip in strips:
        for config in TESS_PAGE_NR_CONFIGS:
            result = _try_extract_page_number(strip, w, config)
            if result is not None:
                return result
    return None


# ── Entry token parsing ──────────────────────────────────────────────────────

def parse_entry_number_token(token: str) -> Optional[int]:
    if not token:
        return None
    raw = token.strip()
    if any(c in raw for c in "[]{}"):
        return None
    t = raw.strip(".,;:!?\"'()")
    if not re.search(r"\d", t):
        return None

    attempts = [t]
    if t and t[0] in "Il|" and len(t) > 1 and t[1].isdigit():
        attempts.append("1" + t[1:])
    if t and t[-1] in "lI" and len(t) > 1 and t[-2].isdigit():
        attempts.append(t[:-1] + "1")
    o_fixed = t.replace("O", "0").replace("o", "0")
    if o_fixed != t:
        attempts.append(o_fixed)

    for attempt in attempts:
        if re.fullmatch(r"\d{1,4}", attempt):
            n = int(attempt)
            if 1 <= n <= entry_upper_bound():
                return n
    return None


def parse_entry_number_candidate_token(token: str) -> tuple[Optional[int], Optional[str]]:
    n = parse_entry_number_token(token)
    if n is not None:
        return n, None
    if not token:
        return None, None
    raw = token.strip()
    if any(c in raw for c in "[]{}"):
        return None, None
    t = raw.strip(".,;:!?\"'()")
    if t and t[0] in "Il|" and len(t) > 1 and t[1].isdigit():
        t = "1" + t[1:]
    m = re.match(r"^(\d{1,4})([A-Za-zÁÉÍÓÚÀÂÊÔÃÄÖÜáéíóúàâêôãäöüß].*)$", t)
    if not m:
        return None, None
    num_s, rest = m.groups()
    num = int(num_s)
    if not (1 <= num <= entry_upper_bound()):
        return None, None
    rest = rest.strip(".,;:!?\"'()")
    if not rest or len(rest) <= 1:
        return None, None
    if re.fullmatch(r"[rvRV]", rest):
        return None, None
    return num, rest


def has_archive_reference_pattern(next_tokens: list[str]) -> bool:
    if not next_tokens:
        return False
    first = next_tokens[0].strip(".,;:")
    if first in ARCHIVE_SIGLA and len(next_tokens) >= 2:
        second = next_tokens[1].strip(".,;:")
        if re.search(r"\d", second):
            return True
    for tok in next_tokens[:3]:
        if re.search(r"\d+[rvs]{1,3}\.?s?\.?$", tok):
            return True
    return False


def score_name_likeness(tokens: list[str]) -> int:
    tokens = [clean_token(t) for t in tokens if clean_token(t)]
    if not tokens:
        return 0

    upper = "A-ZÁÉÍÓÚÀÂÊÔÃÄÖÜ"
    lower = "a-záéíóúàâêôãäöüß"

    particles = {
        "de", "von", "van", "v", "d", "d'", "di", "del", "der", "den",
        "zu", "zum", "zur", "in", "im", "le", "la", "les", "du", "des", "al", "el",
    }
    stopwords = {
        "et", "ad", "ac", "per", "pro", "cum", "m", "arg", "p", "o", "s", "l", "t", "a",
        "in", "de", "ut", "ex", "ob",
    }
    reference_sigla = {
        "A", "S", "L", "V", "T", "R", "E", "I",
        "IE", "II", "III", "IV", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
        "Arm", "Reg", "Sup", "Plut",
    }
    abbreviation_pattern = re.compile(
        r"^(Joh|Johs|Henr|Hein|Fr|Guil|Gerh|Wilh|Theod|Phil|Nic|Alex|Arn|Arnd|"
        r"Berth|Conr|Herm|Pet|Marg|Mich|Bart|Laur|Matth|Clem|Ang|Ant|"
        r"Baldew|Bald|Bert|Christ|Corn|Diet|Diedr|Eberh|Evert|Frid|Gott|Heinr|"
        r"Lamb|Ludw|Lud|Otto|Reinh|Rich|Rud|Walt|Wern|Winand)\.?$",
        re.IGNORECASE,
    )

    def normalize_candidate(tok: str) -> str:
        tok = clean_token(tok).replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        return re.sub(rf"[^{upper}{lower}'\-.]", "", tok)

    def is_reference_like(tok: str) -> bool:
        raw = clean_token(tok)
        stripped = raw.strip(".")
        if not stripped:
            return True
        if re.search(r"\d", stripped):
            return True
        if stripped in reference_sigla:
            return True
        if re.fullmatch(r"[IVXLCDM]+", stripped):
            return True
        if stripped.isupper() and len(stripped) <= 4:
            return True
        return False

    def is_strong_name(tok: str) -> bool:
        raw = clean_token(tok)
        if not raw or re.search(r"\d", raw) or is_reference_like(raw):
            return False
        normalized = normalize_candidate(raw)
        if not normalized:
            return False
        low = normalized.lower().strip(".")
        if low in stopwords:
            return False
        if abbreviation_pattern.match(normalized):
            return True
        stripped_norm = normalized.strip(".")
        if len(stripped_norm) < 3:
            return False
        if normalized.isupper() and len(normalized) <= 4:
            return False
        if re.match(rf"^[{upper}][{lower}]{{2,}}", normalized):
            return True
        if re.match(rf"^[{upper}][{lower}]{{1,}}", normalized):
            return True
        if "(" in raw and len(stripped_norm) >= 4 and re.match(rf"^[{lower}]{{4,}}", normalized):
            return True
        return False

    if len(tokens) >= 2 and is_reference_like(tokens[0]) and is_reference_like(tokens[1]):
        return 0

    first = tokens[0]
    candidate_tokens = [first]
    if first.lower().strip(".") in particles and len(tokens) >= 2:
        candidate_tokens.append(tokens[1])
    if len(tokens) >= 2 and tokens[1].startswith("("):
        candidate_tokens.append(first + tokens[1])

    score = 0
    for tok in candidate_tokens:
        if is_strong_name(tok):
            score += 5
    if len(tokens) >= 2 and is_strong_name(tokens[1]):
        score += 2
    for tok in tokens[:2]:
        if "(" in tok and is_strong_name(tok):
            score += 1
    for tok in tokens[:3]:
        if is_reference_like(tok):
            score -= 3
    return max(score, 0)


# ── Entry extraction ──────────────────────────────────────────────────────────

def _extract_entries_from_column(col: np.ndarray, conf_min: int = OCR_CONF_MIN,
                                  name_score_min: int = 5) -> list[int]:
    found: list[int] = []

    for config in TESS_CONFIGS:
        data = ocr_to_data(col, config=config)
        n_raw = len(data["text"])
        tokens = ocr_tokens(data, min_conf=20)
        if not tokens:
            continue

        xs = [data["left"][i] for i in range(n_raw)
              if data["text"][i].strip() and safe_conf(data["conf"][i]) >= conf_min]
        if not xs:
            continue

        text_start_x = float(np.percentile(xs, 5))
        margin_limit = text_start_x + ENTRY_MARGIN_PX

        pass_found: list[int] = []

        for tok in tokens:
            if tok["conf"] < conf_min:
                continue
            entry_num, attached_name = parse_entry_number_candidate_token(tok["text"])
            if entry_num is None:
                continue
            if tok["x"] > margin_limit + 10:
                continue
            if has_token_left_on_same_visual_line(tok, tokens):
                continue

            right_tokens = tokens_right_on_same_visual_line(tok, tokens, max_tokens=8)
            next_tokens: list[str] = []
            if attached_name:
                next_tokens.append(attached_name)
            next_tokens.extend(t["text"] for t in right_tokens if t["text"].strip())

            if has_archive_reference_pattern(next_tokens):
                continue

            if score_name_likeness(next_tokens) < name_score_min:
                continue

            pass_found.append(entry_num)

        found.extend(pass_found)
        if pass_found:
            break

    return sorted(set(found))


def extract_entry_numbers(img: np.ndarray) -> list[int]:
    left_col, right_col = split_columns(img)
    found: list[int] = []

    for col in (left_col, right_col):
        entries = _extract_entries_from_column(col, conf_min=OCR_CONF_MIN, name_score_min=5)
        if not entries:
            entries = _extract_entries_from_column(col, conf_min=OCR_CONF_MIN_FALLBACK,
                                                    name_score_min=4)
        found.extend(entries)

    return sorted(set(found))


# ── Per-scan processing ───────────────────────────────────────────────────────

def process_scan(path: Path) -> PageInfo:
    img = deskew(load_image(path))
    page = extract_page_number(img)
    entries = extract_entry_numbers(img)
    return PageInfo(filename=path.name, page_number=page,
                     entry_numbers=entries, raw_page_number=page,
                     raw_entry_numbers=list(entries))


# ── Page normalization ────────────────────────────────────────────────────────

def normalize_page_numbers(infos: list[PageInfo]) -> None:
    anchor_idx = anchor_page = None
    for i, info in enumerate(infos):
        if info.page_number is not None and 1 <= info.page_number <= 2000:
            anchor_idx = i
            anchor_page = info.page_number
            break
    if anchor_idx is None:
        print("  [page normalization] No reliable anchor page found.")
        return
    print(f"  [page normalization] anchor: {infos[anchor_idx].filename} -> page {anchor_page}")
    for i, info in enumerate(infos):
        expected = anchor_page + (i - anchor_idx)
        original = info.page_number
        if original is None or abs(original - expected) > 1:
            info.notes.append(f"page normalized: {original} -> {expected}")
            print(f"  [page normalization] {info.filename}: {original} -> {expected}")
            info.page_number = expected


# ── Entry cleanup ─────────────────────────────────────────────────────────────

def correct_by_expected_suffix(n: int, previous_max: int,
                                same_page_others: list[int]) -> Optional[int]:
    if previous_max <= 0:
        return None
    raw = str(n)
    if len(raw) == 1:
        return None
    plausible_others = [x for x in same_page_others
                        if previous_max - 5 <= x <= previous_max + 120]
    if plausible_others:
        lo = min(previous_max + 1, min(plausible_others) - 5)
        hi = max(plausible_others) + 5
    else:
        lo = previous_max + 1
        hi = previous_max + MAX_FORWARD_JUMP
    lo, hi = max(1, lo), min(entry_upper_bound(), hi)

    suffixes = [(raw, 0)]
    if len(raw) >= 3:
        suffixes.append((raw[-3:], 1))
    if len(raw) >= 2:
        suffixes.append((raw[-2:], 2))

    candidates = []
    for cand in range(lo, hi + 1):
        if cand <= previous_max or cand in same_page_others:
            continue
        if VALID_ENTRY_NUMBERS and cand not in VALID_ENTRY_NUMBERS:
            continue
        cand_s = str(cand)
        for suffix, penalty in suffixes:
            if cand_s.endswith(suffix):
                candidates.append((penalty, abs(cand - (previous_max + 1)), cand))
                break
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def clean_entry_numbers(infos: list[PageInfo]) -> None:
    previous_max = 0
    for info in infos:
        original_entries = sorted(set(info.entry_numbers))
        temp: list[int] = []
        for n in original_entries:
            original = n
            same_page_others = [x for x in original_entries if x != n]
            if n < 1 or n > entry_upper_bound():
                info.notes.append(f"entry discarded: {original} outside valid range")
                print(f"  [entry discarded] {info.filename}: {original} outside valid range")
                continue
            if previous_max and n > previous_max + MAX_FORWARD_JUMP:
                corrected = correct_by_expected_suffix(n, previous_max, same_page_others)
                if corrected is not None:
                    info.notes.append(f"entry corrected: {original} -> {corrected}")
                    print(f"  [entry correction] {info.filename}: {original} -> {corrected}")
                    n = corrected
                else:
                    info.notes.append(f"entry discarded: {original} after previous_max={previous_max} (forward jump)")
                    print(f"  [entry discarded] {info.filename}: {original} (forward jump)")
                    continue
            elif previous_max and n <= previous_max - 10:
                if previous_max > 100 and n < 10:
                    info.notes.append(f"entry discarded: {original} (too small)")
                    print(f"  [entry discarded] {info.filename}: {original} (too small)")
                    continue
                corrected = correct_by_expected_suffix(n, previous_max, same_page_others)
                if corrected is not None:
                    info.notes.append(f"entry corrected: {original} -> {corrected}")
                    print(f"  [entry correction] {info.filename}: {original} -> {corrected}")
                    n = corrected
                else:
                    info.notes.append(f"entry discarded: {original} after previous_max={previous_max}")
                    print(f"  [entry discarded] {info.filename}: {original}")
                    continue
            if 1 <= n <= entry_upper_bound() and entry_exists_or_fallback(n):
                if n not in temp:
                    temp.append(n)
            else:
                info.notes.append(f"entry discarded: {n} not present in XML")
                print(f"  [entry discarded] {info.filename}: {n} not in XML")

        temp = sorted(temp)
        cleaned: list[int] = []
        for n in temp:
            if not cleaned:
                cleaned.append(n)
                continue
            gap = n - cleaned[-1]
            if gap > MAX_INTERNAL_GAP:
                info.notes.append(f"entry discarded: {n} (internal gap {gap})")
                print(f"  [entry discarded] {info.filename}: {n} (internal gap {gap})")
                continue
            cleaned.append(n)
        info.entry_numbers = sorted(set(cleaned))
        if info.entry_numbers:
            previous_max = max(previous_max, max(info.entry_numbers))


def resolve_duplicate_entries(infos: list[PageInfo]) -> None:
    occurrences: dict[int, list[PageInfo]] = {}
    for info in infos:
        for entry in info.entry_numbers:
            occurrences.setdefault(entry, []).append(info)
    for entry, places in occurrences.items():
        if len(places) <= 1:
            continue
        scored = []
        for info in places:
            entries = set(info.entry_numbers)
            score = (5 if entry - 1 in entries else 0) + (5 if entry + 1 in entries else 0) + \
                    (2 if entry - 2 in entries else 0) + (2 if entry + 2 in entries else 0)
            nearby = [x for x in entries if x != entry and abs(x - entry) <= 5]
            score += min(len(nearby), 5)
            if not nearby:
                score -= 5
            score += min(len(entries), 10) * 0.1
            scored.append((score, info))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_info = scored[0]
        for score, info in scored[1:]:
            if entry in info.entry_numbers:
                info.entry_numbers.remove(entry)
                msg = f"duplicate resolved: removed entry {entry}; kept in {best_info.filename}"
                info.notes.append(msg)
                print(f"  [duplicate resolved] entry {entry}: removed from {info.filename}")
    for info in infos:
        info.entry_numbers = sorted(set(info.entry_numbers))


# ── Reports ───────────────────────────────────────────────────────────────────

def save_scan_report(infos: list[PageInfo]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCAN_REPORT_TSV, "w", encoding="utf-8") as f:
        f.write("scan_index\tfilename\traw_page\tfinal_page\traw_entries\tfinal_entries\tnotes\n")
        for i, info in enumerate(infos, 1):
            f.write(f"{i}\t{info.filename}\t{info.raw_page_number}\t{info.page_number}\t"
                    f"{','.join(map(str, info.raw_entry_numbers))}\t"
                    f"{','.join(map(str, info.entry_numbers))}\t"
                    f"{' | '.join(info.notes)}\n")
    print(f"Scan report saved -> {SCAN_REPORT_TSV}")


def report_duplicate_entries(infos: list[PageInfo]) -> None:
    seen: dict[int, str] = {}
    duplicates = []
    for info in infos:
        for entry in info.entry_numbers:
            if entry in seen:
                duplicates.append((entry, seen[entry], info.filename))
            else:
                seen[entry] = info.filename
    print("\n=== DUPLICATE ENTRIES ===")
    if not duplicates:
        print("No duplicate entries after cleaning.")
    else:
        for entry, ff, sf in duplicates[:50]:
            print(f"  entry {entry}: {ff} AND {sf}")
        if len(duplicates) > 50:
            print(f"  ... and {len(duplicates) - 50} more")
    out = DEBUG_DIR / "duplicate_entries.tsv"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("entry\tfirst_file\tduplicate_file\n")
        for entry, ff, sf in duplicates:
            f.write(f"{entry}\t{ff}\t{sf}\n")
    print(f"Duplicate report saved -> {out}")


def report_entry_gaps_by_scan(infos: list[PageInfo]) -> None:
    warnings = []
    prev_max = None
    for info in infos:
        entries = sorted(info.entry_numbers)
        if not entries:
            continue
        if prev_max is not None:
            if entries[0] > prev_max + 1 + 2:
                warnings.append((info.filename, "gap_before_scan", prev_max + 1, entries[0], entries))
            if entries[0] < prev_max - 5:
                warnings.append((info.filename, "backward_jump", prev_max, entries[0], entries))
        for a, b in zip(entries, entries[1:]):
            if b - a > 2:
                warnings.append((info.filename, "gap_inside_scan", a, b, entries))
        prev_max = max(prev_max or 0, max(entries))
    print("\n=== ENTRY GAP/JUMP WARNINGS ===")
    if not warnings:
        print("No suspicious entry gaps/jumps.")
    else:
        for fname, typ, a, b, entries in warnings[:50]:
            print(f"  {fname}: {typ} {a} -> {b} entries={entries}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more")
    out = DEBUG_DIR / "entry_gaps_by_scan.tsv"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("filename\ttype\tfrom\tto\tentries\n")
        for fname, typ, a, b, entries in warnings:
            f.write(f"{fname}\t{typ}\t{a}\t{b}\t{','.join(map(str, entries))}\n")
    print(f"Gap/jump report saved -> {out}")


def report_missing_entries(mapping: dict[int, int]) -> None:
    print("\n=== MISSING ENTRIES ===")
    expected_entries = sorted(VALID_ENTRY_NUMBERS) if VALID_ENTRY_NUMBERS else []
    if not expected_entries:
        print("No entry domain known.")
        return
    missing = [n for n in expected_entries if n not in mapping]
    if not missing:
        print("No missing entries.")
    else:
        print(f"Missing entries: {len(missing)}")
        print(missing[:200])
        if len(missing) > 200:
            print(f"... and {len(missing) - 200} more")


def save_missing_entries_context(mapping: dict[int, int]) -> None:
    entries = sorted(mapping)
    if not entries:
        print("No entries in mapping.")
        return
    expected_entries = sorted(VALID_ENTRY_NUMBERS) if VALID_ENTRY_NUMBERS else \
        list(range(entries[0], entries[-1] + 1))
    missing = [n for n in expected_entries if n not in mapping]
    out = DEBUG_DIR / "missing_entries_context.tsv"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("missing_entry\tprev_entry\tprev_page\tnext_entry\tnext_page\tcomment\n")
        for n in missing:
            prev_entries = [e for e in entries if e < n]
            next_entries = [e for e in entries if e > n]
            prev_e = prev_entries[-1] if prev_entries else None
            next_e = next_entries[0] if next_entries else None
            prev_p = mapping.get(prev_e) if prev_e is not None else None
            next_p = mapping.get(next_e) if next_e is not None else None
            comment = f"likely page {prev_p}" if prev_p == next_p else f"between page {prev_p} and page {next_p}"
            f.write(f"{n}\t{prev_e}\t{prev_p}\t{next_e}\t{next_p}\t{comment}\n")
    print(f"Missing entries context saved -> {out}")


def save_mapping_report(mapping: dict[int, int]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_TSV, "w", encoding="utf-8") as f:
        f.write("entry\tpage\n")
        for entry, page in sorted(mapping.items()):
            f.write(f"{entry}\t{page}\n")
    print(f"\nMapping report saved -> {MAPPING_TSV}")


# ── Inference ─────────────────────────────────────────────────────────────────

def infer_missing_entries(mapping: dict[int, int], max_gap_entries: int = 100) -> dict[int, int]:
    """
    Fill missing entries ONLY when they sit between two known entries that
    were assigned to the SAME page. Never infers across a page boundary.
    """
    if not mapping:
        return mapping

    expected_entries = sorted(VALID_ENTRY_NUMBERS) if VALID_ENTRY_NUMBERS else \
        list(range(min(mapping), max(mapping) + 1))
    expected_set = set(expected_entries)

    known_entries = sorted(e for e in expected_entries if e in mapping)

    inferred = dict(mapping)
    added: list[tuple] = []

    for left_entry, right_entry in zip(known_entries, known_entries[1:]):
        gap = right_entry - left_entry
        if gap <= 1:
            continue
        if gap - 1 > max_gap_entries:
            continue

        left_page = mapping[left_entry]
        right_page = mapping[right_entry]

        if left_page != right_page:
            continue

        missing_between = [n for n in range(left_entry + 1, right_entry)
                            if n in expected_set and n not in inferred]

        for n in missing_between:
            inferred[n] = left_page
            added.append((n, left_page, left_entry, right_entry, "same-page"))

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEBUG_DIR / "inferred_missing_entries.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("entry\tinferred_page\tleft_entry\tright_entry\treason\n")
        for entry, page, left_entry, right_entry, reason in added:
            f.write(f"{entry}\t{page}\t{left_entry}\t{right_entry}\t{reason}\n")

    print("\n=== INFERRED MISSING ENTRIES ===")
    print(f"Inferred entries: {len(added)}")
    print(f"Report saved -> {out}")

    return dict(sorted(inferred.items()))


def report_ambiguous_missing_between_pages(mapping: dict[int, int]) -> None:
    """
    Entries missing between two known entries on DIFFERENT pages -> not
    auto-filled, saved for stage2_review_ambiguous_mapping.py to resolve
    manually with the actual scans.
    """
    if not mapping:
        return

    expected_entries = sorted(VALID_ENTRY_NUMBERS) if VALID_ENTRY_NUMBERS else \
        list(range(min(mapping), max(mapping) + 1))
    expected_set = set(expected_entries)

    known_entries = sorted(e for e in expected_entries if e in mapping)

    rows = []

    for left_entry, right_entry in zip(known_entries, known_entries[1:]):
        if right_entry - left_entry <= 1:
            continue

        left_page = mapping[left_entry]
        right_page = mapping[right_entry]

        if left_page == right_page:
            continue

        missing = [n for n in range(left_entry + 1, right_entry)
                   if n in expected_set and n not in mapping]

        for n in missing:
            rows.append((n, left_entry, left_page, right_entry, right_page,
                         f"ambiguous between page {left_page} and page {right_page}"))

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with open(AMBIGUOUS_MAPPING_TSV, "w", encoding="utf-8") as f:
        f.write("missing_entry\tleft_entry\tleft_page\tright_entry\tright_page\tcomment\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

    print("\n=== AMBIGUOUS MISSING BETWEEN PAGES ===")
    print(f"Ambiguous entries: {len(rows)}")
    print(f"Report saved -> {AMBIGUOUS_MAPPING_TSV}")
    print("Run stage2_review_ambiguous_mapping.py to resolve these manually.")


# ── Mapping builder ───────────────────────────────────────────────────────────

def build_page_mapping(force_rebuild: bool = False, max_images: Optional[int] = None) -> dict[int, int]:
    initialize_entry_domain()

    cache_path = MAPPING_CACHE
    if max_images:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DEBUG_DIR / f"page_mapping_test_{max_images}.json"

    if cache_path.exists() and not force_rebuild:
        print(f"Loading mapping from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return {int(k): int(v) for k, v in json.load(f).items()}

    paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images:
        paths = paths[:max_images]
        print(f"  [test mode] processing {max_images} images only")

    infos: list[PageInfo] = []
    print(f"\nProcessing {len(paths)} scans ...")
    print("-" * 60)
    for i, path in enumerate(paths, 1):
        print(f"[{i:>4}/{len(paths)}] {path.name}", end="  ->  ")
        try:
            info = process_scan(path)
            infos.append(info)
            print(f"page={info.page_number}  entries={info.entry_numbers}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    normalize_page_numbers(infos)
    clean_entry_numbers(infos)
    resolve_duplicate_entries(infos)
    report_duplicate_entries(infos)
    report_entry_gaps_by_scan(infos)
    save_scan_report(infos)

    mapping: dict[int, int] = {}
    for info in infos:
        if info.page_number is None:
            continue
        for entry in info.entry_numbers:
            if entry_exists_or_fallback(entry):
                mapping.setdefault(entry, info.page_number)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(mapping.items())}, f, indent=2, ensure_ascii=False)
    print(f"\nCache saved -> {cache_path}")
    return mapping


# ── Validation ────────────────────────────────────────────────────────────────

def validate_mapping(mapping: dict[int, int], sample_size: int = 20) -> None:
    import random
    print("\n=== MAPPING VALIDATION ===")
    if not mapping:
        print("No mapping entries.")
        return
    keys = sorted(mapping)
    sample = sorted(random.sample(keys, min(sample_size, len(keys))))
    print(f"{'Entry':>6}  ->  {'Page':>5}\n" + "-" * 22)
    for k in sample:
        print(f"{k:>6}  ->  {mapping[k]:>5}")
    print(f"\nTotal entries : {len(mapping)}")
    print(f"Entry range   : {keys[0]} - {keys[-1]}")
    print(f"Page range    : {min(mapping.values())} - {max(mapping.values())}")
    anomalies = [f"  entry {keys[i]} p.{mapping[keys[i]]} < entry {keys[i-1]} p.{mapping[keys[i-1]]}"
                 for i in range(1, len(keys)) if mapping[keys[i]] < mapping[keys[i - 1]]]
    if anomalies:
        print(f"\nAnomalies ({len(anomalies)}):")
        for a in anomalies[:10]:
            print(a)
    else:
        print("\nPage sequence is consistent.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force re-OCR, ignore cache.")
    parser.add_argument("--test", type=int, default=None, help="Process only first N images.")
    args = parser.parse_args()

    mapping = build_page_mapping(force_rebuild=args.rebuild, max_images=args.test)
    mapping = infer_missing_entries(mapping)
    report_ambiguous_missing_between_pages(mapping)
    validate_mapping(mapping)
    report_missing_entries(mapping)
    save_missing_entries_context(mapping)
    save_mapping_report(mapping)

    print("\nStage 1 done. Next: stage2_review_ambiguous_mapping.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
