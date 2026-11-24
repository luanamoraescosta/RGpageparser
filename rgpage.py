#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Self-contained page / RG locator.

For every page scan of a Repertorium Germanicum volume this tool finds:

  1. the printed page number;
  2. the entry numbers starting in the margins;
  3. the RG block (head, rg1, rg2, ...) at the bottom of the page and whether
     it ends there or continues on the next page.

It does not read any legacy debug table.  The OCR cache is written to
``debug/rgN/ocr_pages.jsonl``; matching and page spans are computed from that
cache, so a later run can skip the OCR step.

Usage:
    python rgpage.py run --vol 8 --easyocr --show
    python rgpage.py run --vol 8 --limit 30 --show          # quick test
    python rgpage.py scan --vol 8 --rebuild                 # OCR only
    python rgpage.py match --vol 8 --random 20 --easyocr    # match + images
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = Path("/Users/luanamoraescosta/DH/germania_sacra/RGPARSER/H4 6481 (8,1")
DEFAULT_XML_DIR = Path("/Users/luanamoraescosta/DH/germania_sacra/RGPARSER/RG_data/rg_xml")

TESSERACT = "/opt/homebrew/bin/tesseract"
if Path(TESSERACT).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT

MIN_CONF = 20
BLANK_MEAN_CONF = 50.0
END_THRESHOLD = 0.90
START_THRESHOLD = 0.12

Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int


@dataclass
class Line:
    text: str
    x: int
    y: int
    w: int
    h: int
    col: int
    conf: float = 0.0


@dataclass
class Doc:
    entry: int
    block_id: str
    kind: str
    text: str
    norm: str

    @property
    def is_head(self) -> bool:
        return self.kind == "head"


@dataclass
class PageOCR:
    scan: str
    scan_no: int
    page: Optional[int]
    page_box: Optional[Box]
    page_source: str
    entries: list[int]
    entry_boxes: dict[int, Box]
    lines: list[Line]
    mean_conf: float
    blank: bool = False


@dataclass
class PageMatch:
    page: PageOCR
    block: Optional[Doc]
    score: float
    position: str
    status: str
    span_start: Optional[int] = None
    span_end: Optional[int] = None


@dataclass
class Span:
    entry: int
    block_id: str
    kind: str
    label: str
    page_start: int
    page_end: int
    source: str
    warning: str = ""
    score: float = 0.0


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ſ", "s").replace("°", "r")
    text = re.sub(r"[\"'`\u2018\u2019\u201c\u201d´]", "v", text)
    text = text.lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def kind_label(block_id: str, kind: str) -> str:
    if kind == "head":
        return "head"
    if block_id.startswith("ov"):
        return block_id
    m = re.search(r"-(\d+)$", block_id)
    return f"rg{m.group(1)}" if m else "regest"


def position_label(position: str, span_end: Optional[int] = None,
                   page: Optional[int] = None) -> str:
    if position in ("starts-here", "continues"):
        text = "continues to next page"
    elif position in ("ends-here", "complete-on-page"):
        text = "ends here"
    else:
        text = position or "unknown"
    if span_end and page and span_end > page:
        text += f" (to p. {span_end})"
    return text


def classify(rel_start: float, rel_end: float) -> str:
    if rel_start <= START_THRESHOLD and rel_end >= END_THRESHOLD:
        return "complete-on-page"
    if rel_end >= END_THRESHOLD:
        return "ends-here"
    if rel_start <= START_THRESHOLD:
        return "starts-here"
    return "continues"


# ---------------------------------------------------------------------------
# XML / scans
# ---------------------------------------------------------------------------

def load_docs(xml_path: Path) -> list[Doc]:
    sys.path.insert(0, str(ROOT))
    import rg_xml

    docs: list[Doc] = []
    for block in rg_xml.load_blocks(xml_path):
        norm = normalize(block.text)
        if norm:
            docs.append(Doc(block.entry, block.block_id, block.kind, block.text, norm))
    return docs


def scan_number(path: Path) -> Optional[int]:
    m = re.search(r"_(\d+)(?=\.[^.]+$)", path.name)
    return int(m.group(1)) if m else None


def ordered_scans(image_dir: Path, first_scan: int, from_scan: Optional[int] = None,
                  to_scan: Optional[int] = None, limit: Optional[int] = None) -> list[Path]:
    paths = []
    for path in image_dir.iterdir():
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        no = scan_number(path)
        if no is None or no < first_scan:
            continue
        if from_scan is not None and no < from_scan:
            continue
        if to_scan is not None and no > to_scan:
            continue
        paths.append((no, path))
    paths.sort(key=lambda item: item[0])
    if limit is not None:
        paths = paths[:limit]
    return [path for _, path in paths]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_words(img: np.ndarray, config: str = "--psm 6 --oem 3") -> list[Word]:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, config=config, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1
        if not text or conf < MIN_CONF:
            continue
        words.append(Word(text, conf, int(data["left"][i]), int(data["top"][i]),
                          int(data["width"][i]), int(data["height"][i])))
    return words


_EASY = None
_EASY_FAILED = False


def easy_read(img: np.ndarray) -> list[tuple[Box, str, float]]:
    global _EASY, _EASY_FAILED
    if _EASY_FAILED:
        return []
    if _EASY is None:
        try:
            import easyocr
        except Exception as exc:
            print(f"EasyOCR is not available ({exc}); using Tesseract only.")
            _EASY_FAILED = True
            return []
        _EASY = easyocr.Reader(["en"], gpu=False, verbose=False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []
    for bbox, text, conf in _EASY.readtext(gray, detail=1, paragraph=False):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        out.append(((int(min(xs)), int(min(ys)),
                     int(max(xs) - min(xs)), int(max(ys) - min(ys))),
                    (text or "").strip(), float(conf)))
    return out


def split_x_from_words(words: list[Word], width: int) -> int:
    centers = sorted(w.x + w.w / 2 for w in words if 0.10 * width <= w.x + w.w / 2 <= 0.90 * width)
    if len(centers) < 20:
        return width // 2
    best_gap, best = 0.0, width // 2
    for a, b in zip(centers, centers[1:]):
        mid = (a + b) / 2
        if 0.35 * width <= mid <= 0.65 * width and b - a > best_gap:
            best_gap, best = b - a, int(mid)
    return best if best_gap >= 80 else width // 2


def words_to_lines(words: list[Word], split_x: int, height: int) -> list[Line]:
    lines: list[Line] = []
    for col, subset in ((0, [w for w in words if w.x + w.w / 2 < split_x]),
                        (1, [w for w in words if w.x + w.w / 2 >= split_x])):
        subset = sorted(subset, key=lambda w: (w.y + w.h / 2, w.x))
        groups: list[list[Word]] = []
        for word in subset:
            if not groups:
                groups.append([word])
                continue
            last = groups[-1]
            last_cy = sum(w.y + w.h / 2 for w in last) / len(last)
            if abs((word.y + word.h / 2) - last_cy) <= 24:
                last.append(word)
            else:
                groups.append([word])
        for group in groups:
            group = sorted(group, key=lambda w: w.x)
            text = " ".join(w.text for w in group).strip()
            if len(normalize(text)) < 2:
                continue
            x1, y1 = min(w.x for w in group), min(w.y for w in group)
            x2 = max(w.x + w.w for w in group)
            y2 = max(w.y + w.h for w in group)
            if re.fullmatch(r"\d{1,4}", text) and y1 > 0.85 * height:
                continue
            lines.append(Line(text, x1, y1, x2 - x1, y2 - y1, col,
                              sum(w.conf for w in group) / len(group)))
    return lines


def find_page_number(img: np.ndarray, expected: Optional[int] = None
                     ) -> tuple[Optional[int], Optional[Box], str]:
    h, w = img.shape[:2]
    strips = [(int(h * 0.85), h), (0, int(h * 0.12))]
    best = None
    for y0, y1 in strips:
        strip = img[y0:y1]
        for config in ("--psm 6 --oem 3", "--psm 7 --oem 3"):
            for word in ocr_words(strip, config):
                if not re.fullmatch(r"\d{1,4}", word.text):
                    continue
                num = int(word.text)
                if not 1 <= num <= 2000:
                    continue
                cx = word.x + word.w / 2
                if abs(cx - w / 2) > 0.35 * w:
                    continue
                rank = (0 if num == expected else 1, abs(cx - w / 2) / w, -word.conf)
                box = (word.x, word.y + y0, word.w, word.h)
                if best is None or rank < best[0]:
                    best = (rank, num, box, "tesseract")
    if best is None:
        return None, None, "none"
    return best[1], best[2], best[3]


def last_printed_lines(img: np.ndarray, words: list[Word], split_x: int
                       ) -> tuple[list[Line], float]:
    lines = words_to_lines(words, split_x, img.shape[0])

    def printed(candidates: list[Line]) -> list[Line]:
        if not candidates:
            return []
        heights = sorted(ln.h for ln in candidates)
        median_h = heights[len(heights) // 2]
        keep = [ln for ln in candidates if ln.h <= 1.8 * median_h and ln.conf >= 35]
        return keep or candidates

    right = printed([ln for ln in lines if ln.col == 1])
    source = right if len(right) >= 2 else printed(lines)
    last = [ln for ln in source if len(normalize(ln.text)) >= 12][-2:]
    if not last:
        last = source[-2:] if source else []
    body = [w.conf for w in words if w.conf >= 20]
    mean_conf = sum(body) / len(body) if body else 0.0
    return last, mean_conf


ENTRY_MARGIN_PX = 70
ENTRY_CONF_MIN = 40
ENTRY_CONF_FALLBACK = 25
ENTRY_TESS_CONFIGS = ["--psm 6 --oem 3", "--psm 11 --oem 3"]

ARCHIVE_SIGLA = {
    "L", "S", "V", "T", "A", "R", "E", "Q",
    "IE", "II", "III", "IV", "VI", "VII", "VIII", "IX", "X",
    "Arm", "Reg", "Sup", "Plut", "Chigi", "Misc",
}


def safe_conf(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return -1


def clean_token(token: str) -> str:
    return token.strip().strip(".,;:!?\"'\u201c\u201d\u2018\u2019[]{}")


def split_columns(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    h, w = img.shape[:2]
    s, e = int(w * 0.30), int(w * 0.70)
    whiteness = np.mean(binary[:, s:e], axis=0)
    whiteness = np.convolve(whiteness, np.ones(20) / 20, mode="same")
    gutter_x = s + int(np.argmax(whiteness))
    gutter_x = max(int(w * 0.35), min(int(w * 0.65), gutter_x))
    return img[:, :gutter_x], img[:, gutter_x:], gutter_x


def col_tokens(col: np.ndarray, config: str) -> tuple[list[dict], list[float]]:
    pil = Image.fromarray(cv2.cvtColor(col, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, config=config, output_type=pytesseract.Output.DICT)
    tokens = []
    xs = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf = safe_conf(data["conf"][i])
        if not text or conf < 20:
            continue
        tok = {"index": i, "text": text, "conf": conf,
               "x": data["left"][i], "y": data["top"][i],
               "w": data["width"][i], "h": data["height"][i]}
        tokens.append(tok)
        if conf >= ENTRY_CONF_FALLBACK:
            xs.append(tok["x"])
    return tokens, xs


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


def tokens_right_on_same_line(candidate: dict, tokens: list[dict],
                              max_tokens: int = 8) -> list[dict]:
    right = candidate["x"] + candidate["w"]
    same = [t for t in tokens if t["index"] != candidate["index"]
            and t["x"] > right and same_visual_line(candidate, t)]
    same.sort(key=lambda t: t["x"])
    return same[:max_tokens]


def has_token_left_on_same_line(candidate: dict, tokens: list[dict]) -> bool:
    for tok in tokens:
        if tok["index"] == candidate["index"]:
            continue
        if tok["x"] >= candidate["x"] - 3:
            continue
        if same_visual_line(candidate, tok):
            return True
    return False


def parse_entry_number_token(token: str, valid: set[int], max_entry: int) -> Optional[int]:
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
    fixed = t.replace("O", "0").replace("o", "0")
    if fixed != t:
        attempts.append(fixed)
    for attempt in attempts:
        if re.fullmatch(r"\d{1,5}", attempt):
            n = int(attempt)
            if n in valid or (not valid and 1 <= n <= max_entry):
                return n
    return None


def parse_entry_candidate(token: str, valid: set[int], max_entry: int
                          ) -> tuple[Optional[int], Optional[str]]:
    n = parse_entry_number_token(token, valid, max_entry)
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
    m = re.match(r"^(\d{1,5})([A-Za-zÁÉÍÓÚÀÂÊÔÃÄÖÜáéíóúàâêôãäöüß].*)$", t)
    if not m:
        return None, None
    num_s, rest = m.groups()
    num = int(num_s)
    if not (num in valid or (not valid and 1 <= num <= max_entry)):
        return None, None
    rest = rest.strip(".,;:!?\"'()")
    if not rest or len(rest) <= 1 or re.fullmatch(r"[rvRV]", rest):
        return None, None
    return num, rest


def has_archive_reference(next_tokens: list[str]) -> bool:
    if not next_tokens:
        return False
    first = next_tokens[0].strip(".,;:")
    if first in ARCHIVE_SIGLA and len(next_tokens) >= 2 and re.search(r"\d", next_tokens[1]):
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
    particles = {"de", "von", "van", "v", "d", "d'", "di", "del", "der", "den",
                 "zu", "zum", "zur", "in", "im", "le", "la", "les", "du", "des",
                 "al", "el"}
    stopwords = {"et", "ad", "ac", "per", "pro", "cum", "m", "arg", "p", "o",
                 "s", "l", "t", "a", "in", "de", "ut", "ex", "ob"}
    reference_sigla = {"A", "S", "L", "V", "T", "R", "E", "I", "IE", "II", "III",
                       "IV", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII",
                       "XIV", "XV", "Arm", "Reg", "Sup", "Plut"}
    abbreviation_pattern = re.compile(
        r"^(Joh|Johs|Henr|Hein|Fr|Guil|Gerh|Wilh|Theod|Phil|Nic|Alex|Arn|Arnd|"
        r"Berth|Conr|Herm|Pet|Marg|Mich|Bart|Laur|Matth|Clem|Ang|Ant|"
        r"Baldew|Bald|Bert|Christ|Corn|Diet|Diedr|Eberh|Evert|Frid|Gott|Heinr|"
        r"Lamb|Ludw|Lud|Otto|Reinh|Rich|Rud|Walt|Wern|Winand)\.?$", re.IGNORECASE)

    def normalize_candidate(tok: str) -> str:
        tok = clean_token(tok).replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        return re.sub(rf"[^{upper}{lower}'\-.]", "", tok)

    def is_reference_like(tok: str) -> bool:
        raw = clean_token(tok)
        stripped = raw.strip(".")
        if not stripped or re.search(r"\d", stripped):
            return True
        if stripped in reference_sigla or re.fullmatch(r"[IVXLCDM]+", stripped):
            return True
        return stripped.isupper() and len(stripped) <= 4

    def is_strong_name(tok: str) -> bool:
        raw = clean_token(tok)
        if not raw or re.search(r"\d", raw) or is_reference_like(raw):
            return False
        norm = normalize_candidate(raw)
        if not norm:
            return False
        low = norm.lower().strip(".")
        if low in stopwords:
            return False
        if abbreviation_pattern.match(norm):
            return True
        stripped = norm.strip(".")
        if len(stripped) < 3:
            return False
        if norm.isupper() and len(norm) <= 4:
            return False
        return bool(re.match(rf"^[{upper}][{lower}]{{1,}}", norm))

    if len(tokens) >= 2 and is_reference_like(tokens[0]) and is_reference_like(tokens[1]):
        return 0
    candidate_tokens = [tokens[0]]
    if tokens[0].lower().strip(".") in particles and len(tokens) >= 2:
        candidate_tokens.append(tokens[1])
    score = sum(5 for tok in candidate_tokens if is_strong_name(tok))
    if len(tokens) >= 2 and is_strong_name(tokens[1]):
        score += 2
    for tok in tokens[:2]:
        if "(" in tok and is_strong_name(tok):
            score += 1
    for tok in tokens[:3]:
        if is_reference_like(tok):
            score -= 3
    return max(score, 0)


def entries_from_tokens(tokens: list[dict], xs: list[float], offset_x: int,
                        valid: set[int], max_entry: int, conf_min: int,
                        name_min: int) -> list[tuple[int, Box]]:
    if not tokens or not xs:
        return []
    margin_limit = float(np.percentile(xs, 5)) + ENTRY_MARGIN_PX
    found: list[tuple[int, Box]] = []
    for tok in tokens:
        if tok["conf"] < conf_min or tok["x"] > margin_limit + 10:
            continue
        entry, attached = parse_entry_candidate(tok["text"], valid, max_entry)
        if entry is None or has_token_left_on_same_line(tok, tokens):
            continue
        next_tokens = []
        if attached:
            next_tokens.append(attached)
        next_tokens.extend(t["text"] for t in
                           tokens_right_on_same_line(tok, tokens) if t["text"].strip())
        if has_archive_reference(next_tokens):
            continue
        if score_name_likeness(next_tokens) < name_min:
            continue
        found.append((entry, (tok["x"] + offset_x, tok["y"], tok["w"], tok["h"])))
    return found


def entries_from_column(col: np.ndarray, offset_x: int, valid: set[int],
                        max_entry: int, conf_min: int, name_min: int
                        ) -> list[tuple[int, Box]]:
    passes = [(ENTRY_TESS_CONFIGS[0], conf_min, name_min),
              (ENTRY_TESS_CONFIGS[1], ENTRY_CONF_FALLBACK, 4),
              (ENTRY_TESS_CONFIGS[0], ENTRY_CONF_FALLBACK, 4)]
    for config, cmin, nmin in passes:
        tokens, xs = col_tokens(col, config)
        found = entries_from_tokens(tokens, xs, offset_x, valid, max_entry,
                                    cmin, nmin)
        if found:
            return found
    return []


def read_margin_entries(img: np.ndarray, valid: set[int],
                        words: Optional[list[Word]] = None,
                        split_x: Optional[int] = None
                        ) -> tuple[list[int], dict[int, Box]]:
    left, right, gutter = split_columns(img)
    left = left[:, :int(left.shape[1] * 0.42)]
    right = right[:, :int(right.shape[1] * 0.42)]
    max_entry = max(valid) if valid else 99999
    found: list[tuple[int, Box]] = []
    for col, offset in ((left, 0), (right, gutter)):
        found.extend(entries_from_column(col, offset, valid, max_entry,
                                         ENTRY_CONF_MIN, 5))

    # Fallback: reuse the full-page OCR tokens when the margin strips found
    # nothing (tight binding, faded margin, etc.).
    if not found and words:
        split = split_x if split_x is not None else img.shape[1] // 2
        groups = [([w for w in words if w.x < split], 0),
                  ([w for w in words if w.x >= split], split)]
        for group, offset in groups:
            tokens = [{"index": i, "text": w.text, "conf": w.conf, "x": w.x,
                       "y": w.y, "w": w.w, "h": w.h}
                      for i, w in enumerate(group)]
            xs = [t["x"] for t in tokens if t["conf"] >= ENTRY_CONF_FALLBACK]
            part = entries_from_tokens(tokens, xs, offset, valid, max_entry,
                                       ENTRY_CONF_MIN, 5)
            if not part:
                part = entries_from_tokens(tokens, xs, offset, valid, max_entry,
                                           ENTRY_CONF_FALLBACK, 4)
            found.extend(part)

    boxes = {}
    for entry, box in found:
        boxes.setdefault(entry, box)
    return sorted(boxes), boxes


def ocr_page(path: Path, valid_entries: set[int]) -> PageOCR:
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    words = ocr_words(img, "--psm 11 --oem 3")
    split_x = split_x_from_words(words, img.shape[1])
    page, page_box, page_source = find_page_number(img)
    entries, entry_boxes = read_margin_entries(img, valid_entries, words, split_x)
    lines, mean_conf = last_printed_lines(img, words, split_x)
    blank = mean_conf < BLANK_MEAN_CONF and not entries
    return PageOCR(scan=path.name, scan_no=scan_number(path) or 0, page=page,
                   page_box=page_box, page_source=page_source, entries=entries,
                   entry_boxes=entry_boxes, lines=lines, mean_conf=mean_conf,
                   blank=blank)


# ---------------------------------------------------------------------------
# OCR cache
# ---------------------------------------------------------------------------

def page_to_dict(page: PageOCR) -> dict:
    return {
        "scan": page.scan, "scan_no": page.scan_no, "page": page.page,
        "page_box": list(page.page_box) if page.page_box else None,
        "page_source": page.page_source, "entries": page.entries,
        "entry_boxes": {str(k): list(v) for k, v in page.entry_boxes.items()},
        "lines": [{"text": ln.text, "x": ln.x, "y": ln.y, "w": ln.w,
                   "h": ln.h, "col": ln.col, "conf": ln.conf} for ln in page.lines],
        "mean_conf": page.mean_conf, "blank": page.blank,
    }


def dict_to_page(data: dict) -> PageOCR:
    box = tuple(data["page_box"]) if data.get("page_box") else None
    boxes = {int(k): tuple(v) for k, v in (data.get("entry_boxes") or {}).items()}
    lines = [Line(**item) for item in data.get("lines") or []]
    return PageOCR(
        scan=data["scan"], scan_no=int(data["scan_no"]), page=data.get("page"),
        page_box=box, page_source=data.get("page_source", "none"),
        entries=[int(x) for x in data.get("entries") or []], entry_boxes=boxes,
        lines=lines, mean_conf=float(data.get("mean_conf") or 0),
        blank=bool(data.get("blank")),
    )


def cache_path(debug_dir: Path) -> Path:
    return debug_dir / "ocr_pages.jsonl"


def load_cache(path: Path) -> dict[str, PageOCR]:
    pages: dict[str, PageOCR] = {}
    if not path.exists():
        return pages
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                page = dict_to_page(json.loads(line))
                pages[page.scan] = page
    return pages


def save_cache(path: Path, pages: dict[str, PageOCR]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(pages.values(), key=lambda p: p.scan_no)
    with open(path, "w", encoding="utf-8") as fh:
        for page in ordered:
            fh.write(json.dumps(page_to_dict(page), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# live view
# ---------------------------------------------------------------------------

class LiveView:
    def __init__(self, out_dir: Path, show: bool, delay_ms: int = 250,
                 recent: int = 8) -> None:
        self.out_dir = out_dir
        self.show = show
        self.delay_ms = delay_ms
        self.recent: deque[np.ndarray] = deque(maxlen=recent)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.window_ok = show

    def update(self, img: np.ndarray, label: str) -> None:
        cv2.imwrite(str(self.out_dir / "latest.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        small = cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])),
                           interpolation=cv2.INTER_AREA)
        self.recent.append(small)
        self._write_recent()
        if self.show and self.window_ok:
            try:
                cv2.imshow("rgpage", img)
                cv2.waitKey(max(1, self.delay_ms))
            except Exception:
                self.window_ok = False

    def _write_recent(self) -> None:
        if not self.recent:
            return
        cols = min(4, len(self.recent))
        rows = (len(self.recent) + cols - 1) // cols
        cell_w = max(img.shape[1] for img in self.recent) + 8
        cell_h = max(img.shape[0] for img in self.recent) + 8
        sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
        for i, img in enumerate(self.recent):
            r, c = divmod(i, cols)
            y0, x0 = r * cell_h + 4, c * cell_w + 4
            sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.imwrite(str(self.out_dir / "recent.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def close(self) -> None:
        if self.show and self.window_ok:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# annotation
# ---------------------------------------------------------------------------

def annotate(img: np.ndarray, page: PageOCR, match: Optional[PageMatch] = None,
             max_h: int = 1500) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_h / h)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    def box(raw: Optional[Box]):
        if not raw:
            return None
        x, y, bw, bh = raw
        return (int(x * scale), int(y * scale), int(bw * scale), int(bh * scale))

    if page.blank:
        lines = [f"scan {page.scan}", "BLANK PAGE"]
    else:
        page_box = box(page.page_box)
        if page_box:
            x, y, bw, bh = page_box
            cv2.rectangle(img, (x - 6, y - 6), (x + bw + 6, y + bh + 6), (0, 200, 0), 4)
            cv2.putText(img, f"page {page.page}", (x, max(22, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        for entry, raw in page.entry_boxes.items():
            b = box(raw)
            if not b:
                continue
            x, y, bw, bh = b
            cv2.rectangle(img, (x - 4, y - 4), (x + bw + 4, y + bh + 4), (255, 80, 0), 3)
            text = str(entry)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            lx, ly = max(2, x - tw - 14), y + max(18, bh)
            cv2.rectangle(img, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 4),
                          (255, 80, 0), -1)
            cv2.putText(img, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        for ln in page.lines:
            b = box((ln.x, ln.y, ln.w, ln.h))
            if b:
                x, y, bw, bh = b
                cv2.rectangle(img, (x - 4, y - 4), (x + bw + 4, y + bh + 4), (0, 140, 255), 2)

        entries_text = ", ".join(map(str, page.entries)) or "-"
        if len(entries_text) > 55:
            if len(page.entries) > 8:
                entries_text = (", ".join(map(str, page.entries[:6])) + ", ..., "
                                + ", ".join(map(str, page.entries[-2:])))
            else:
                entries_text = entries_text[:52] + "..."
        lines = [
            f"scan {page.scan}",
            f"page {page.page}  ({page.page_source})",
            "entries on page: " + entries_text,
            "last printed lines detected" if match is None else "",
        ]
        if match is not None:
            if match.block:
                label = kind_label(match.block.block_id, match.block.kind)
                block_text = f"#{match.block.entry} {match.block.block_id} [{label}]"
                if len(block_text) > 55:
                    block_text = block_text[:52] + "..."
                lines[3] = f"last block: {block_text}"
                lines.append(f"position: {position_label(match.position, match.span_end, page.page)}")
                lines.append(f"score {match.score:.1f}  ({match.status})")
                if match.block.text:
                    tail = " ".join(match.block.text.split())[-95:]
                    lines.append("bloco (XML): " + tail)
                n = len(page.lines)
                tail_idx = set(range(max(0, n - 3), n))
                for i, ln in enumerate(page.lines):
                    bb = box((ln.x, ln.y, ln.w, ln.h))
                    if bb:
                        x, y, bw, bh = bb
                        color = (0, 200, 0) if i in tail_idx else (0, 140, 255)
                        cv2.rectangle(img, (x - 6, y - 6), (x + bw + 6, y + bh + 6),
                                      color, 3)
            else:
                lines[3] = f"last block: none ({match.status})"

    panel_w = min(w - 20, 780)
    panel_h = 26 * len(lines) + 16
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (22, 38 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def contact_sheet(paths: list[Path], labels: list[str], out: Path,
                  cols: int = 4, cell_w: int = 380, cell_h: int = 540) -> None:
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for i, (path, label) in enumerate(zip(paths, labels)):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (cell_w - 12, cell_h - 40), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0, x0 = r * cell_h + 30, c * cell_w + 6
        sheet[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.putText(sheet, label, (c * cell_w + 10, r * cell_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (c * cell_w, r * cell_h),
                      (c * cell_w + cell_w - 1, r * cell_h + cell_h - 1),
                      (180, 180, 180), 1)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------------------
# page numbers and entry mapping
# ---------------------------------------------------------------------------

def normalize_pages(pages: dict[str, PageOCR], first_scan: int,
                    first_page: int, image_dir: Optional[Path] = None) -> None:
    index: dict[int, int] = {}
    if image_dir is not None:
        for i, path in enumerate(ordered_scans(Path(image_dir), first_scan)):
            number = scan_number(path)
            if number is not None:
                index[number] = i
    ordered = sorted(pages.values(), key=lambda p: p.scan_no)
    for i, page in enumerate(ordered):
        expected = first_page + index.get(page.scan_no, i)
        if page.page is None or abs(page.page - expected) > 2:
            page.page = expected
            page.page_source = "inferred"


def clean_page_entries(page: PageOCR) -> None:
    if page.blank or not page.entries:
        page.entries = []
        page.entry_boxes = {}
        return
    values = sorted(page.entries)
    centre = median(values)
    keep = [n for n in values if abs(n - centre) <= 25]
    page.entries = keep
    page.entry_boxes = {n: b for n, b in page.entry_boxes.items() if n in keep}


def build_entry_pages(pages: dict[str, PageOCR]) -> dict[int, int]:
    ordered = sorted(pages.values(), key=lambda p: p.scan_no)
    last_max = 0
    known: dict[int, int] = {}
    for page in ordered:
        clean_page_entries(page)
        if page.page is None:
            continue
        kept = []
        for entry in page.entries:
            if entry < last_max - 2:
                continue
            kept.append(entry)
        if kept:
            last_max = max(last_max, max(kept))
        page.entries = sorted(set(kept))
        page.entry_boxes = {n: b for n, b in page.entry_boxes.items() if n in kept}
        for entry in page.entries:
            known.setdefault(entry, page.page)

    # fill same-page gaps between two detected entries
    by_page: dict[int, list[int]] = {}
    for entry, page_no in known.items():
        by_page.setdefault(page_no, []).append(entry)
    for page_no, entries in by_page.items():
        entries.sort()
        for a, b in zip(entries, entries[1:]):
            if 0 < b - a <= 30:
                for n in range(a + 1, b):
                    known.setdefault(n, page_no)

    # interpolate gaps between the last entry of one anchored page and the
    # first entry of the next anchored page, instead of dumping them all on
    # the earlier page
    page_numbers = sorted(by_page)
    for p1, p2 in zip(page_numbers, page_numbers[1:]):
        last = max(by_page[p1])
        first = min(by_page[p2])
        if 0 < first - last <= 80:
            page_span = p2 - p1
            for n in range(last + 1, first):
                frac = (n - last) / max(1, first - last)
                page_no = p1 + round(frac * page_span)
                known.setdefault(n, min(page_no, p2))
    return known


def docs_near_page(page_no: int, entry_pages: dict[int, int],
                   page_entries: list[int], docs_by_entry: dict[int, list[Doc]],
                   pages_around: int = 2, extra: int = 4,
                   pages_after: int = 1) -> list[Doc]:
    entries = set(page_entries)
    if page_entries:
        top = max(page_entries)
        entries.update(range(top + 1, top + 1 + extra))
        entries.add(min(page_entries) - 1)
    for entry, p in entry_pages.items():
        if page_no - pages_around <= p <= page_no + pages_after:
            entries.add(entry)
    return [doc for entry in sorted(entries) for doc in docs_by_entry.get(entry, [])]


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def fuzzy_score(text: str, doc: Doc) -> tuple[float, float, float]:
    try:
        from rapidfuzz import fuzz
    except Exception:
        import difflib
        score = 100.0 * difflib.SequenceMatcher(None, normalize(text), doc.norm).ratio()
        return score, 0.0, 1.0
    ocr_n = normalize(text)
    if not ocr_n:
        return 0.0, 0.0, 0.0
    score = float(fuzz.partial_ratio(ocr_n, doc.norm))
    try:
        al = fuzz.partial_ratio_alignment(ocr_n, doc.norm)
        return (score, al.dest_start / max(1, len(doc.norm)),
                al.dest_end / max(1, len(doc.norm)))
    except Exception:
        return score, 0.0, 1.0


def _norm_map(text: str) -> tuple[str, list[Optional[int]]]:
    """Normalize like `normalize` but map each normalized char to its original
    index in `text` (used to highlight the matched span in the raw text)."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("ſ", "s").replace("°", "r")
    t = re.sub(r"[\"'`\u2018\u2019\u201c\u201d´]", "v", t)
    t = t.lower()
    out: list[str] = []
    pos: list[Optional[int]] = []
    pending = False
    for i, ch in enumerate(t):
        if ch.isalnum():
            if pending and out:
                out.append(" ")
                pos.append(None)
            out.append(ch)
            pos.append(i)
            pending = False
        else:
            pending = True
    while out and out[0] == " ":
        out.pop(0)
        pos.pop(0)
    while out and out[-1] == " ":
        out.pop()
        pos.pop()
    return "".join(out), pos


def match_highlight(text: str, doc: Doc) -> Optional[tuple[int, int]]:
    """Return (start, end) char indices in `doc.text` aligned with `text`, or None."""
    try:
        from rapidfuzz import fuzz
    except Exception:
        return None
    ocr_n = normalize(text)
    if not ocr_n or not doc.norm:
        return None
    try:
        al = fuzz.partial_ratio_alignment(ocr_n, doc.norm)
    except Exception:
        return None
    if al is None:
        return None
    built, pos = _norm_map(doc.text)
    if built != doc.norm:
        return None
    n_start, n_end = al.dest_start, al.dest_end
    if n_start >= len(pos) or n_end <= n_start:
        return None
    s = pos[n_start] if pos[n_start] is not None else 0
    e = (pos[n_end - 1] + 1) if pos[n_end - 1] is not None else len(doc.text)
    return s, e


def match_text(text: str, docs: list[Doc], first_line: str = "",
               last_line: str = "") -> tuple[Optional[Doc], float, str]:
    ranked: list[tuple[Doc, float, str]] = []
    for doc in docs:
        score, rel_start, rel_end = fuzzy_score(text, doc)
        starts = rel_start <= START_THRESHOLD
        ends = rel_end >= END_THRESHOLD
        if not ends and last_line:
            last_score, _, last_end = fuzzy_score(last_line, doc)
            ends = last_end >= END_THRESHOLD and last_score >= 60
        if not starts and first_line:
            first_score, first_start, _ = fuzzy_score(first_line, doc)
            starts = first_start <= START_THRESHOLD and first_score >= 60
        pos = ("complete-on-page" if starts and ends else
               "ends-here" if ends else
               "starts-here" if starts else "continues")
        ranked.append((doc, score, pos))
    if not ranked:
        return None, 0.0, "continues"
    order = {doc.block_id: i for i, doc in enumerate(docs)}

    def rank_key(item: tuple[Doc, float, str]):
        doc, score, _ = item
        return (round(score, 1), doc.kind != "head", len(doc.norm))

    best = max(ranked, key=rank_key)
    best_doc, best_score, best_pos = best
    if best_doc.kind == "head":
        visible = [
            item for item in ranked
            if item[0].entry == best_doc.entry and item[0].kind != "head"
            and item[2] in ("starts-here", "complete-on-page", "ends-here")
            and item[1] >= 60
        ]
        if visible:
            best_doc, best_score, best_pos = max(
                visible, key=lambda item: (order.get(item[0].block_id, 0), item[1]))
    # If the page's last line clearly starts a *different* block, the page ends
    # with the beginning of the next block — prefer that block (starts-here)
    # over the whole-page match so the boundary is not lost.  Only when the
    # whole-page match actually *ends* on this page (a boundary), the new block
    # is a successor of it, and the last-line start match is strong.
    if (best_doc is not None and last_line
            and best_pos in ("ends-here", "complete-on-page")):
        best_idx = order.get(best_doc.block_id, -1)
        starts_here = []
        for doc, _, _ in ranked:
            if doc.block_id == best_doc.block_id:
                continue
            doc_idx = order.get(doc.block_id, -1)
            if doc_idx <= best_idx or doc_idx - best_idx > 3:
                continue
            last_score, last_start, _ = fuzzy_score(last_line, doc)
            if last_score >= 70 and last_start <= START_THRESHOLD:
                starts_here.append((doc, last_score))
        if starts_here:
            best_doc, best_score = max(starts_here, key=lambda item: item[1])
            best_pos = "starts-here"
    return best_doc, best_score, best_pos
def match_pages(pages: dict[str, PageOCR], entry_pages: dict[int, int],
                docs_by_entry: dict[int, list[Doc]], image_dir: Path,
                use_easy: bool) -> list[PageMatch]:
    matches: list[PageMatch] = []
    ordered = sorted(pages.values(), key=lambda p: p.scan_no)
    for page in ordered:
        if page.blank or page.page is None:
            matches.append(PageMatch(page, None, 0.0, "", "blank-page"))
            continue
        docs = docs_near_page(page.page, entry_pages, page.entries, docs_by_entry,
                              pages_around=3, extra=6, pages_after=4)
        text = " ".join(ln.text for ln in page.lines).strip()
        first_line = page.lines[0].text if page.lines else ""
        last_line = page.lines[-1].text if page.lines else ""
        doc, score, position = match_text(text, docs, first_line, last_line)
        status = "ok" if score >= 55 else "low-score"

        if use_easy and score < 55:
            img = cv2.imdecode(np.fromfile(str(image_dir / page.scan), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is not None:
                y0 = int(img.shape[0] * 0.60)
                easy_lines = []
                for box, easy_text, conf in easy_read(img[y0:]):
                    if len(normalize(easy_text)) < 12:
                        continue
                    x, y, bw, bh = box
                    easy_lines.append(Line(easy_text, x, y + y0, bw, bh, 1, conf * 100))
                if easy_lines:
                    easy_lines.sort(key=lambda ln: (ln.y, ln.x))
                    easy_text = " ".join(ln.text for ln in easy_lines[-2:])
                    easy_first = easy_lines[0].text if easy_lines else ""
                    easy_last = easy_lines[-1].text if easy_lines else ""
                    easy_doc, easy_score, easy_pos = match_text(
                        easy_text, docs, easy_first, easy_last)
                    if easy_doc and easy_score > score + 5:
                        doc, score, position = easy_doc, easy_score, easy_pos
                        status = "ok" if score >= 55 else "low-score"
        matches.append(PageMatch(page, doc, score, position, status))
    return matches


# ---------------------------------------------------------------------------
# page spans
# ---------------------------------------------------------------------------

def build_spans(matches: list[PageMatch], docs: list[Doc],
                entry_pages: dict[int, int]) -> dict[str, Span]:
    by_id = {doc.block_id: doc for doc in docs}
    order = {doc.block_id: i for i, doc in enumerate(docs)}
    occurrence: dict[str, list[PageMatch]] = {}
    for match in matches:
        if match.block and match.page.page is not None:
            occurrence.setdefault(match.block.block_id, []).append(match)

    spans: dict[str, Span] = {}
    last_end = 1
    for doc in docs:
        label = kind_label(doc.block_id, doc.kind)
        occ = occurrence.get(doc.block_id, [])
        source = "anchor"
        warning = ""
        score = 0.0
        if occ:
            first = min(occ, key=lambda m: m.page.page or 0)
            last = max(occ, key=lambda m: m.page.page or 0)
            start = first.page.page
            end = last.page.page
            score = max(m.score for m in occ)
            if last.position in ("starts-here", "continues") and end:
                end += 1
                source = "anchor-extended-end"
        elif doc.is_head:
            page = entry_pages.get(doc.entry)
            start = end = page if page else max(last_end, 1)
            source = "inferred-entry"
        else:
            page = entry_pages.get(doc.entry)
            start = end = page if page else max(last_end, 1)
            source = "inferred-entry"
            warning = "regest page inferred from entry head"

        if occ:
            start = start or 1
        else:
            start = max(start or 1, last_end)
        end = max(end or start, start)
        if end - start > 6:
            end = start + 1
            warning = "span truncated to 2 pages"
        last_end = end
        spans[doc.block_id] = Span(doc.entry, doc.block_id, doc.kind, label,
                                   start, end, source, warning, score)

    # page spans for the matched pages
    for match in matches:
        if match.block and match.block.block_id in spans:
            span = spans[match.block.block_id]
            match.span_start, match.span_end = span.page_start, span.page_end
    return spans


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def write_pages_tsv(matches: list[PageMatch], out: Path,
                    overridden: Optional[set[int]] = None) -> None:
    overridden = overridden or set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["page", "scan", "page_source", "entries_on_page",
                         "last_entry", "last_block", "last_kind",
                         "position", "position_en", "span_start", "span_end",
                         "status", "score", "overridden", "ocr_last_lines"])
        for m in matches:
            page = m.page
            block = m.block
            pos_en = "overridden" if page.page in overridden else \
                position_label(m.position, m.span_end, page.page)
            writer.writerow([
                page.page, page.scan, page.page_source,
                ",".join(map(str, page.entries)),
                block.entry if block else "", block.block_id if block else "",
                block.kind if block else "",
                m.position, pos_en,
                m.span_start, m.span_end, m.status, f"{m.score:.2f}",
                "yes" if page.page in overridden else "",
                " ".join(ln.text for ln in page.lines),
            ])


def write_blocks_tsv(spans: dict[str, Span], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(spans.values(), key=lambda s: (s.entry, s.block_id))
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["entry", "block_id", "label", "kind", "page_start",
                         "page_end", "pages", "status", "source", "warning"])
        for s in rows:
            pages = str(s.page_start) if s.page_start == s.page_end else f"{s.page_start}-{s.page_end}"
            writer.writerow([s.entry, s.block_id, s.label, s.kind, s.page_start,
                             s.page_end, pages,
                             "complete" if s.page_start == s.page_end else "continues",
                             s.source, s.warning])


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def scan_pages(image_dir: Path, cache: Path, valid_entries: set[int],
               first_scan: int, first_page: int, live: LiveView, rebuild: bool,
               from_scan: Optional[int], to_scan: Optional[int],
               limit: Optional[int], progress_cb=None) -> dict[str, PageOCR]:
    pages = {} if rebuild else load_cache(cache)
    paths = ordered_scans(image_dir, first_scan, from_scan, to_scan, limit)
    index = {}
    for i, path in enumerate(ordered_scans(image_dir, first_scan)):
        number = scan_number(path)
        if number is not None:
            index[number] = i
    total = len(paths)
    for i, path in enumerate(paths, 1):
        if path.name in pages and not rebuild:
            continue
        page = ocr_page(path, valid_entries)
        expected = first_page + index.get(page.scan_no, i - 1)
        if page.page is None or abs(page.page - expected) > 2:
            page.page = expected
            page.page_source = "inferred"
        pages[path.name] = page
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        live.update(annotate(img, page), f"scan {page.scan_no} page {page.page}")
        if i % 10 == 0:
            save_cache(cache, pages)
        print(f"[{i}/{total}] scan {page.scan_no:04d} page {page.page} "
              f"entries {len(page.entries)} lines {len(page.lines)} "
              f"conf {page.mean_conf:.0f}", flush=True)
        if progress_cb:
            progress_cb(i, total, page.scan_no, page.page)
    save_cache(cache, pages)
    if limit is not None:
        kept = {p.name for p in paths}
        pages = {k: v for k, v in pages.items() if k in kept}
    return pages


def run(args: argparse.Namespace) -> int:
    xml_path = args.xml or (DEFAULT_XML_DIR / f"rg{args.vol}.xml")
    debug_dir = args.debug_dir or (ROOT / "debug" / f"rg{args.vol}")
    out_dir = args.out_dir or (ROOT / "output")
    cache = cache_path(debug_dir)
    live_dir = args.live_dir or (out_dir / f"rg{args.vol}_live")

    if args.command == "review":
        pages_tsv = args.pages_out or (out_dir / f"rg{args.vol}_pages.tsv")
        serve_review(args, pages_tsv=pages_tsv, xml_path=xml_path, cache=cache,
                     debug_dir=debug_dir, live_dir=live_dir)
        return 0

    docs = load_docs(xml_path)
    docs_by_entry: dict[int, list[Doc]] = {}
    for doc in docs:
        docs_by_entry.setdefault(doc.entry, []).append(doc)
    valid_entries = set(docs_by_entry)

    live = LiveView(live_dir, args.show, args.delay)
    pages = load_cache(cache) if not args.rebuild and cache.exists() else {}
    if args.command in ("run", "scan"):
        pages = scan_pages(args.images, cache, valid_entries, args.first_scan,
                           args.first_page, live, args.rebuild, args.from_scan,
                           args.to_scan, args.limit)
    if args.command == "scan":
        live.close()
        return 0

    if not pages:
        print("No OCR cache. Run `python rgpage.py scan --vol N` first.")
        return 1

    normalize_pages(pages, args.first_scan, args.first_page, args.images)
    entry_pages = build_entry_pages(pages)
    matches = match_pages(pages, entry_pages, docs_by_entry, args.images, args.easyocr)
    spans = build_spans(matches, docs, entry_pages)

    pages_tsv = args.pages_out or (out_dir / f"rg{args.vol}_pages.tsv")
    blocks_tsv = args.blocks_out or (out_dir / f"rg{args.vol}_blocks.tsv")
    write_pages_tsv(matches, pages_tsv)
    write_blocks_tsv(spans, blocks_tsv)

    # live pass with matched block boxes
    for match in matches:
        if not args.show and args.command != "match":
            break
        path = args.images / match.page.scan
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            live.update(annotate(img, match.page, match), f"page {match.page.page}")
    live.close()

    # annotated samples
    candidates = [m for m in matches if m.block and not m.page.blank and m.page.page]
    if args.random and candidates:
        selected = random.Random(args.seed).sample(candidates, min(args.random, len(candidates)))
        selected.sort(key=lambda m: m.page.page or 0)
        img_dir = args.images_out or (out_dir / f"rg{args.vol}_random{len(selected)}")
        img_dir.mkdir(parents=True, exist_ok=True)
        saved, labels = [], []
        for m in selected:
            path = args.images / m.page.scan
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            ann = annotate(img, m.page, m)
            out_img = img_dir / f"page_{m.page.page:04d}_{m.page.scan_no:04d}.jpg"
            cv2.imwrite(str(out_img), ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(out_img)
            labels.append(f"p.{m.page.page} scan {m.page.scan_no:04d}")
        if saved:
            contact_sheet(saved, labels, img_dir / "contact_sheet.jpg")

    ok = sum(1 for m in matches if m.status == "ok")
    start = sum(1 for m in matches if m.position == "starts-here")
    cont = sum(1 for m in matches if m.position == "continues")
    ends = sum(1 for m in matches if m.position == "ends-here")
    complete = sum(1 for m in matches if m.position == "complete-on-page")
    blank = sum(1 for m in matches if m.status == "blank-page")
    print(f"\npages: {len(matches)}  ok: {ok}  blank: {blank}")
    print(f"positions: starts {start}  continues {cont}  ends {ends}  complete {complete}")
    print(f"blocks: {len(spans)}  continuing: {sum(1 for s in spans.values() if s.page_start != s.page_end)}")
    print(f"pages TSV : {pages_tsv}")
    print(f"blocks TSV: {blocks_tsv}")
    if args.random and candidates:
        print(f"sample images: {args.images_out or (out_dir / f'rg{args.vol}_random{min(args.random, len(candidates))}')}")

    if args.command in ("run", "match") and not args.no_review:
        serve_review(args, pages_tsv=pages_tsv, xml_path=xml_path, cache=cache,
                     debug_dir=debug_dir, live_dir=live_dir)
    return 0


def serve_review(args: argparse.Namespace, pages_tsv: Path, xml_path: Path,
                 cache: Path, debug_dir: Path, live_dir: Path) -> None:
    """Launch the review web app (default after run/match)."""
    import webbrowser

    try:
        import review_app
    except ImportError as exc:
        print(f"Review app não disponível: {exc} (instale `flask`)")
        return

    out_dir = args.out_dir or (ROOT / "output")
    cfg = {
        "vol": args.vol,
        "xml_path": xml_path,
        "cache": cache,
        "pages_tsv": pages_tsv,
        "blocks_tsv": args.blocks_out or (out_dir / f"rg{args.vol}_blocks.tsv"),
        "review_json": args.review_json or (out_dir / f"rg{args.vol}_review.json"),
        "page_review_json": out_dir / f"rg{args.vol}_page_review.json",
        "live_dir": live_dir,
        "images_dir": args.images,
        "first_scan": args.first_scan,
        "first_page": args.first_page,
    }
    app = review_app.build_app(cfg)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nReview app: {url}  (Ctrl-C para parar)")
    webbrowser.open(url)
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Find page numbers and RG blocks on RG scans.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "scan", "match", "review"):
        p = sub.add_parser(name)
        p.add_argument("--vol", type=int, default=8)
        p.add_argument("--xml", type=Path)
        p.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
        p.add_argument("--debug-dir", type=Path)
        p.add_argument("--out-dir", type=Path)
        p.add_argument("--live-dir", type=Path)
        p.add_argument("--pages-out", type=Path)
        p.add_argument("--blocks-out", type=Path)
        p.add_argument("--images-out", type=Path)
        p.add_argument("--review-json", type=Path)
        p.add_argument("--first-scan", type=int, default=71)
        p.add_argument("--first-page", type=int, default=1)
        p.add_argument("--from-scan", type=int)
        p.add_argument("--to-scan", type=int)
        p.add_argument("--limit", type=int)
        p.add_argument("--rebuild", action="store_true")
        p.add_argument("--easyocr", action="store_true")
        p.add_argument("--show", action="store_true")
        p.add_argument("--delay", type=int, default=250)
        p.add_argument("--random", type=int, default=0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--no-review", action="store_true")
        p.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
