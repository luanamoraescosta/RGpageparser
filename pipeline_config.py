#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_config.py

Single source of truth for every path and shared constant used across the
whole RG8 pipeline. Every stageN_*.py script imports from here instead of
redefining its own copies of BASE_DIR / XML_INPUT / etc.

Pipeline overview
-----------------

    stage1_build_page_mapping.py
        OCR every scan: read the entry numbers printed in the margin and
        the page number printed at top/bottom of the page.
        -> MAPPING_TSV               (entry -> page, rough)
        -> AMBIGUOUS_MAPPING_TSV     (entries ambiguous between two pages)

    stage2_review_ambiguous_mapping.py   (manual, always shows images)
        For every entry flagged as ambiguous in stage 1, show the two
        candidate page scans side by side and let you pick A/B/skip.
        -> MAPPING_TSV                (updated in place)

    stage3_locate_sublemmas.py
        OCR the last 1-2 lines of each page (right column), fuzzy-match
        them against every <head>/<sublemma> text in the XML, using the
        mapping from stage 1+2 to narrow down candidates.
        -> MATCHES_TSV
        --review flag (manual, always shows the scan image + text diff)
        -> MATCHES_REVIEWED_TSV

    stage4_build_enriched_xml.py
        Reconstruct page_start/page_end for EVERY <head>/<sublemma> block
        from the per-page anchors produced in stage 3, and write the ONE
        final enriched XML.
        -> XML_OUTPUT   <-- the only XML this pipeline writes

    stage5_check_enriched_xml.py
        Audit XML_OUTPUT: which heads/sublemmas are missing
        page_start/page_end, spans that look suspiciously long, etc.
        -> XML_CHECK_DIR/*.tsv

Only ONE enriched XML is ever produced (XML_OUTPUT below). Nothing else in
this pipeline should create a second XML file.
"""

from pathlib import Path

# ── Adjust these for your machine ─────────────────────────────────────────

TESSERACT_CMD = r"C:\Users\moraescosta1\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

BASE_DIR = Path(r"W:\Hiwis\Luana\RGSPAGES\RG-Utils\notebooks\RG8")

IMAGE_DIR = BASE_DIR / "H4 6481 (8,1"

# The ORIGINAL, unmodified source XML. Every stage reads from this -- never
# from XML_OUTPUT -- so re-running any stage always starts from the same
# ground truth.
XML_INPUT = Path(
    r"C:\Users\moraescosta1\Downloads\RGpageparser-main\RGpageparser-main\rg8.xml"
)

# The ONE final enriched XML this whole pipeline produces.
XML_OUTPUT = BASE_DIR / "rg8_enriched_pages_sublemma.xml"

DEBUG_DIR = BASE_DIR / "debug"

# ── Stage 1+2 outputs: entry -> page rough mapping ────────────────────────
MAPPING_TSV = DEBUG_DIR / "entry_to_page_report.tsv"
AMBIGUOUS_MAPPING_TSV = DEBUG_DIR / "ambiguous_missing_between_pages.tsv"
SCAN_REPORT_TSV = DEBUG_DIR / "scan_report.tsv"
MAPPING_RESOLUTIONS_JSON = DEBUG_DIR / "manual_mapping_resolutions.json"
REVIEW_RESOLVED_TSV = DEBUG_DIR / "review_resolved_missing.tsv"
REVIEW_UNSURE_TSV = DEBUG_DIR / "review_unsure_missing.tsv"

# ── Stage 3 outputs: per-page sublemma/head matches ───────────────────────
MATCHES_TSV = DEBUG_DIR / "lastline_sublemma_matches.tsv"
MATCHES_REVIEWED_TSV = DEBUG_DIR / "lastline_sublemma_matches_reviewed.tsv"
MANUAL_CORRECTIONS_TSV = DEBUG_DIR / "manual_sublemma_corrections.tsv"
DEBUG_IMG_DIR = DEBUG_DIR / "ocr_lastline_debug_images"

# ── Stage 4 outputs: reconstructed spans ──────────────────────────────────
SPANS_TSV = DEBUG_DIR / "sublemma_page_spans.tsv"
MULTIPAGE_SPANS_TSV = DEBUG_DIR / "sublemma_multipage_spans.tsv"
SPAN_WARNINGS_TSV = DEBUG_DIR / "sublemma_span_warnings.tsv"

# ── Stage 5 output ─────────────────────────────────────────────────────────
XML_CHECK_DIR = DEBUG_DIR / "xml_check"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

FIRST_SCAN_NO = 71
FIRST_LOGICAL_PAGE = 1
