#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime configuration for the RG page parser.

Every stage imports the shared ``CFG`` object and the CLI helpers below. Paths
are no longer hard-coded: a volume is selected with ``--volume`` (or ``--xml``),
everything else is derived from it and can be overridden on the command line
or through ``RG_*`` environment variables.

Defaults assume the repository is checked out next to ``RG_data``::

    RGPARSER/
    ├── RGpageparser/       (this repository)
    └── RG_data/rg_xml/rg1.xml ... rg10.xml
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_DIR = Path(__file__).resolve().parent

DEFAULT_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def _env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() else None


def _detect_rg_data_dir() -> Path:
    env = _env_path("RG_DATA_DIR")
    if env:
        return env
    for candidate in (REPO_DIR.parent / "RG_data" / "rg_xml", REPO_DIR / "RG_data" / "rg_xml"):
        if candidate.exists():
            return candidate
    return REPO_DIR.parent / "RG_data" / "rg_xml"


def _detect_default_volume() -> Optional[int]:
    for candidate in (REPO_DIR.parent / "H4 6481 (8,1", REPO_DIR / "H4 6481 (8,1"):
        if candidate.is_dir():
            return 8
    return None


def _volume_from_name(name: str) -> Optional[int]:
    stem = Path(name).stem.lower()
    if stem.startswith("rg") and stem[2:].isdigit():
        return int(stem[2:])
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


@dataclass
class Config:
    volume: Optional[int] = None
    base_dir: Optional[Path] = None
    _images: Optional[Path] = None
    _xml: Optional[Path] = None
    _output: Optional[Path] = None
    _debug: Optional[Path] = None
    tesseract_cmd: str = "tesseract"
    first_scan_no: int = 71
    first_logical_page: int = 1
    image_extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS
    rg_data_dir: Path = _detect_rg_data_dir()

    @property
    def base(self) -> Path:
        return self.base_dir or REPO_DIR

    @property
    def xml_input(self) -> Path:
        if self._xml is not None:
            return self._xml
        if self.volume is None:
            return self.rg_data_dir / "rg.xml"
        return self.rg_data_dir / f"rg{self.volume}.xml"

    @property
    def image_dir(self) -> Path:
        if self._images is not None:
            return self._images
        suffix = f"rg{self.volume}" if self.volume is not None else ""
        candidates = [self.base / "scans" / suffix]
        if self.volume == 8:
            candidates += [
                REPO_DIR.parent / "H4 6481 (8,1",
                self.base / "H4 6481 (8,1",
            ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]

    @property
    def debug_dir(self) -> Path:
        if self._debug is not None:
            return self._debug
        suffix = f"rg{self.volume}" if self.volume is not None else ""
        return self.base / "debug" / suffix

    @property
    def xml_output(self) -> Path:
        if self._output is not None:
            return self._output
        suffix = f"rg{self.volume}" if self.volume is not None else ""
        return self.base / "output" / f"{suffix}_enriched_pages.xml"

    @property
    def xml_check_dir(self) -> Path:
        return self.debug_dir / "xml_check"

    @property
    def mapping_tsv(self) -> Path:
        return self.debug_dir / "entry_to_page_report.tsv"

    @property
    def ambiguous_mapping_tsv(self) -> Path:
        return self.debug_dir / "ambiguous_missing_between_pages.tsv"

    @property
    def scan_report_tsv(self) -> Path:
        return self.debug_dir / "scan_report.tsv"

    @property
    def mapping_resolutions_json(self) -> Path:
        return self.debug_dir / "manual_mapping_resolutions.json"

    @property
    def review_resolved_tsv(self) -> Path:
        return self.debug_dir / "review_resolved_missing.tsv"

    @property
    def review_unsure_tsv(self) -> Path:
        return self.debug_dir / "review_unsure_missing.tsv"

    @property
    def mapping_cache(self) -> Path:
        return self.debug_dir / "page_mapping.json"

    @property
    def manual_mapping_overrides_tsv(self) -> Path:
        return self.debug_dir / "manual_mapping_overrides.tsv"

    @property
    def matches_tsv(self) -> Path:
        return self.debug_dir / "lastline_block_matches.tsv"

    @property
    def matches_reviewed_tsv(self) -> Path:
        return self.debug_dir / "lastline_block_matches_reviewed.tsv"

    @property
    def manual_corrections_tsv(self) -> Path:
        return self.debug_dir / "manual_block_corrections.tsv"

    @property
    def debug_img_dir(self) -> Path:
        return self.debug_dir / "ocr_lastline_debug_images"

    @property
    def spans_tsv(self) -> Path:
        return self.debug_dir / "block_page_spans.tsv"

    @property
    def multipage_spans_tsv(self) -> Path:
        return self.debug_dir / "block_multipage_spans.tsv"

    @property
    def span_warnings_tsv(self) -> Path:
        return self.debug_dir / "block_span_warnings.tsv"


CFG = Config(
    volume=_env_int("RG_VOLUME"),
    base_dir=_env_path("RG_BASE_DIR"),
    _images=_env_path("RG_IMAGES"),
    _xml=_env_path("RG_XML"),
    _output=_env_path("RG_OUTPUT"),
    _debug=_env_path("RG_DEBUG_DIR"),
    tesseract_cmd=os.environ.get("RG_TESSERACT", "tesseract"),
    first_scan_no=_env_int("RG_FIRST_SCAN_NO") or 71,
    first_logical_page=_env_int("RG_FIRST_LOGICAL_PAGE") or 1,
    rg_data_dir=_detect_rg_data_dir(),
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("RG paths")
    group.add_argument("--volume", type=int, default=None,
                       help="RG volume number (selects rgN.xml under RG_data/rg_xml).")
    group.add_argument("--xml", type=Path, default=None,
                       help="Explicit source XML (overrides --volume).")
    group.add_argument("--images", type=Path, default=None,
                       help="Directory containing the page scans.")
    group.add_argument("--base-dir", type=Path, default=None,
                       help="Working directory for debug/output folders.")
    group.add_argument("--debug-dir", type=Path, default=None,
                       help="Explicit directory for intermediate reports.")
    group.add_argument("--output", type=Path, default=None,
                       help="Explicit enriched XML output path.")
    group.add_argument("--tesseract", default=None,
                       help="Path to the tesseract executable.")
    group.add_argument("--first-scan-no", type=int, default=None,
                       help="Physical number of the first scan to process.")
    group.add_argument("--first-logical-page", type=int, default=None,
                       help="Printed page number of the first processed scan.")


def configure_from_args(args: argparse.Namespace) -> Config:
    volume = getattr(args, "volume", None)
    xml = getattr(args, "xml", None)
    images = getattr(args, "images", None)
    base_dir = getattr(args, "base_dir", None)
    debug_dir = getattr(args, "debug_dir", None)
    output = getattr(args, "output", None)
    tesseract = getattr(args, "tesseract", None)
    first_scan_no = getattr(args, "first_scan_no", None)
    first_logical_page = getattr(args, "first_logical_page", None)

    if xml is not None:
        CFG._xml = xml.expanduser()
        if volume is None:
            volume = _volume_from_name(xml.name)
    if volume is None and xml is None and CFG.volume is None:
        volume = _detect_default_volume()
    if volume is not None:
        CFG.volume = volume
    if images is not None:
        CFG._images = images.expanduser()
    if base_dir is not None:
        CFG.base_dir = base_dir.expanduser()
    if debug_dir is not None:
        CFG._debug = debug_dir.expanduser()
    if output is not None:
        CFG._output = output.expanduser()
    if tesseract is not None:
        CFG.tesseract_cmd = tesseract
    if first_scan_no is not None:
        CFG.first_scan_no = first_scan_no
    if first_logical_page is not None:
        CFG.first_logical_page = first_logical_page
    return CFG
