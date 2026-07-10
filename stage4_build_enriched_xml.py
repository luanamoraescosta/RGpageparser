#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage4_build_enriched_xml.py

Stage 4 of the pipeline (final XML-writing stage).

Reconstructs page_start / page_end for EVERY <head>/<sublemma> block in the
XML, using the per-page winning matches from stage3
(MATCHES_REVIEWED_TSV, or MATCHES_TSV as a fallback) plus the
text_position/match_rel_end columns that tell us WHERE inside each
winning block's text the OCR match landed.

Algorithm
---------
The document is one continuous ordered sequence of blocks:

    entry N:   head-0, sub-1, sub-2, ...
    entry N+1: head-0, sub-1, sub-2, ...

For each page we know the winning "anchor" block. Given the anchor
sequence:

    page  9  -> anchor 96-4
    page 10  -> anchor 96-5
    page 11  -> anchor 96-5
    page 12  -> anchor 96-5
    page 13  -> anchor 96-6

  1. A block becomes anchor for the FIRST time on page P -> page_start = P.
  2. Same block anchor again on later page Q -> extend page_end to Q.
  3. A DIFFERENT block becomes anchor on page R -> every block strictly
     between the previous anchor and this new one (that never won any page
     itself) must fit entirely within page R.

Asymmetry between start and end checks
----------------------------------------
text_position is always computed from the LAST visible lines of a page
compared against the winning block's own text. This makes the END
direction reliable: if those last lines don't land near 100% of the
block's text, the block genuinely continues onto a later page.

The START direction is NOT reliable with this same signal: a bottom-of-page
snippet landing in the middle of a block's text is completely normal
whether the block started on this same page (and printed several lines
before reaching the bottom) or started on an earlier page (and is merely
continuing here) -- both produce the same signature. So start_confidence
is recorded as "unknown" and never drives any warning.

Extending incomplete endings
------------------------------
Whenever a block wins as anchor on exactly ONE page, but that page's
text_position for it is NOT a real ending ("starts-here"/"continues"
instead of "ends-here"/"complete-on-page"), the anchor-transition
algorithm alone would still leave page_start == page_end (it looks like
the block "fit entirely" on that page). But we know from text_position
that it did NOT actually end there. Since nothing else won any page
between this block and the next block in document order, the current
block cannot have finished before the page where that next block starts.

extend_incomplete_endings() fixes exactly this: it pushes page_end forward
to the next block's page_start whenever this situation is detected. This
is the minimum correction consistent with the evidence.

Input
-----
debug/lastline_sublemma_matches_reviewed.tsv   (preferred)
debug/lastline_sublemma_matches.tsv            (fallback)

Output
------
debug/sublemma_page_spans.tsv
    Every block (head + sublemma), with page_start/page_end/source/confidence.
    (These extra columns are for YOUR review only -- they are NOT written
    into the XML.)

debug/sublemma_multipage_spans.tsv
    Only blocks where page_start != page_end (the riskiest cases, worth
    spot-checking against the scans).

debug/sublemma_span_warnings.tsv
    Only blocks still flagged after extension as possibly having an
    incomplete/uncertain page_end (e.g. end-of-corpus, or next block never
    resolved to a page). Check these first.

rg8_enriched.xml (--write-xml)
    Each <head>/<sublemma> gets plain:
        <page_start>N</page_start>
        <page_end>M</page_end>
    No extra attributes. Confidence/source info stays only in the TSV
    reports, not in the XML.

Usage
-----
    python stage4_build_enriched_xml.py
    python stage4_build_enriched_xml.py --write-xml
    python stage4_build_enriched_xml.py --write-xml --max-span-warning 6
    python stage4_build_enriched_xml.py --write-xml --no-extend-incomplete-endings
"""

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline_config import (
    XML_INPUT,
    XML_OUTPUT,
    DEBUG_DIR,
    MATCHES_REVIEWED_TSV,
    MATCHES_TSV,
)

DEFAULT_MAX_SPAN_WARNING = 6

SPANS_TSV = DEBUG_DIR / "sublemma_page_spans.tsv"
MULTIPAGE_SPANS_TSV = DEBUG_DIR / "sublemma_multipage_spans.tsv"
SPAN_WARNINGS_TSV = DEBUG_DIR / "sublemma_span_warnings.tsv"

POSITIONS_MEANING_START = {"starts-here", "complete-on-page"}
POSITIONS_MEANING_END = {"ends-here", "complete-on-page"}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BlockDoc:
    sub_id: str
    entry: int
    is_head: bool
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source: str = "none"
    start_confidence: str = ""
    end_confidence: str = ""
    warning: str = ""


@dataclass
class AnchorInfo:
    page: int
    block_index: int
    text_position: str
    rel_start: float
    rel_end: float


# ── Load the global ordered sequence of blocks from the XML ──────────────────

def load_all_blocks_in_order(xml_path: Path) -> list[BlockDoc]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    blocks: list[BlockDoc] = []

    for lemma in root.iter("lemma"):
        m = re.search(r"(\d{4})$", lemma.get("id", ""))

        if not m:
            continue

        entry = int(m.group(1))

        head = lemma.find(".//head")

        if head is not None:
            head_id = head.get("id", "").strip() or f"{lemma.get('id', '')}-0"
            blocks.append(BlockDoc(sub_id=head_id, entry=entry, is_head=True))

        for sub in lemma.findall(".//sublemma"):
            sub_id = sub.get("id", "").strip()

            if not sub_id:
                continue

            blocks.append(BlockDoc(sub_id=sub_id, entry=entry, is_head=False))

    return blocks


# ── Load per-page winning block + text position from stage3's report ─────────

def load_page_anchors(path: Path) -> dict[int, dict]:
    page_data: dict[int, dict] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            page_raw = (row.get("page") or "").strip()
            sub_id = (row.get("sub_id") or "").strip()

            if not page_raw or not sub_id:
                continue

            def _float(key: str) -> float:
                raw = (row.get(key) or "").strip()
                try:
                    return float(raw)
                except Exception:
                    return 0.0

            page_data[int(page_raw)] = {
                "sub_id": sub_id,
                "text_position": (row.get("text_position") or "").strip(),
                "rel_start": _float("match_rel_start"),
                "rel_end": _float("match_rel_end"),
            }

    return page_data


def pick_matches_file() -> Path:
    if MATCHES_REVIEWED_TSV.exists():
        print(f"Using reviewed matches: {MATCHES_REVIEWED_TSV}")
        return MATCHES_REVIEWED_TSV

    if MATCHES_TSV.exists():
        print(f"Reviewed matches not found. Using: {MATCHES_TSV}")
        return MATCHES_TSV

    raise FileNotFoundError(
        "Neither lastline_sublemma_matches_reviewed.tsv nor "
        "lastline_sublemma_matches.tsv were found. Run stage3_locate_sublemmas.py first."
    )


# ── Core reconstruction algorithm ─────────────────────────────────────────────

def build_spans(
    blocks: list[BlockDoc],
    page_data: dict[int, dict],
    max_span_warning: int,
) -> list[BlockDoc]:
    index_of: dict[str, int] = {b.sub_id: i for i, b in enumerate(blocks)}

    anchors: list[AnchorInfo] = []

    for page, data in page_data.items():
        sub_id = data["sub_id"]
        idx = index_of.get(sub_id)

        if idx is None:
            print(f"WARNING: sub_id '{sub_id}' from page {page} not found in XML. Skipping.")
            continue

        anchors.append(
            AnchorInfo(
                page=page,
                block_index=idx,
                text_position=data["text_position"],
                rel_start=data["rel_start"],
                rel_end=data["rel_end"],
            )
        )

    anchors.sort(key=lambda a: a.page)

    if not anchors:
        print("No usable anchors found. Cannot reconstruct spans.")
        return blocks

    for a1, a2 in zip(anchors, anchors[1:]):
        if a2.block_index < a1.block_index:
            b1 = blocks[a1.block_index].sub_id
            b2 = blocks[a2.block_index].sub_id
            print(
                f"WARNING: non-monotonic anchors: page {a1.page} -> {b1} "
                f"then page {a2.page} -> {b2} (goes BACKWARDS in document order). "
                f"Check these pages manually."
            )

    runs: list[list[AnchorInfo]] = []

    for a in anchors:
        if runs and runs[-1][0].block_index == a.block_index:
            runs[-1].append(a)
        else:
            runs.append([a])

    first_run = runs[0]
    first_page = first_run[0].page
    first_idx = first_run[0].block_index

    for i in range(0, first_idx + 1):
        blocks[i].page_end = first_page
        blocks[i].source = "anchor" if i == first_idx else "inferred"
        if blocks[i].page_start is None:
            blocks[i].page_start = first_page

    _apply_run(blocks[first_idx], first_run)

    prev_run = first_run

    for run in runs[1:]:
        prev_idx = prev_run[-1].block_index
        idx = run[0].block_index
        transition_page = run[0].page

        for mid in range(prev_idx + 1, idx):
            blocks[mid].page_start = transition_page
            blocks[mid].page_end = transition_page
            blocks[mid].source = "inferred"
            blocks[mid].start_confidence = "confirmed"
            blocks[mid].end_confidence = "confirmed"

        blocks[idx].page_start = transition_page
        blocks[idx].page_end = run[-1].page
        blocks[idx].source = "anchor"

        _apply_run(blocks[idx], run)

        prev_run = run

    for b in blocks:
        if b.source == "anchor" and (not b.warning) and b.page_start is not None and b.page_end is not None:
            span = b.page_end - b.page_start
            if span > max_span_warning:
                b.warning = (
                    f"long span: {b.page_start}-{b.page_end} "
                    f"({span + 1} pages). Please spot-check against the scans."
                )

    return blocks


def _apply_run(block: BlockDoc, run: list[AnchorInfo]) -> None:
    """
    Cross-check text_position at the edges of a run of consecutive pages
    where the SAME block won as anchor.

    start_confidence is always "unknown" -- the bottom-of-page signal
    cannot reliably distinguish "started here" from "continued from an
    earlier page" for any block longer than ~2 lines (see module
    docstring). end_confidence IS reliable, since it comes from the very
    last visible lines of the page.
    """
    last = run[-1]

    block.start_confidence = "unknown"

    if last.text_position in POSITIONS_MEANING_END:
        block.end_confidence = "confirmed"
    elif last.text_position:
        block.end_confidence = "uncertain"
        block.warning = (
            f"page_end={block.page_end}: text_position='{last.text_position}' "
            f"does not look like a real end (rel_end={last.rel_end:.2f}); "
            f"block may continue onto later page(s)."
        )
    else:
        block.end_confidence = "unknown"


# ── Extend page_end for single-page blocks that did not really end there ─────

def extend_incomplete_endings(blocks: list[BlockDoc]) -> int:
    """
    For every anchor block where page_start == page_end (looks like it fit
    entirely on one page) but end_confidence == "uncertain" (text_position
    shows it did NOT actually end there), push page_end forward to the
    page_start of the very next block in document order -- the minimum
    correction consistent with the evidence, since nothing else won any
    page in between.

    Returns the number of blocks that were extended.
    """
    count = 0

    for i, block in enumerate(blocks):
        if block.source != "anchor":
            continue
        if block.page_start is None or block.page_end is None:
            continue
        if block.page_start != block.page_end:
            continue
        if block.end_confidence != "uncertain":
            continue

        if i + 1 >= len(blocks):
            continue

        next_block = blocks[i + 1]

        if next_block.page_start is None:
            continue

        if next_block.page_start <= block.page_end:
            continue

        old_end = block.page_end
        block.page_end = next_block.page_start
        block.warning = (
            f"page_end extended from {old_end} to {block.page_end}: "
            f"text_position did not indicate a real ending on page {old_end}, "
            f"and no other block won any page in between, so it must "
            f"finish no earlier than the page where '{next_block.sub_id}' starts."
        )
        count += 1

    return count


# ── Reports ──────────────────────────────────────────────────────────────────

def save_spans_report(blocks: list[BlockDoc], path: Path, only_multipage: bool = False) -> int:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "sub_id\tentry\tis_head\tpage_start\tpage_end\tn_pages\tsource\t"
            "start_confidence\tend_confidence\twarning\n"
        )

        for b in blocks:
            if only_multipage:
                if b.page_start is None or b.page_end is None or b.page_start == b.page_end:
                    continue

            if b.page_start is None or b.page_end is None:
                n_pages = ""
            else:
                n_pages = b.page_end - b.page_start + 1

            f.write(
                f"{b.sub_id}\t{b.entry}\t{int(b.is_head)}\t"
                f"{b.page_start if b.page_start is not None else ''}\t"
                f"{b.page_end if b.page_end is not None else ''}\t{n_pages}\t{b.source}\t"
                f"{b.start_confidence}\t{b.end_confidence}\t{b.warning}\n"
            )
            count += 1

    return count


def save_warnings_report(blocks: list[BlockDoc]) -> int:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    flagged = [b for b in blocks if b.warning]

    with open(SPAN_WARNINGS_TSV, "w", encoding="utf-8") as f:
        f.write("sub_id\tentry\tis_head\tpage_start\tpage_end\tstart_confidence\tend_confidence\twarning\n")

        for b in flagged:
            f.write(
                f"{b.sub_id}\t{b.entry}\t{int(b.is_head)}\t"
                f"{b.page_start if b.page_start is not None else ''}\t"
                f"{b.page_end if b.page_end is not None else ''}\t"
                f"{b.start_confidence}\t{b.end_confidence}\t{b.warning}\n"
            )

    return len(flagged)


def print_summary(blocks: list[BlockDoc]) -> None:
    total = len(blocks)
    resolved = sum(1 for b in blocks if b.page_start is not None and b.page_end is not None)
    anchors = sum(1 for b in blocks if b.source == "anchor")
    inferred = sum(1 for b in blocks if b.source == "inferred")
    multipage = sum(
        1 for b in blocks
        if b.page_start is not None and b.page_end is not None and b.page_start != b.page_end
    )
    flagged = sum(1 for b in blocks if b.warning)

    print(f"\nTotal blocks         : {total}")
    print(f"Resolved (start+end) : {resolved} ({100 * resolved / total:.1f}%)")
    print(f"  from anchors       : {anchors}")
    print(f"  inferred           : {inferred}")
    print(f"Unresolved           : {total - resolved}")
    print(f"Spanning >1 page     : {multipage}")
    print(f"Flagged for review   : {flagged}")


# ── XML writing ────────────────────────────────────────────────────────────────

def write_enriched_xml(blocks: list[BlockDoc]) -> None:
    """
    Write <page_start>/<page_end> into every <head> and <sublemma> element
    as plain values, always as the very first two children, with the
    block's original leading text preserved AFTER them (not sandwiched
    before). See the long comment in earlier versions for why this matters
    (ElementTree's .text vs .tail semantics).
    """
    lookup = {b.sub_id: b for b in blocks}

    tree = ET.parse(str(XML_INPUT))
    root = tree.getroot()

    enriched = 0
    skipped = 0

    for tag in ("head", "sublemma"):
        for el in root.iter(tag):
            sid = el.get("id", "")
            b = lookup.get(sid)

            if not b:
                continue

            for old_tag in ("page_start", "page_end", "page"):
                old = el.find(old_tag)
                if old is not None:
                    el.remove(old)

            if b.page_start is None and b.page_end is None:
                skipped += 1
                continue

            original_leading_text = el.text
            el.text = None

            new_elements = []

            if b.page_start is not None:
                start_el = ET.Element("page_start")
                start_el.text = str(b.page_start)
                new_elements.append(start_el)

            if b.page_end is not None:
                end_el = ET.Element("page_end")
                end_el.text = str(b.page_end)
                new_elements.append(end_el)

            for i, new_el in enumerate(new_elements):
                el.insert(i, new_el)

            new_elements[-1].tail = original_leading_text

            enriched += 1

    if XML_OUTPUT.exists():
        import shutil
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = XML_OUTPUT.with_name(XML_OUTPUT.stem + f"_backup_{ts}" + XML_OUTPUT.suffix)
        shutil.copy2(XML_OUTPUT, backup)
        print(f"Backup created -> {backup}")

    tmp = XML_OUTPUT.with_suffix(".tmp.xml")
    tree.write(str(tmp), encoding="utf-8", xml_declaration=True)
    tmp.replace(XML_OUTPUT)

    print(f"\nXML enriched blocks: {enriched}")
    print(f"XML skipped blocks : {skipped}")
    print(f"XML output -> {XML_OUTPUT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--write-xml",
        action="store_true",
        help="Write plain page_start/page_end into the XML output.",
    )
    parser.add_argument(
        "--max-span-warning",
        type=int,
        default=DEFAULT_MAX_SPAN_WARNING,
        help="Warn when a block's span is longer than this many pages. Default: 6.",
    )
    parser.add_argument(
        "--no-extend-incomplete-endings",
        action="store_true",
        help=(
            "Disable the heuristic that extends page_end for single-page "
            "blocks whose text_position shows they did not actually end "
            "on that page."
        ),
    )

    args = parser.parse_args(argv[1:])

    if not XML_INPUT.exists():
        print(f"ERROR: XML not found: {XML_INPUT}", file=sys.stderr)
        return 1

    matches_path = pick_matches_file()

    print(f"Loading global block order from: {XML_INPUT}")
    blocks = load_all_blocks_in_order(XML_INPUT)
    print(f"Total blocks (heads + sublemmas): {len(blocks)}")

    print(f"Loading page anchors from: {matches_path}")
    page_data = load_page_anchors(matches_path)
    print(f"Pages with a usable anchor: {len(page_data)}")

    blocks = build_spans(blocks, page_data, max_span_warning=args.max_span_warning)

    if not args.no_extend_incomplete_endings:
        n_extended = extend_incomplete_endings(blocks)
        print(f"\nExtended page_end for {n_extended} single-page blocks with incomplete endings.")
    else:
        print("\nSkipping incomplete-ending extension (--no-extend-incomplete-endings passed).")

    save_spans_report(blocks, SPANS_TSV, only_multipage=False)
    print(f"\nFull spans report saved: {SPANS_TSV}")

    n_multi = save_spans_report(blocks, MULTIPAGE_SPANS_TSV, only_multipage=True)
    print(f"Multi-page spans report saved: {MULTIPAGE_SPANS_TSV} ({n_multi} blocks span >1 page)")

    n_warn = save_warnings_report(blocks)
    print(f"Warnings report saved: {SPAN_WARNINGS_TSV} ({n_warn} blocks flagged)")

    print_summary(blocks)

    if args.write_xml:
        write_enriched_xml(blocks)
    else:
        print("\n--write-xml not passed: XML was NOT modified. Reports only.")

    print("\nDone.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
