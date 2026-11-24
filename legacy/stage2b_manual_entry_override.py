#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage2b_manual_entry_override.py

Manual, image-assisted tool to add/fix entry -> page mappings that fell
through the cracks.

Two independent modes
----------------------

1. Entry-lookup mode (original): you know an entry number is missing/wrong,
   and want to look at the neighboring pages to figure out which page it
   belongs to.

       python stage2b_manual_entry_override.py

   or directly, if you already know the answer:

       python stage2b_manual_entry_override.py --entry 441 --page 71

2. Page mode (new): a WHOLE PAGE had problems (stage1's OCR missed some
   entries, or assigned wrong ones to it). Shows that single page's scan,
   lists which entries are currently mapped to it, and lets you type the
   entries that actually belong there.

       python stage2b_manual_entry_override.py --page 381

   While in page mode:
     - type one or more entry numbers (space or comma separated) to map
       them to this page, e.g.:  444 445 446
     - prefix an entry with '-' to REMOVE it from this page's mapping,
       e.g.:  -448   (only removes it if it currently points to this page)
     - type 'list' to reprint the entries currently mapped to this page
     - type 'q' to finish this page

Both modes write directly into CFG.mapping_tsv (upsert) and log every manual
decision to CFG.manual_mapping_overrides_tsv for traceability.

Usage
-----
    python stage2b_manual_entry_override.py
    python stage2b_manual_entry_override.py --entry 441 --page 71
    python stage2b_manual_entry_override.py --page 381
"""

import argparse
import sys
import re
from typing import Optional
import cv2

from imaging import load_display_image
from pipeline_config import CFG, add_common_args, configure_from_args

# Reuse the loader/display helpers already written in stage2.
from stage2_review_ambiguous_mapping import (
    load_mapping_tsv,
    make_side_by_side,
    read_scan_report_page_to_image,
    save_mapping_tsv,
)

_WINDOW_NAME = "manual-entry-override"
_window_open = False
_window_thread_started = False


def _ensure_window_thread() -> None:
    """
    Start OpenCV's background window-event thread once.

    Without this, the cv2 window only processes OS messages (repaint,
    hover, click) when cv2.waitKey() is called. Since our workflow blocks
    on input() right after showing the image, the window would otherwise
    freeze / show "Not Responding" whenever you hover or click on it while
    the terminal is waiting for your answer.
    """
    global _window_thread_started

    if _window_thread_started:
        return

    try:
        cv2.startWindowThread()
    except Exception:
        # Not available/needed on some builds; safe to ignore.
        pass

    _window_thread_started = True


def show_image(img, title: str = "") -> None:
    global _window_open

    _ensure_window_thread()

    if not _window_open:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
        _window_open = True

    if title:
        cv2.setWindowTitle(_WINDOW_NAME, title)

    cv2.imshow(_WINDOW_NAME, img)
    cv2.waitKey(1)  # initial paint


def close_display() -> None:
    global _window_open

    if _window_open:
        cv2.destroyAllWindows()
        _window_open = False
# ── Data loading ──────────────────────────────────────────────────────────────

def append_manual_log(entry: int, page: Optional[int], note: str) -> None:
    CFG.debug_dir.mkdir(parents=True, exist_ok=True)
    exists = CFG.manual_mapping_overrides_tsv.exists()
    with open(CFG.manual_mapping_overrides_tsv, "a", encoding="utf-8") as f:
        if not exists:
            f.write("entry\tpage\tnote\n")
        f.write(f"{entry}\t{page if page is not None else ''}\t{note}\n")


def find_neighbors(entry: int, mapping: dict[int, int]) -> tuple[Optional[int], Optional[int]]:
    known = sorted(mapping.keys())
    left = max((e for e in known if e < entry), default=None)
    right = min((e for e in known if e > entry), default=None)
    return left, right


# ── Entry-lookup mode (original) ──────────────────────────────────────────────

def run_interactive() -> None:
    mapping = load_mapping_tsv(CFG.mapping_tsv)
    page_to_image = read_scan_report_page_to_image()

    print("\n=== MANUAL ENTRY -> PAGE OVERRIDE ===")
    print(f"Current mapping size: {len(mapping)} entries")
    print("Type an entry number to look it up, or 'q' to quit.\n")

    while True:
        raw = input("Entry number (or q): ").strip()

        if raw.lower() == "q":
            break

        if not raw.isdigit():
            print("Please type a number, or 'q' to quit.")
            continue

        entry = int(raw)

        if entry in mapping:
            print(f"Entry {entry} is already mapped to page {mapping[entry]}.")
            overwrite = input("Overwrite? [y/N]: ").strip().lower()
            if overwrite != "y":
                continue

        left_entry, right_entry = find_neighbors(entry, mapping)
        left_page = mapping.get(left_entry) if left_entry is not None else None
        right_page = mapping.get(right_entry) if right_entry is not None else None

        print(f"Nearest known entries: left={left_entry} (page {left_page})  "
              f"right={right_entry} (page {right_page})")

        img_a = None
        img_b = None
        header_lines = [f"Looking up entry {entry}"]

        if left_page is not None and left_page in page_to_image:
            img_a_path = CFG.image_dir / page_to_image[left_page]
            if img_a_path.exists():
                img_a = load_display_image(img_a_path)
                header_lines.append(f"A: page {left_page} (entry {left_entry}), {img_a_path.name}")

        if right_page is not None and right_page in page_to_image:
            img_b_path = CFG.image_dir / page_to_image[right_page]
            if img_b_path.exists():
                img_b = load_display_image(img_b_path)
                header_lines.append(f"B: page {right_page} (entry {right_entry}), {img_b_path.name}")

        if img_a is not None and img_b is not None:
            canvas = make_side_by_side(img_a, img_b, header_lines)
            show_image(canvas, title=f"entry {entry}")
        elif img_a is not None:
            show_image(img_a, title=f"entry {entry} (only left neighbor known)")
        elif img_b is not None:
            show_image(img_b, title=f"entry {entry} (only right neighbor known)")
        else:
            print("No neighboring page image available for visual reference.")

        page_raw = input(f"Page number for entry {entry} (blank to skip): ").strip()

        if not page_raw:
            print("Skipped.")
            continue

        if not page_raw.isdigit():
            print("Invalid page number, skipped.")
            continue

        page = int(page_raw)
        mapping[entry] = page
        save_mapping_tsv(CFG.mapping_tsv, mapping)
        append_manual_log(entry, page, "manual interactive (entry lookup)")
        print(f"Saved: entry {entry} -> page {page}")

    close_display()
    print(f"\nMAPPING_TSV updated -> {CFG.mapping_tsv}")


def run_direct(entry: int, page: int) -> None:
    mapping = load_mapping_tsv(CFG.mapping_tsv)
    old = mapping.get(entry)
    mapping[entry] = page
    save_mapping_tsv(CFG.mapping_tsv, mapping)
    append_manual_log(entry, page, "manual direct (--entry/--page)")

    if old is None:
        print(f"Added: entry {entry} -> page {page}")
    elif old != page:
        print(f"Updated: entry {entry}  {old} -> {page}")
    else:
        print(f"Entry {entry} was already page {page}, no change.")

    print(f"CFG.mapping_tsv updated -> {CFG.mapping_tsv}")


# ── Page mode (new) ────────────────────────────────────────────────────────────

def entries_on_page(mapping: dict[int, int], page: int) -> list[int]:
    return sorted(e for e, p in mapping.items() if p == page)


def parse_entry_tokens(raw: str) -> tuple[list[int], list[int]]:
    """
    Parse a line like "444 445 446" or "444,445,-448" into:
        (entries_to_add, entries_to_remove)

    A token prefixed with '-' means "remove this entry".
    """
    tokens = [t for t in re.split(r"[,\s]+", raw.strip()) if t]

    to_add: list[int] = []
    to_remove: list[int] = []

    for tok in tokens:
        if tok.startswith("-") and tok[1:].isdigit():
            to_remove.append(int(tok[1:]))
        elif tok.isdigit():
            to_add.append(int(tok))
        else:
            print(f"  ignoring unrecognized token: '{tok}'")

    return to_add, to_remove


def run_page_mode(page: int) -> None:
    mapping = load_mapping_tsv(CFG.mapping_tsv)
    page_to_image = read_scan_report_page_to_image()

    image_name = page_to_image.get(page)

    if not image_name:
        print(f"ERROR: no scan image found for page {page} in {CFG.scan_report_tsv}.")
        return

    image_path = CFG.image_dir / image_name

    if not image_path.exists():
        print(f"ERROR: image file does not exist: {image_path}")
        return

    print(f"\n=== MANUAL PAGE ENTRY OVERRIDE: page {page} ===")
    print(f"Image: {image_name}")

    def show_current_page() -> None:
        current = entries_on_page(mapping, page)
        print(f"\nEntries currently mapped to page {page}: {current if current else '(none)'}")

    img = load_display_image(image_path)
    show_image(img, title=f"page {page} - {image_name}")

    show_current_page()

    print(
        "\nType entry numbers that belong on this page (space or comma "
        "separated), e.g.:  444 445 446\n"
        "Prefix with '-' to remove an entry from this page, e.g.: -448\n"
        "Type 'list' to reprint current entries, or 'q' to finish this page.\n"
    )

    while True:
        raw = input(f"[page {page}] entries: ").strip()

        if not raw:
            continue

        if raw.lower() == "q":
            break

        if raw.lower() == "list":
            show_current_page()
            continue

        to_add, to_remove = parse_entry_tokens(raw)

        for entry in to_remove:
            if mapping.get(entry) == page:
                del mapping[entry]
                append_manual_log(entry, None, f"manual removed from page {page}")
                print(f"  removed entry {entry} from page {page}")
            elif entry in mapping:
                print(
                    f"  entry {entry} is mapped to page {mapping[entry]}, not {page}. "
                    f"Not removed (edit that page instead)."
                )
            else:
                print(f"  entry {entry} was not mapped anywhere. Nothing to remove.")

        for entry in to_add:
            old = mapping.get(entry)
            mapping[entry] = page
            if old is None:
                append_manual_log(entry, page, f"manual added to page {page}")
                print(f"  added entry {entry} -> page {page}")
            elif old != page:
                append_manual_log(entry, page, f"manual moved from page {old} to page {page}")
                print(f"  moved entry {entry}: page {old} -> {page}")
            else:
                print(f"  entry {entry} already on page {page}, no change")

        if to_add or to_remove:
            save_mapping_tsv(CFG.mapping_tsv, mapping)
            show_current_page()

    close_display()
    print(f"\nMAPPING_TSV updated -> {CFG.mapping_tsv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", type=int, default=None,
                         help="Entry-lookup mode: entry number to set directly.")
    parser.add_argument("--page", type=int, default=None,
                         help="Page mode: show this page and edit its entries. "
                              "If used together with --entry, sets that single "
                              "entry -> page directly (same as before).")
    add_common_args(parser)
    args = parser.parse_args(argv[1:] if argv is not None else None)
    configure_from_args(args)

    if args.entry is not None and args.page is not None:
        run_direct(args.entry, args.page)
    elif args.page is not None:
        run_page_mode(args.page)
    else:
        run_interactive()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
