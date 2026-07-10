#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage3b_manual_sublemma_override.py

Manual, image-assisted tool to fix individual pages in the sublemma
matches TSVs that stage3's automatic/interactive review left unresolved
(low-score, ambiguous, ocr-too-short, or skipped with 's').

Always merges the freshest full run (MATCHES_TSV) with any previously
reviewed decisions (MATCHES_REVIEWED_TSV), so pages from a newer/bigger
run are never hidden behind an older/smaller reviewed file.

For a given page it shows the actual scan (with the OCR last-lines
highlighted, same as stage3's review), lets you filter candidates by
entry number, and set the sub_id directly. Upserts that single row and
writes the result back to MATCHES_REVIEWED_TSV, plus logs the decision to
MANUAL_CORRECTIONS_TSV, same as stage3's own interactive review does.

Image display uses interactive_display.py (matplotlib/Tkinter-based)
instead of cv2.imshow, to avoid the Windows freeze/"Not Responding" issue
that happens when a cv2 HighGUI window sits idle while the terminal blocks
on input().

Modes
-----

1. Interactive queue of everything still unresolved (status != "ok" and
   not already reviewed):

       python stage3b_manual_sublemma_override.py --queue-unresolved

2. Jump straight to one page:

       python stage3b_manual_sublemma_override.py --page 79

Usage
-----
    python stage3b_manual_sublemma_override.py --queue-unresolved
    python stage3b_manual_sublemma_override.py --page 79
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

from pipeline_config import (
    IMAGE_DIR,
    XML_INPUT,
    MATCHES_REVIEWED_TSV,
    MATCHES_TSV,
)

from interactive_display import show_image, close_display, pump_events

# Reuse everything already built in stage3.
from stage3_locate_sublemmas import (
    PageMatch,
    SublemmaDoc,
    load_sublemmas,
    ordered_scan_paths,
    load_image,
    deskew,
    ocr_tokens,
    tokens_to_lines,
    fuzzy_score,
    find_match_position,
    classify_text_position,
    clean_preview,
    make_review_image,
    ranked_candidates,
    append_manual_correction,
    save_matches_report,
)


# ── Load/save the matches TSV(s) back into PageMatch objects ──────────────────

def load_matches_tsv(path: Path) -> list[PageMatch]:
    matches: list[PageMatch] = []

    if not path.exists():
        return matches

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            def _int(key: str) -> Optional[int]:
                raw = (row.get(key) or "").strip()
                return int(raw) if raw.isdigit() else None

            def _float(key: str) -> float:
                raw = (row.get(key) or "").strip()
                try:
                    return float(raw)
                except Exception:
                    return 0.0

            candidate_entries_raw = (row.get("candidate_entries") or "").strip()
            candidate_entries = (
                [int(x) for x in candidate_entries_raw.split("|") if x.strip().isdigit()]
                if candidate_entries_raw else []
            )

            matches.append(PageMatch(
                page=int(row["page"]),
                scan=row.get("scan", ""),
                entry=_int("entry"),
                sub_id=(row.get("sub_id") or "").strip() or None,
                score=_float("score"),
                status=row.get("status", ""),
                candidate_entries=candidate_entries,
                ocr_last_lines=row.get("ocr_last_lines", ""),
                xml_preview=row.get("xml_sublemma_preview", ""),
                matched_is_head=bool(_int("is_head")),
                detected_header_entry=_int("detected_header_entry"),
                reviewed=bool(_int("reviewed")),
                review_note=row.get("review_note", ""),
                match_rel_start=_float("match_rel_start"),
                match_rel_end=_float("match_rel_end"),
                text_position=row.get("text_position", ""),
            ))

    return matches


def load_merged_matches() -> list[PageMatch]:
    """
    Always start from MATCHES_TSV (the freshest, complete run -- e.g. all
    839 pages), then overlay any rows from MATCHES_REVIEWED_TSV (which may
    be from an older/partial run, e.g. --test100 --review) ON TOP, keyed
    by page number.

    This makes sure:
      - pages that only exist in the fresh MATCHES_TSV (never seen by an
        older reviewed file) are still included and correctly flagged as
        unresolved if their status != "ok".
      - pages that WERE already manually reviewed in an older
        MATCHES_REVIEWED_TSV keep their manual decision instead of being
        reset to the automatic one.
    """
    if not MATCHES_TSV.exists():
        raise FileNotFoundError(
            f"{MATCHES_TSV} not found. Run stage3_locate_sublemmas.py first."
        )

    print(f"Base (freshest) matches: {MATCHES_TSV}")
    base_matches = load_matches_tsv(MATCHES_TSV)
    by_page: dict[int, PageMatch] = {m.page: m for m in base_matches}

    if MATCHES_REVIEWED_TSV.exists():
        print(f"Overlaying previously reviewed rows from: {MATCHES_REVIEWED_TSV}")
        reviewed_matches = load_matches_tsv(MATCHES_REVIEWED_TSV)

        overlaid = 0
        skipped_stale = 0

        for rm in reviewed_matches:
            if rm.page not in by_page:
                by_page[rm.page] = rm
                continue

            if rm.reviewed or rm.status == "manual":
                by_page[rm.page] = rm
                overlaid += 1
            else:
                skipped_stale += 1

        print(f"  applied manual/reviewed overrides: {overlaid}")
        print(f"  ignored stale non-reviewed rows   : {skipped_stale}")
    else:
        print("No reviewed file found yet -- starting fresh from MATCHES_TSV only.")

    merged = sorted(by_page.values(), key=lambda m: m.page)
    print(f"Total merged pages: {len(merged)}")

    return merged


def upsert_match(matches: list[PageMatch], updated: PageMatch) -> list[PageMatch]:
    out = [m for m in matches if m.page != updated.page]
    out.append(updated)
    out.sort(key=lambda m: m.page)
    return out


# ── Build a scan_to_path lookup for any page ──────────────────────────────────

def build_scan_to_path() -> dict[str, Path]:
    paths = ordered_scan_paths(IMAGE_DIR, limit=None)
    return {p.name: p for p in paths}


def show_page_for_review(match: PageMatch, scan_to_path: dict[str, Path]) -> None:
    path = scan_to_path.get(match.scan)

    if path is None:
        print(f"Could not find scan file for '{match.scan}'.")
        return

    try:
        img = deskew(load_image(path))
        img_h, img_w = img.shape[:2]
        tokens = ocr_tokens(img)
        lines, split_x = tokens_to_lines(tokens, img_w, img_h)
        last_lines = [ln for ln in lines if ln.text in match.ocr_last_lines]

        header_lines = [
            f"page={match.page}  scan={match.scan}  status={match.status}",
            f"auto entry/sub: {match.entry} / {match.sub_id}",
            f"OCR lines: {clean_preview(match.ocr_last_lines, 160)}",
        ]

        canvas = make_review_image(img, last_lines, split_x, header_lines)
        show_image(canvas, title=f"page {match.page}")
    except Exception as exc:
        print(f"Could not render image: {exc}")


# ── Manual resolution of a single page ────────────────────────────────────────

def resolve_page(match: PageMatch, sublemmas: list[SublemmaDoc], top_n: int = 15) -> Optional[PageMatch]:
    """
    Returns the updated PageMatch, or None if the user chose to skip.
    """
    entry_filter: Optional[int] = None

    while True:
        candidates = [
            s for s in sublemmas
            if (entry_filter is None and s.entry in set(match.candidate_entries))
            or (entry_filter is not None and s.entry == entry_filter)
        ]

        scored = ranked_candidates(match.ocr_last_lines, candidates)

        print("\n" + "-" * 80)
        print(f"page={match.page}  scan={match.scan}  status={match.status}")
        print(f"auto entry/sub: {match.entry} / {match.sub_id}")
        print(f"candidate entries: {'|'.join(map(str, match.candidate_entries))}")
        print(f"OCR lines: {clean_preview(match.ocr_last_lines, 200)}")

        print("\nCandidates:")
        for n, (score, sub) in enumerate(scored[:top_n], start=1):
            head_tag = " [HEAD]" if sub.is_head else ""
            print(f"{n:>2}. score={score:>6.2f} entry={sub.entry:<6} sub={sub.sub_id}{head_tag}")
            print(f"    {clean_preview(sub.xml_text, 220)}")

        pump_events()  # let the window repaint/respond before we block on input()

        cmd = input(
            "\nChoice [number / m SUB_ID / entry N (filter) / s=skip / q=quit page]: "
        ).strip()

        if not cmd:
            continue

        if cmd.lower() == "q":
            return None

        if cmd.lower() == "s":
            return None

        if cmd.lower().startswith("entry "):
            arg = cmd.split(maxsplit=1)[1].strip()
            if arg.isdigit():
                entry_filter = int(arg)
                print(f"Filtering candidates to entry {entry_filter}.")
            else:
                print("Usage: entry N")
            continue

        chosen_sub: Optional[SublemmaDoc] = None

        if cmd.lower().startswith("m "):
            manual_sub_id = cmd.split(maxsplit=1)[1].strip()
            chosen_sub = next((s for s in sublemmas if s.sub_id == manual_sub_id), None)
            if chosen_sub is None:
                print(f"Unknown sub_id: {manual_sub_id}")
                continue

        elif cmd.isdigit():
            choice = int(cmd)
            if not (1 <= choice <= min(top_n, len(scored))):
                print("Invalid candidate number.")
                continue
            _, chosen_sub = scored[choice - 1]

        else:
            print("Unknown command.")
            continue

        score = fuzzy_score(match.ocr_last_lines, chosen_sub.xml_text)
        rel_start, rel_end = find_match_position(match.ocr_last_lines, chosen_sub.xml_text)
        text_position = classify_text_position(rel_start, rel_end)

        match.entry = chosen_sub.entry
        match.sub_id = chosen_sub.sub_id
        match.score = score
        match.status = "manual"
        match.reviewed = True
        match.review_note = "manual override via stage3b"
        match.xml_preview = clean_preview(chosen_sub.xml_text)
        match.matched_is_head = chosen_sub.is_head
        match.match_rel_start = rel_start
        match.match_rel_end = rel_end
        match.text_position = text_position

        append_manual_correction(match)
        print(f"Set page {match.page} -> {chosen_sub.sub_id}")
        return match


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, default=None,
                         help="Jump straight to this page.")
    parser.add_argument("--queue-unresolved", action="store_true",
                         help="Go through every page with status != 'ok' and not yet reviewed.")
    args = parser.parse_args()

    matches = load_merged_matches()

    if not matches:
        print("No matches loaded. Nothing to do.")
        return 1

    print(f"Loading XML sublemmas: {XML_INPUT}")
    sublemmas = load_sublemmas(XML_INPUT)
    print(f"Candidate docs loaded: {len(sublemmas)}")

    scan_to_path = build_scan_to_path()

    if args.page is not None:
        targets = [m for m in matches if m.page == args.page]
        if not targets:
            print(f"Page {args.page} not found in merged matches.")
            return 1
    elif args.queue_unresolved:
        targets = [m for m in matches if m.status != "ok" and not m.reviewed]
        print(f"Unresolved pages to review: {len(targets)}")
    else:
        print("Specify --page N or --queue-unresolved.")
        return 1

    for match in targets:
        show_page_for_review(match, scan_to_path)
        updated = resolve_page(match, sublemmas)

        if updated is not None:
            matches = upsert_match(matches, updated)
            save_matches_report(matches, MATCHES_REVIEWED_TSV)
        else:
            print(f"Skipped page {match.page}.")

    close_display()
    print(f"\nMATCHES_REVIEWED_TSV updated -> {MATCHES_REVIEWED_TSV}")
    print("Next: re-run stage4_build_enriched_xml.py --write-xml to apply changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
