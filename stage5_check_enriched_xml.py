#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage5_check_enriched_xml.py

Stage 5 of the pipeline: final audit of XML_OUTPUT.

Reports:
  1. lemmas without any <sublemma>
  2. heads/sublemmas without page_start
  3. heads/sublemmas without page_end
  4. blocks spanning suspiciously many pages
  5. overall coverage summary

Usage
-----
    python stage5_check_enriched_xml.py
"""

import csv
import re
import xml.etree.ElementTree as ET

from pipeline_config import XML_OUTPUT, XML_CHECK_DIR


def get_entry_num(lemma_id: str):
    m = re.search(r"(\d{4})$", lemma_id or "")
    return int(m.group(1)) if m else None


def main() -> int:
    XML_CHECK_DIR.mkdir(parents=True, exist_ok=True)

    if not XML_OUTPUT.exists():
        print(f"ERROR: {XML_OUTPUT} not found. Run stage4_build_enriched_xml.py --write-xml first.")
        return 1

    tree = ET.parse(str(XML_OUTPUT))
    root = tree.getroot()

    rows_all = []
    rows_no_sublemma = []
    rows_missing_start = []
    rows_missing_end = []
    rows_long_span = []

    total_lemmas = 0
    total_blocks = 0

    for lemma in root.iter("lemma"):
        lemma_id = lemma.get("id", "")
        entry_num = get_entry_num(lemma_id)
        if entry_num is None:
            continue

        total_lemmas += 1

        head = lemma.find(".//head")
        subs = lemma.findall(".//sublemma")

        if not subs:
            rows_no_sublemma.append({"lemma_id": lemma_id, "entry": entry_num})

        blocks = ([("head", head)] if head is not None else []) + \
                 [("sublemma", s) for s in subs]

        for kind, el in blocks:
            total_blocks += 1
            sid = el.get("id", "")
            page_start_el = el.find("page_start")
            page_end_el = el.find("page_end")

            page_start = page_start_el.text.strip() if page_start_el is not None and page_start_el.text else ""
            page_end = page_end_el.text.strip() if page_end_el is not None and page_end_el.text else ""

            row = {
                "sub_id": sid,
                "entry": entry_num,
                "kind": kind,
                "page_start": page_start,
                "page_end": page_end,
            }
            rows_all.append(row)

            if not page_start:
                rows_missing_start.append(row)
            if not page_end:
                rows_missing_end.append(row)

            if page_start and page_end:
                try:
                    span = int(page_end) - int(page_start)
                    if span > 6:
                        rows_long_span.append({**row, "span_pages": span + 1})
                except ValueError:
                    pass

    def write_tsv(name: str, rows: list[dict], fieldnames: list[str]):
        path = XML_CHECK_DIR / name
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    all_path = write_tsv(
        "all_blocks_status.tsv", rows_all,
        ["sub_id", "entry", "kind", "page_start", "page_end"],
    )
    no_sublemma_path = write_tsv(
        "lemmas_without_sublemma.tsv", rows_no_sublemma, ["lemma_id", "entry"],
    )
    missing_start_path = write_tsv(
        "blocks_missing_page_start.tsv", rows_missing_start,
        ["sub_id", "entry", "kind", "page_start", "page_end"],
    )
    missing_end_path = write_tsv(
        "blocks_missing_page_end.tsv", rows_missing_end,
        ["sub_id", "entry", "kind", "page_start", "page_end"],
    )
    long_span_path = write_tsv(
        "blocks_long_span.tsv", rows_long_span,
        ["sub_id", "entry", "kind", "page_start", "page_end", "span_pages"],
    )

    print("\n=== ENRICHED XML CHECK ===")
    print(f"XML: {XML_OUTPUT}\n")
    print(f"Total lemmas             : {total_lemmas}")
    print(f"Total blocks (head+sub)  : {total_blocks}")
    print(f"Lemmas without sublemma  : {len(rows_no_sublemma)}")
    print(f"Blocks missing page_start: {len(rows_missing_start)}")
    print(f"Blocks missing page_end  : {len(rows_missing_end)}")
    print(f"Blocks spanning >6 pages : {len(rows_long_span)}")
    print()
    print(f"All blocks report        : {all_path}")
    print(f"No sublemma report       : {no_sublemma_path}")
    print(f"Missing page_start report: {missing_start_path}")
    print(f"Missing page_end report  : {missing_end_path}")
    print(f"Long span report         : {long_span_path}")

    if rows_missing_start:
        print("\nExamples missing page_start:")
        for row in rows_missing_start[:20]:
            print(f"  sub_id={row['sub_id']} entry={row['entry']} kind={row['kind']}")

    if rows_long_span:
        print("\nExamples with long span (spot-check against scans):")
        for row in rows_long_span[:20]:
            print(
                f"  sub_id={row['sub_id']} entry={row['entry']} "
                f"pages={row['page_start']}-{row['page_end']} ({row['span_pages']} pages)"
            )

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())