#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage5_check_enriched_xml.py

Stage 5 of the pipeline: final audit of XML_OUTPUT.

Reports:
  1. containers without any <regest>/<sublemma>
  2. heads/blocks without page_start
  3. heads/blocks without page_end
  4. blocks spanning suspiciously many pages
  5. overall coverage summary

Usage
-----
    python stage5_check_enriched_xml.py
"""

import argparse
import sys
import csv
import xml.etree.ElementTree as ET
from typing import Optional

import rg_xml
from pipeline_config import CFG, add_common_args, configure_from_args


def _page_value(el, tag: str) -> str:
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def _milestone_page_value(el, parent, tag: str) -> str:
    if parent is None:
        return ""
    siblings = list(parent)
    try:
        index = siblings.index(el)
    except ValueError:
        return ""
    for sibling in siblings[index + 1:]:
        if sibling.tag not in ("page_start", "page_end", "page"):
            break
        if sibling.tag == tag:
            return (sibling.text or "").strip()
    return ""


def _block_page_value(el, parent, tag: str) -> str:
    if el.get("milestone") == "start":
        return _milestone_page_value(el, parent, tag)
    return _page_value(el, tag)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(argv[1:] if argv is not None else None)
    configure_from_args(args)

    CFG.xml_check_dir.mkdir(parents=True, exist_ok=True)

    if not CFG.xml_output.exists():
        print(f"ERROR: {CFG.xml_output} not found. Run stage4_build_enriched_xml.py --write-xml first.")
        return 1

    tree = ET.parse(str(CFG.xml_output))
    root = tree.getroot()

    rows_all = []
    rows_no_sublemma = []
    rows_missing_start = []
    rows_missing_end = []
    rows_long_span = []

    total_lemmas = 0
    total_blocks = 0

    parent_map = {child: parent for parent in root.iter() for child in parent}
    blocks_by_entry: dict[int, list[rg_xml.Block]] = {}
    for block in rg_xml.iter_blocks(root):
        blocks_by_entry.setdefault(block.entry, []).append(block)

    for container, entry_num in rg_xml.iter_container_entries(root):
        total_lemmas += 1
        container_blocks = blocks_by_entry.get(entry_num, [])
        subs = [b for b in container_blocks if not b.is_head]

        if not subs:
            rows_no_sublemma.append({"lemma_id": container.get("id", ""), "entry": entry_num})

        for block in container_blocks:
            total_blocks += 1
            parent = parent_map.get(block.element)
            page_start = _block_page_value(block.element, parent, "page_start")
            page_end = _block_page_value(block.element, parent, "page_end")

            row = {
                "sub_id": block.block_id,
                "entry": entry_num,
                "kind": block.kind,
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
        path = CFG.xml_check_dir / name
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
        "containers_without_sublemma.tsv", rows_no_sublemma, ["lemma_id", "entry"],
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
    print(f"XML: {CFG.xml_output}\n")
    print(f"Total containers         : {total_lemmas}")
    print(f"Total blocks             : {total_blocks}")
    print(f"Containers without regest: {len(rows_no_sublemma)}")
    print(f"Blocks missing page_start: {len(rows_missing_start)}")
    print(f"Blocks missing page_end  : {len(rows_missing_end)}")
    print(f"Blocks spanning >6 pages : {len(rows_long_span)}")
    print()
    print(f"All blocks report        : {all_path}")
    print(f"No regest report         : {no_sublemma_path}")
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
    raise SystemExit(main(sys.argv))