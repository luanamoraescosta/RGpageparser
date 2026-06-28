#!/usr/bin/env python3
"""
Quick XML check:

Reports:
  1. lemmas without sublemma
  2. lemmas without page
  3. lemmas with page inside sublemma
  4. lemmas with page inside reg
  5. lemmas with page elsewhere
"""

import re
import csv
import xml.etree.ElementTree as ET

from pathlib import Path


BASE_DIR = Path(r"/Users/luanamoraescosta/Downloads/RG-Utils/notebooks/RG8")

XML_PATH = BASE_DIR / "rg8_enriched.xml"

DEBUG_DIR = BASE_DIR / "debug" / "xml_check"


def get_entry_num(lemma_id: str):
    m = re.search(r"(\d{4})$", lemma_id or "")
    if not m:
        return None
    return int(m.group(1))


def element_path_contains(parent, child) -> bool:
    """
    True if child is somewhere under parent.
    """
    if parent is None or child is None:
        return False

    for el in parent.iter():
        if el is child:
            return True

    return False


def main():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(str(XML_PATH))
    root = tree.getroot()

    rows_all = []
    rows_no_sublemma = []
    rows_no_page = []
    rows_page_in_sublemma = []
    rows_page_in_reg = []
    rows_page_elsewhere = []

    total = 0

    for lemma in root.iter("lemma"):
        lemma_id = lemma.get("id", "")
        entry_num = get_entry_num(lemma_id)

        if entry_num is None:
            continue

        total += 1

        reg = lemma.find("reg")
        sublemma = lemma.find("./reg/sublemma")

        if sublemma is None:
            sublemma = lemma.find(".//sublemma")

        # Direct pages
        page_in_sublemma = sublemma.find("page") if sublemma is not None else None
        page_in_reg = reg.find("page") if reg is not None else None

        # Any page under lemma
        all_pages = list(lemma.iter("page"))

        has_sublemma = sublemma is not None
        has_page = len(all_pages) > 0

        page_location = ""
        page_text = ""
        page_source = ""

        if page_in_sublemma is not None:
            page_location = "sublemma"
            page_text = page_in_sublemma.text or ""
            page_source = page_in_sublemma.get("source", "")

        elif page_in_reg is not None:
            page_location = "reg"
            page_text = page_in_reg.text or ""
            page_source = page_in_reg.get("source", "")

        elif all_pages:
            page_location = "elsewhere"
            page_text = all_pages[0].text or ""
            page_source = all_pages[0].get("source", "")

        else:
            page_location = "none"

        row = {
            "lemma_id": lemma_id,
            "entry": entry_num,
            "has_sublemma": str(has_sublemma),
            "has_page": str(has_page),
            "page_location": page_location,
            "page_text": page_text.strip(),
            "page_source": page_source,
        }

        rows_all.append(row)

        if not has_sublemma:
            rows_no_sublemma.append(row)

        if not has_page:
            rows_no_page.append(row)

        if page_location == "sublemma":
            rows_page_in_sublemma.append(row)

        elif page_location == "reg":
            rows_page_in_reg.append(row)

        elif page_location == "elsewhere":
            rows_page_elsewhere.append(row)

    def write_tsv(name: str, rows: list[dict]):
        path = DEBUG_DIR / name

        with open(path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "lemma_id",
                "entry",
                "has_sublemma",
                "has_page",
                "page_location",
                "page_text",
                "page_source",
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter="\t",
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

        return path

    all_path = write_tsv("all_lemmas_page_status.tsv", rows_all)
    no_sublemma_path = write_tsv("lemmas_without_sublemma.tsv", rows_no_sublemma)
    no_page_path = write_tsv("lemmas_without_page.tsv", rows_no_page)
    sublemma_path = write_tsv("pages_in_sublemma.tsv", rows_page_in_sublemma)
    reg_path = write_tsv("pages_in_reg.tsv", rows_page_in_reg)
    elsewhere_path = write_tsv("pages_elsewhere.tsv", rows_page_elsewhere)

    print("\n=== XML PAGE CHECK ===")
    print(f"XML: {XML_PATH}")
    print()
    print(f"Total lemmas             : {total}")
    print(f"No sublemma              : {len(rows_no_sublemma)}")
    print(f"No page                  : {len(rows_no_page)}")
    print(f"Page in sublemma         : {len(rows_page_in_sublemma)}")
    print(f"Page in reg              : {len(rows_page_in_reg)}")
    print(f"Page elsewhere           : {len(rows_page_elsewhere)}")
    print()
    print(f"All status               : {all_path}")
    print(f"No sublemma report       : {no_sublemma_path}")
    print(f"No page report           : {no_page_path}")
    print(f"Page in sublemma report  : {sublemma_path}")
    print(f"Page in reg report       : {reg_path}")
    print(f"Page elsewhere report    : {elsewhere_path}")

    if rows_no_sublemma:
        print("\nExamples without sublemma:")
        for row in rows_no_sublemma[:20]:
            print(
                f"  id={row['lemma_id']} "
                f"entry={row['entry']} "
                f"page={row['page_text']} "
                f"location={row['page_location']}"
            )

    if rows_no_page:
        print("\nExamples without page:")
        for row in rows_no_page[:20]:
            print(
                f"  id={row['lemma_id']} "
                f"entry={row['entry']} "
                f"has_sublemma={row['has_sublemma']}"
            )


if __name__ == "__main__":
    main()