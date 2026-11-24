#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared RG XML model.

Supports the current RG_data layout (``<rg><petentenvita><head/><regest/>``)
and the legacy layout (``<rg><lemma><reg><head/><sublemma/>``). All stages use
this module instead of parsing the XML themselves.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

CONTAINER_TAGS = ("petentenvita", "lemma")
HEAD_TAG = "head"
BLOCK_TAGS = ("regest", "sublemma")
SKIP_TEXT_TAGS = {"page", "page_start", "page_end", "seite"}


@dataclass
class Block:
    element: ET.Element
    block_id: str
    entry: int
    kind: str
    text: str

    @property
    def is_head(self) -> bool:
        return self.kind == HEAD_TAG


def entry_number(value: Optional[str]) -> Optional[int]:
    """Extract the entry number from a container id (new and legacy schemes)."""
    if not value:
        return None
    base = value.split("-", 1)[0]
    digits = re.sub(r"\D", "", base)
    if not digits:
        return None
    if len(digits) >= 8:
        return int(digits[3:]) or None
    return int(digits)


def volume_of(root: ET.Element) -> Optional[int]:
    raw = root.get("vol", "").strip()
    if raw.isdigit():
        return int(raw)
    return None


def iter_containers(root: ET.Element) -> Iterator[ET.Element]:
    for el in root.iter():
        if el.tag in CONTAINER_TAGS:
            yield el


def element_text(el: ET.Element) -> str:
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag in SKIP_TEXT_TAGS:
            return
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return " ".join(parts)


def _milestone_text(start_el: ET.Element, parent: ET.Element) -> str:
    """Collect the text between ``milestone="start"`` and its matching end."""
    ref = (start_el.get("id") or "").strip()
    siblings = list(parent)
    try:
        index = siblings.index(start_el)
    except ValueError:
        return ""
    parts: list[str] = []
    if start_el.tail:
        parts.append(start_el.tail)
    for sibling in siblings[index + 1:]:
        if (
            sibling.tag == start_el.tag
            and sibling.get("milestone") == "end"
            and (sibling.get("ref") or "").strip() == ref
        ):
            break
        if sibling.tag in SKIP_TEXT_TAGS:
            continue
        if sibling.text:
            parts.append(sibling.text)
        for child in sibling.iter():
            if child is not sibling and child.tag not in SKIP_TEXT_TAGS and child.text:
                parts.append(child.text)
            if child is not sibling and child.tail:
                parts.append(child.tail)
        if sibling.tail:
            parts.append(sibling.tail)
    return " ".join(parts)


def iter_blocks(root: ET.Element) -> Iterator[Block]:
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for container in iter_containers(root):
        entry = entry_number(container.get("id"))
        if entry is None:
            continue
        for el in container.iter():
            if el.tag != HEAD_TAG and el.tag not in BLOCK_TAGS:
                continue
            milestone = el.get("milestone")
            if milestone == "end":
                continue
            block_id = (el.get("id") or "").strip()
            if not block_id:
                continue
            if milestone == "start":
                text = _milestone_text(el, parent_map.get(el, container))
            else:
                text = element_text(el)
            kind = "regest" if el.tag in ("regest", "sublemma") else el.tag
            yield Block(
                element=el,
                block_id=block_id,
                entry=entry,
                kind=kind,
                text=text,
            )


def load_blocks(xml_path: Path) -> list[Block]:
    root = ET.parse(str(xml_path)).getroot()
    return list(iter_blocks(root))


def entry_numbers(xml_path: Path) -> set[int]:
    if not xml_path.exists():
        return set()
    root = ET.parse(str(xml_path)).getroot()
    return {
        entry
        for container in iter_containers(root)
        if (entry := entry_number(container.get("id"))) is not None
    }


def iter_container_entries(root: ET.Element) -> Iterator[tuple[ET.Element, int]]:
    for container in iter_containers(root):
        entry = entry_number(container.get("id"))
        if entry is not None:
            yield container, entry


def _write_milestone_tags(
    el: ET.Element,
    page_start: Optional[int],
    page_end: Optional[int],
    parent: Optional[ET.Element],
) -> bool:
    if parent is None:
        return False

    siblings = list(parent)
    index = siblings.index(el)
    insert_at = index + 1
    while insert_at < len(siblings) and siblings[insert_at].tag in ("page_start", "page_end", "page"):
        parent.remove(siblings[insert_at])
        siblings = list(parent)
        insert_at = siblings.index(el) + 1

    if page_start is None and page_end is None:
        return False

    new_elements: list[ET.Element] = []
    if page_start is not None:
        start_el = ET.Element("page_start")
        start_el.text = str(page_start)
        new_elements.append(start_el)
    if page_end is not None:
        end_el = ET.Element("page_end")
        end_el.text = str(page_end)
        new_elements.append(end_el)

    for offset, new_el in enumerate(new_elements, start=1):
        parent.insert(index + offset, new_el)
    return True


def write_page_tags(
    el: ET.Element,
    page_start: Optional[int],
    page_end: Optional[int],
    parent: Optional[ET.Element] = None,
) -> bool:
    if el.get("milestone") == "start":
        return _write_milestone_tags(el, page_start, page_end, parent)

    for old_tag in ("page_start", "page_end", "page"):
        old = el.find(old_tag)
        if old is not None:
            el.remove(old)

    if page_start is None and page_end is None:
        return False

    original_leading = el.text
    el.text = None

    new_elements: list[ET.Element] = []
    if page_start is not None:
        start_el = ET.Element("page_start")
        start_el.text = str(page_start)
        new_elements.append(start_el)
    if page_end is not None:
        end_el = ET.Element("page_end")
        end_el.text = str(page_end)
        new_elements.append(end_el)

    for i, new_el in enumerate(new_elements):
        el.insert(i, new_el)
    new_elements[-1].tail = original_leading
    return True
