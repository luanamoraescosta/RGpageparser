#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single web panel for the RG page / block workflow.

Everything happens here: start the OCR, watch the live prediction, see which
pages need review, and decide (correct block / position / status) — all in the
browser.  Run it with:

    python review_app.py                     # defaults (vol 8)
    python rgpage.py review --vol 8          # same, via rgpage

Routes:
  GET  /                 panel: controls + live + review queue + table
  POST /run              start the OCR (+ match) in a background thread
  GET  /api/status       JSON job status (polled by the page)
  GET  /latest.jpg, /recent.jpg   live prediction frames
  GET  /page/<n>         review a single page (edit form)
  POST /save             save a manual review override
  GET  /export           write TSV + overrides
"""

from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_file, url_for
from markupsafe import Markup, escape

import rgpage
import rg_xml

STATUSES = ["ok", "low-score", "manual-review", "blank-page"]
POSITIONS = ["starts-here", "continues", "ends-here", "complete-on-page"]

# global state (guarded by LOCK)
LOCK = threading.Lock()
STATE = {"running": False, "done": False, "stage": "idle", "error": "",
         "progress": 0, "total": 0, "current": "", "rebuild": False,
         "easyocr": False, "pages_limit": "all"}
MATCHES: list[rgpage.PageMatch] = []
REVIEW: dict = {}
PAGE_REVIEW: dict[int, int] = {}

# loaded context
_DOCS: dict = {}
_DOCS_BY_ENTRY: dict = {}
_ENTRY_PAGES: dict = {}
_PAGES_BY_NUM: dict = {}
_PAGES_BY_SCAN: dict = {}


def _int(v) -> int | None:
    try:
        return int(v) if v else None
    except ValueError:
        return None


def load_context(cfg: dict) -> None:
    """Load docs + pages + entry_pages once into module globals."""
    global _DOCS, _DOCS_BY_ENTRY, _ENTRY_PAGES, _PAGES_BY_NUM, _PAGES_BY_SCAN
    load_page_review(cfg)
    try:
        docs = rgpage.load_docs(cfg["xml_path"])
    except (FileNotFoundError, OSError):
        docs = []
    _DOCS = {d.block_id: d for d in docs}
    _DOCS_BY_ENTRY = {}
    for d in docs:
        _DOCS_BY_ENTRY.setdefault(d.entry, []).append(d)
    pages = rgpage.load_cache(cfg["cache"])
    rgpage.normalize_pages(pages, cfg["parts"])
    _refresh_context(cfg, pages)


def load_review_data(cfg: dict) -> None:
    """(Re)load MATCHES from the pages TSV (call after a job finishes)."""
    global MATCHES
    with LOCK:
        matches = []
        if cfg["pages_tsv"].exists():
            with open(cfg["pages_tsv"], encoding="utf-8") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    page = _PAGES_BY_NUM.get(int(row["page"]))
                    if page is None:
                        continue
                    matches.append(rgpage.PageMatch(
                        page, _DOCS.get(row["last_block"]),
                        float(row["score"] or 0), row["position"], row["status"],
                        _int(row["span_start"]), _int(row["span_end"]),
                    ))
            spans = rgpage.build_spans(matches, list(_DOCS.values()),
                                       _ENTRY_PAGES)
            for m in matches:
                if m.block and m.block.block_id in spans:
                    s = spans[m.block.block_id]
                    m.span_start, m.span_end = s.page_start, s.page_end
        MATCHES = matches
        if cfg["review_json"].exists():
            REVIEW.clear()
            REVIEW.update(json.loads(cfg["review_json"].read_text(encoding="utf-8")))


def apply_overrides(m: rgpage.PageMatch) -> rgpage.PageMatch:
    o = REVIEW.get(str(m.page.page))
    if not o:
        return m
    if o.get("last_block") in _DOCS:
        m.block = _DOCS[o["last_block"]]
    m.position = o.get("position", m.position)
    m.status = o.get("status", m.status)
    m.score = float(o.get("score", m.score))
    return m


def load_page_review(cfg: dict) -> None:
    global PAGE_REVIEW
    PAGE_REVIEW = {}
    p = cfg.get("page_review_json")
    if p and Path(p).exists():
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        PAGE_REVIEW = {int(k): int(v) for k, v in data.items()}


def apply_page_overrides(pages: dict) -> None:
    if not PAGE_REVIEW:
        return
    for page in pages.values():
        if page.scan_no in PAGE_REVIEW:
            page.page = PAGE_REVIEW[page.scan_no]
            page.page_source = "manual"


def _refresh_context(cfg: dict, pages: dict) -> None:
    global _ENTRY_PAGES, _PAGES_BY_NUM, _PAGES_BY_SCAN
    apply_page_overrides(pages)
    _PAGES_BY_SCAN = {p.scan_no: p for p in pages.values()}
    _ENTRY_PAGES = rgpage.build_entry_pages(pages)
    _PAGES_BY_NUM = {p.page: p for p in pages.values() if p.page is not None}


def _unknown_block_msg(block_id: str) -> str:
    """Help find the right regest/head when the typed block_id does not exist."""
    entry = rg_xml.entry_number(block_id)
    msg = f"regest/head '{block_id}' not recognized in the XML."
    if entry is not None:
        docs = _DOCS_BY_ENTRY.get(entry)
        if docs:
            opts = ", ".join(
                f"{d.block_id}"
                for d in sorted(docs, key=lambda d: d.block_id))
            msg += f" Petentenvita {entry} has: {opts}."
    return msg


def last_block_desc(m: rgpage.PageMatch) -> str:
    """Description of the regest/head of the page's last lines + continuation."""
    if not m.block:
        return "no regest/head associated with the last lines."
    b = m.block
    base = f"petentenvita #{b.entry} · regest/head {b.block_id}"
    pos = m.position
    if pos in ("starts-here", "continues"):
        nxt = m.span_end if m.span_end and m.span_end > m.page.page else None
        if nxt:
            return f"{base} — continues on page {nxt}"
        return f"{base} — continues to the next page"
    return f"{base} — ends on this page"


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def _norm_int(v):
    try:
        return int(str(v))
    except (ValueError, TypeError):
        return None


def _match_page(m, f: str) -> bool:
    f = f.strip()
    if not f:
        return True
    if "-" in f:
        a, b = f.split("-", 1)
        lo, hi = _norm_int(a), _norm_int(b)
        if lo is None or hi is None:
            return True
        return lo <= m.page.page <= hi
    n = _norm_int(f)
    if n is None:
        return True
    return m.page.page == n


def _match_entry(m, f: str) -> bool:
    f = f.strip()
    if not f:
        return True
    q = f.split("-")[0]
    n = _norm_int(q)
    cands = set()
    cands.update(m.page.entries)
    if m.block:
        cands.add(m.block.entry)
    if n is None:
        return any(f.lower() in str(e).lower() for e in cands)
    for e in cands:
        if e == n:
            return True
        if len(str(n)) > len(str(e)) and str(n).endswith(str(e)):
            return True
    return False


def _match_block(m, f: str) -> bool:
    f = f.strip()
    if not f:
        return True
    bid = m.block.block_id if m.block else ""
    return f.lower() in bid.lower()


def match_filters(m, page_f, entry_f, block_f, status_f) -> bool:
    if not _match_page(m, page_f):
        return False
    if not _match_entry(m, entry_f):
        return False
    if not _match_block(m, block_f):
        return False
    if status_f and m.status != status_f:
        return False
    return True


# ---------------------------------------------------------------------------
# report: pages by petentenvita / head / rg
# ---------------------------------------------------------------------------

def build_report():
    with LOCK:
        ms = [apply_overrides(m) for m in MATCHES]
    if not ms:
        return []
    spans = rgpage.build_spans(ms, list(_DOCS.values()), _ENTRY_PAGES)
    reviewed = {int(k) for k in REVIEW}
    block_reviewed = {m.block.block_id for m in ms
                      if m.block and m.page.page in reviewed}
    rows = []
    for blk in sorted(_DOCS.values(), key=lambda d: (d.entry, d.block_id)):
        s = spans.get(blk.block_id)
        label = rgpage.kind_label(blk.block_id, blk.kind)
        page_start, page_end, span, score = None, None, "", ""
        if s:
            page_start, page_end = s.page_start, s.page_end
            span = (f"p. {page_start}" if page_start == page_end
                    else f"p. {page_start} → {page_end}")
            label = s.label
            score = f"{s.score:.0f}"
        pages = (str(page_start) if page_start == page_end and page_start
                 else f"{page_start}-{page_end}" if page_start and page_end else "")
        rows.append({"entry": blk.entry, "block_id": blk.block_id,
                     "label": label, "kind": blk.kind,
                     "pages": pages, "page_start": page_start, "page_end": page_end,
                     "span": span, "score": score,
                     "overridden": "yes" if blk.block_id in block_reviewed else ""})
    return rows


def group_report(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["entry"], []).append(r)
    out = []
    for e in sorted(groups):
        items = groups[e]
        head = next((i for i in items if i["kind"] == "head"), None)
        rgs = [i for i in items if i["kind"] == "regest"]
        pages = set()
        for i in items:
            if i["pages"]:
                for part in str(i["pages"]).split("-"):
                    if part.isdigit():
                        pages.add(int(part))
        out.append({
            "entry": e,
            "head": f"{head['block_id']} ({head['pages']})"
                    if head and head["pages"] else (head["block_id"] if head else "—"),
            "rgs": [f"{i['label']} {i['pages']}" for i in rgs if i["pages"]],
            "pages": ",".join(sorted(map(str, pages))) if pages else "—",
        })
    return out


def _rematch(cfg: dict) -> None:
    """Re-run matching with corrected page numbers (no OCR). Updates MATCHES."""
    global MATCHES
    with LOCK:
        STATE["running"] = True
        STATE["done"] = False
        STATE["error"] = ""
        STATE["stage"] = "matching"
        STATE["current"] = "re-matching…"
        STATE["progress"] = 0
        STATE["total"] = 0
    try:
        pages = rgpage.load_cache(cfg["cache"])
        rgpage.normalize_pages(pages, cfg["parts"])
        apply_page_overrides(pages)
        entry_pages = rgpage.build_entry_pages(pages)
        matches = rgpage.match_pages(pages, entry_pages, _DOCS_BY_ENTRY,
                                     cfg["parts"], False)
        spans = rgpage.build_spans(matches, list(_DOCS.values()), entry_pages)
        rgpage.write_pages_tsv(matches, cfg["pages_tsv"])
        rgpage.write_blocks_tsv(spans, cfg["blocks_tsv"])
        with LOCK:
            MATCHES = matches
            _refresh_context(cfg, pages)
            STATE["running"] = False
            STATE["done"] = True
            STATE["stage"] = "done"
            STATE["current"] = "done"
    except Exception as exc:
        with LOCK:
            STATE["running"] = False
            STATE["done"] = True
            STATE["stage"] = "done"
            STATE["error"] = str(exc)


def _run_job(cfg: dict) -> None:
    with LOCK:
        STATE["running"] = True
        STATE["done"] = False
        STATE["error"] = ""
        STATE["stage"] = "OCR"
        STATE["current"] = "starting…"
        STATE["progress"] = 0
        MATCHES.clear()
    try:
        live = rgpage.LiveView(cfg["live_dir"], False, 250)

        def cb(i, total, scan_no, page):
            with LOCK:
                STATE["progress"] = i
                STATE["total"] = total
                STATE["current"] = f"scan {scan_no:04d} · page {page}"

        with LOCK:
            rebuild = STATE["rebuild"]
            easyocr = STATE["easyocr"]
            limit = None
            pl = STATE.get("pages_limit")
            if pl not in (None, "all"):
                try:
                    limit = int(pl)
                except ValueError:
                    limit = None
        pages = rgpage.scan_pages(
            cfg["parts"], cfg["cache"], set(_DOCS_BY_ENTRY),
            live, rebuild, None, None, limit, progress_cb=cb)

        with LOCK:
            STATE["stage"] = "matching"
            STATE["current"] = "generating tables…"
        rgpage.normalize_pages(pages, cfg["parts"])
        entry_pages = rgpage.build_entry_pages(pages)
        matches = rgpage.match_pages(pages, entry_pages, _DOCS_BY_ENTRY,
                                     cfg["parts"], easyocr)
        spans = rgpage.build_spans(matches, list(_DOCS.values()), entry_pages)
        rgpage.write_pages_tsv(matches, cfg["pages_tsv"])
        rgpage.write_blocks_tsv(spans, cfg["blocks_tsv"])
        with LOCK:
            STATE["current"] = "annotating results…"
        for mm in matches:
            if mm.page.blank or not mm.page.page or not mm.block:
                continue
            path = rgpage.find_scan_path(cfg["parts"], mm.page.scan)
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                live.update(rgpage.annotate(img, mm.page, mm),
                            f"page {mm.page.page} · {mm.block.block_id}")
        live.close()
        with LOCK:
            STATE["stage"] = "done"
            STATE["done"] = True
            STATE["progress"] = STATE["total"] or STATE["progress"]
            STATE["current"] = "done"
        _refresh_context(cfg, pages)
        load_review_data(cfg)
    except Exception as exc:  # noqa: BLE001
        with LOCK:
            STATE["stage"] = "error"
            STATE["error"] = str(exc)
            STATE["done"] = False
    finally:
        with LOCK:
            STATE["running"] = False


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

INDEX_TMPL = """<!doctype html><html><head><meta charset="utf-8">
<title>RG{{ vol }} · painel</title>
<style>
 body{font-family:-apple-system,Arial,sans-serif;margin:22px;color:#222}
 a{color:#0a58ca;text-decoration:none}
 h2{margin-top:0}
 .card{background:#f6f7f9;border:1px solid #ddd;border-radius:8px;padding:14px;margin-bottom:16px}
 .live img{max-height:70vh;border:1px solid #ccc;background:#000}
 .live{display:flex;gap:14px;align-items:flex-start}
 .side{min-width:280px}
 .bar{height:14px;background:#e4e4e4;border-radius:7px;overflow:hidden;margin:6px 0}
 .bar>div{height:100%;background:#0a58ca;width:0}
 .filters a{margin-right:8px;padding:3px 8px;border-radius:4px;background:#eee}
 .filters a.on{background:#0a58ca;color:#fff}
 .filters input,.filters select{padding:4px 6px;border:1px solid #ccc;border-radius:4px;margin-right:6px}
 .filters button{padding:4px 10px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{padding:4px 7px;border-bottom:1px solid #ddd;text-align:left}
 th{background:#f0f0f0}
 .ok{color:#137a3a}.low{color:#b02a1c}.blank{color:#777}.man{color:#7a3bb0}
 .err{color:#b02a1c}
.tabnav{margin-bottom:14px;border-bottom:2px solid #ddd}
 .tabnav a{display:inline-block;padding:8px 16px;margin-right:4px;border-radius:6px 6px 0 0;background:#eee}
 .tabnav a.on{background:#0a58ca;color:#fff}
 .dl a{display:inline-block;padding:8px 14px;margin:4px 8px 4px 0;background:#0a58ca;color:#fff;border-radius:6px}
</style></head><body>
<h2>Repertorium Germanicum RG{{ vol }} — panel</h2>

{% if volumes %}
<div class="card" style="margin-bottom:12px">
 <label>Volume:
  <select onchange="if(this.value) location.href='/switch/'+this.value">
   {% for v in volumes|sort %}
    <option value="{{ v }}" {{ 'selected' if v|int == vol }}>RG{{ v }}</option>
   {% endfor %}
  </select>
 </label>
 <span style="color:#666">switch volume to review / run a different Repertorium Germanicum volume</span>
</div>
{% endif %}

<div class="tabnav">
 <a data-tab="ocr" href="?tab=ocr">▶ Run OCR</a>
 <a href="{{ url_for('page_review') }}">📄 Page review</a>
 <a data-tab="review" href="?tab=review">✎ Review</a>
 <a data-tab="export" href="?tab=export">⬇ Export</a>
</div>

<div id="tab-ocr" class="tab">
<div class="card">
  <form method="post" action="{{ url_for('run') }}">
   <label><input type="checkbox" name="rebuild"> OCR from scratch (ignore cache)</label>
  &nbsp; <label>pages
    <select name="pages">
     <option value="all">all pages</option>
     <option value="20">first 20 pages</option>
    </select>
  </label>
   &nbsp; <button class="btn" type="submit" {{ 'disabled' if running }}>▶ Run</button>
  </form>
 <div id="status">
  <b>Status:</b> <span id="stage">{{ stage }}</span><span id="cur"> {{ current }}</span>
  <div class="bar"><div id="pbar"></div></div>
  <div id="err" class="err"></div>
 </div>
</div>

<div class="card live">
 <img id="latest" src="{{ url_for('latest') }}" alt="current page">
 <div class="side">
  <h4>Live prediction</h4>
  <p>Images auto-refresh every ~1s while OCR is running.</p>
  <p><a href="{{ url_for('live') }}">open fullscreen →</a></p>
  <img id="recent" src="{{ url_for('recent') }}" style="width:100%" alt="recent pages">
 </div>
</div>

{% if not matches %}
<div class="card"><p>No results yet. Click <b>▶ Run</b> to run OCR and generate the tables.</p></div>
{% endif %}
</div>

<div id="tab-review" class="tab" style="display:none">
{% if matches %}
<div class="card">
 <h3>Review first ({{ queue|length }})</h3>
 {% if queue %}
 <table>
   <tr><th>page</th><th>scan</th><th>status</th><th>score</th><th>last regest/head</th><th>position</th></tr>
   {% for m in queue %}
   <tr>
    <td><a href="{{ url_for('page', n=m.page.page) }}">{{ m.page.page }}</a></td>
    <td>{{ m.page.scan_no }}</td>
    <td class="{{ 'low' if m.status=='low-score' else 'man' if m.status=='manual-review' else '' }}">{{ m.status }}</td>
    <td>{{ '%.1f'|format(m.score) }}</td>
    <td>{{ m.block.block_id if m.block else '—' }}</td>
    <td>{{ m.position | pos }}</td>
   </tr>
  {% endfor %}
 </table>
 {% else %}<p>Nothing to review.</p>{% endif %}
</div>

<div class="card">
 <h3>All pages <small>({{ shown }} of {{ matches|length }})</small></h3>
 <form class="filters" method="get" action="{{ url_for('index') }}">
  <input type="text" name="page" value="{{ page_f }}" placeholder="page (182 or 180-185)">
  <input type="text" name="entry" value="{{ entry_f }}" placeholder="petentenvita (1243 or 10801243)">
  <input type="text" name="block" value="{{ block_f }}" placeholder="regest/head (10801243-0)">
  <select name="status">
   <option value="">all statuses</option>
   {% for s in ['ok','low-score','blank-page','manual-review'] %}
   <option value="{{ s }}" {{ 'selected' if status_f==s }}>{{ s }} ({{ counts[s] }})</option>
   {% endfor %}
  </select>
  <button type="submit">Filter</button>
  <a href="{{ url_for('index') }}">clear</a>
 </form>
 <div class="filters">
  {% for s,label in [('','all'),('ok','ok'),('low-score','low confidence'),('blank-page','blank'),('manual-review','reviewed')] %}
   <a class="{{ 'on' if status_f==s else '' }}" href="{{ url_for('index', status=s, page=page_f, entry=entry_f, block=block_f) }}">{{ label }} ({{ counts[s] }})</a>
  {% endfor %}
 </div>
 <table>
   <tr><th>page</th><th>scan</th><th>status</th><th>score</th><th>regest/head</th><th>position</th><th>entries</th></tr>
   {% for m in rows %}
   <tr>
    <td><a href="{{ url_for('page', n=m.page.page) }}">{{ m.page.page }}</a></td>
    <td>{{ m.page.scan_no }}</td>
    <td class="{{ 'ok' if m.status=='ok' else 'low' if m.status=='low-score' else 'blank' if m.status=='blank-page' else 'man' }}">{{ m.status }}</td>
    <td>{{ '%.1f'|format(m.score) if m.status!='blank-page' else '—' }}</td>
    <td>{{ m.block.block_id if m.block else '—' }}</td>
    <td>{% if m.status!='blank-page' %}{{ m.position | pos }}{% else %}—{% endif %}</td>
    <td>{{ m.page.entries | join(', ') if m.page.entries else '—' }}</td>
   </tr>
  {% endfor %}
 </table>
  {% if not rows %}<p>No pages match the filter.</p>{% endif %}
</div>
{% endif %}
</div>

<div id="tab-export" class="tab" style="display:none">
 <div class="card">
   <h3>Report: regests and their pages</h3>
   <p>{{ entries_n }} petentenvitas · {{ report|length }} blocks (regest/head) in XML ·
      {{ report|length - missing|length }} with pages ·
      <b>{{ missing|length }} missing</b></p>
  <div class="dl">
   <a href="{{ url_for('download_report') }}">download report: all regests + pages (.tsv)</a>
  </div>
 </div>
  <div class="card">
   <h3>Regests and their pages</h3>
   <p>{{ regests|length }} / {{ regest_total }} regests · {{ regest_total - missing|length }} with a page{% if regest_total > 100 and not regest_entry_f and not regest_block_f %} · showing first 100, use the filter{% endif %}</p>
   <form method="get" action="{{ url_for('index') }}">
    <input type="hidden" name="tab" value="export">
    <input type="text" name="regest_entry" value="{{ regest_entry_f }}" placeholder="filter by petentenvita (e.g. 1259)">
    <input type="text" name="regest_block" value="{{ regest_block_f }}" placeholder="filter by regest (e.g. -2 or 10800004-2)">
    <button type="submit">filter</button>
    <a href="{{ url_for('index', tab='export') }}">clear</a>
   </form>
   <table id="subtable">
    <tr><th>last regest/head</th><th>petentenvita</th><th>kind</th><th>pages</th><th>conf</th><th>overridden</th></tr>
    {% for s in regests %}
    <tr>
     <td>{{ s.block_id }}</td>
     <td>{{ s.entry }}</td>
     <td>{{ s.kind }}</td>
     <td>{{ s.span or '—' }}</td>
     <td>{{ s.score or '—' }}</td>
     <td>{{ s.overridden or '' }}</td>
    </tr>
    {% endfor %}
   </table>
   <p class="note">Regest = the -1, -2, ... regests in the XML (these are what matter). Page = where the regest falls (start-end). Filter by petentenvita to see all its regests; filter by regest to locate one.</p>
  </div>
 {% if missing %}
 <div class="card">
  <h3>Blocks without pages ({{ missing|length }})</h3>
  <table>
   <tr><th>petentenvita</th><th>regest/head</th><th>type</th></tr>
   {% for r in missing[:100] %}
   <tr>
    <td>{{ r.entry }}</td>
    <td>{{ r.block_id }}</td>
    <td>{{ r.label }}</td>
   </tr>
   {% endfor %}
  </table>
  <p class="note">Showing the first 100. These blocks exist in the XML but have no associated pages (outside the scanned range or no match). The full list is in the .tsv report.</p>
 </div>
 {% endif %}
</div>

<script>
var params = new URLSearchParams(location.search);
var cur = params.get('tab') || sessionStorage.getItem('tab') || 'ocr';
function showTab(name){
  document.querySelectorAll('.tab').forEach(function(e){e.style.display='none';});
  var el = document.getElementById('tab-'+name);
  if(el){ el.style.display='block'; }
  document.querySelectorAll('.tabnav a').forEach(function(a){a.classList.toggle('on', a.dataset.tab===name);});
}
document.querySelectorAll('.tabnav a[data-tab]').forEach(function(a){
  a.addEventListener('click', function(ev){ ev.preventDefault(); showTab(a.dataset.tab); history.replaceState(null,'','?tab='+a.dataset.tab); sessionStorage.setItem('tab', a.dataset.tab); });
});
showTab(cur);

function tick(){
  fetch('{{ url_for('api_status') }}').then(r=>r.json()).then(s=>{
    document.getElementById('stage').textContent = s.stage;
    document.getElementById('cur').textContent = s.current || '';
    var pct = s.total ? Math.round(100*s.progress/s.total) : 0;
    document.getElementById('pbar').style.width = pct+'%';
    document.getElementById('err').textContent = s.error || '';
    var t = Date.now();
    document.getElementById('latest').src = '{{ url_for('latest') }}?t='+t;
    document.getElementById('recent').src = '{{ url_for('recent') }}?t='+t;
    if(s.stage==='done'){
      if(!sessionStorage.getItem('doneR')){ sessionStorage.setItem('doneR','1'); location.reload(); }
    } else { sessionStorage.removeItem('doneR'); }
  });
}
setInterval(tick, 1000); tick();
</script>
</body></html>"""

LIVE_TMPL = """<!doctype html><html><head><meta charset="utf-8">
<title>RG{{ vol }} · ao vivo</title>
<style>
 body{margin:0;background:#111;color:#ddd;font-family:-apple-system,Arial,sans-serif}
 .top{display:flex;align-items:center;gap:16px;padding:10px 16px;background:#1a1a1a;position:sticky;top:0}
 .top a{color:#7ab8ff;text-decoration:none}
 .bar{flex:1;height:12px;background:#333;border-radius:6px;overflow:hidden}
 .bar>div{height:100%;background:#0a58ca;width:0}
 img{display:block;width:100%;max-height:calc(100vh - 60px);object-fit:contain;background:#000}
 .sp{color:#ffd479}
</style></head><body>
<div class="top">
 <a href="{{ url_for('index') }}">&larr; panel</a>
 <span><b>RG{{ vol }}</b> · live</span>
 <div class="bar"><div id="pbar"></div></div>
 <span id="stat" class="sp">…</span>
</div>
<img id="live" src="{{ url_for('latest') }}" alt="current page">
<script>
function tick(){
  fetch('{{ url_for('api_status') }}').then(r=>r.json()).then(s=>{
    var pct = s.total ? Math.round(100*s.progress/s.total) : 0;
    document.getElementById('pbar').style.width = pct+'%';
    document.getElementById('stat').textContent = s.stage + (s.current? ' · '+s.current : '');
    document.getElementById('live').src = '{{ url_for('latest') }}?t='+Date.now();
  });
}
setInterval(tick, 800); tick();
</script>
</body></html>"""

PAGES_TMPL = """<!doctype html><html><head><meta charset="utf-8">
<title>RG{{ vol }} · page review</title>
<style>
 body{font-family:-apple-system,Arial,sans-serif;margin:22px;color:#222}
 a{color:#0a58ca;text-decoration:none}
 h2{margin-top:0}
 .hint{color:#555;font-size:13px}
 .tabs{margin:14px 0}
 .tabs a{padding:6px 12px;border:1px solid #ccc;border-radius:6px;margin-right:6px;background:#fff;font-size:14px}
 .tabs a.cur{background:#0a58ca;color:#fff;border-color:#0a58ca}
 table{border-collapse:collapse;width:100%;font-size:14px}
 th,td{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:middle}
 td input{width:80px;padding:3px 5px;font-size:14px}
 .badge{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px}
 .badge.inf{background:#fff3cd;color:#8a6d1a}
 .badge.none{background:#f8d7da;color:#842029}
 .badge.tess{background:#e3e6ea;color:#444}
 .fixed{position:sticky;bottom:0;background:#f6f7f9;border:1px solid #ddd;padding:12px;border-radius:8px}
 button{padding:8px 16px}
 .done{color:#137a3a;font-weight:600}
 .empty{color:#777;font-style:italic}
</style></head><body>
<p><a href="{{ url_for('index') }}">&larr; back to panel</a></p>
<h2>Page review <small>RG{{ vol }}</small></h2>
<p class="hint">Set the correct folio (page) number for any scan, then click <b>Save &amp; re-match</b> to re-locate the regests.
Pages whose number was <b>inferred</b> / <b>not detected</b> are highlighted; low OCR confidence shows in the <b>Conf</b> column.</p>
<p class="tabs">
 <a href="{{ url_for('page_review', filter='all') }}" class="{{ 'cur' if flt=='all' }}">All ({{ total }})</a>
 <a href="{{ url_for('page_review', filter='inferred') }}" class="{{ 'cur' if flt=='inferred' }}">Inferred / none ({{ n_inferred }})</a>
 <a href="{{ url_for('page_review', filter='low') }}" class="{{ 'cur' if flt=='low' }}">Low conf ({{ n_low }})</a>
</p>
{% if saved %}<p class="done">Saved ✓ — re-matching.</p>{% endif %}
{% if not pages %}
 <p class="empty">No pages in this filter.</p>
{% else %}
<form method="post" action="{{ url_for('pages_save') }}">
<table>
 <tr><th>Scan</th><th>Page</th><th>Source</th><th>Conf</th><th>View</th></tr>
 {% for p in pages %}
 <tr>
  <td>{{ p.scan_no }}</td>
  <td><input type="number" name="page_{{ p.scan_no }}" min="1" value="{{ p.page }}"></td>
  <td><span class="badge {{ 'inf' if p.page_source=='inferred' else ('none' if p.page_source=='none' else 'tess') }}">{{ p.page_source }}</span></td>
  <td>{{ '%.0f'|format(p.mean_conf) if p.mean_conf else '—' }}</td>
  <td><a href="{{ url_for('scan', n=p.scan_no) }}" target="_blank">view</a></td>
 </tr>
 {% endfor %}
</table>
 <div class="fixed"><button type="submit">Save &amp; re-match</button></div>
</form>
{% endif %}
</body></html>"""

PAGE_TMPL = """<!doctype html><html><head><meta charset="utf-8">
<title>RG{{ vol }} · page {{ m.page.page }}</title>
<style>
 body{font-family:-apple-system,Arial,sans-serif;margin:20px;color:#222}
 .wrap{display:flex;gap:24px;align-items:flex-start}
 img{max-height:92vh;border:1px solid #ccc;background:#fff}
 form{min-width:330px;font-size:14px}
 label{display:block;margin-top:10px;font-weight:600}
 select,input[type=text],input[type=number],textarea{width:100%;box-sizing:border-box;margin-top:2px}
 .btns{margin-top:16px}
 button{padding:7px 14px}
 .note{font-size:12px;color:#555;margin-top:4px}
 .toprow{display:flex;gap:16px;margin:8px 0;align-items:flex-start}
 .pane{flex:1;min-width:0;background:#fff;border:1px solid #d5e3f7;border-radius:6px;overflow:hidden}
 .panehead{background:#eaf2ff;border-bottom:1px solid #d5e3f7;padding:5px 10px;font-size:12px;font-weight:600;color:#0a58ca}
 .panebody{max-height:180px;overflow:auto;padding:8px 10px;font-size:12px;white-space:pre-wrap;color:#333}
 .panebody mark{background:#ffe08a;text-decoration:underline;font-weight:600}
 .cands{font-size:12px;margin-top:6px}
 .cands b{color:#0a58ca}
 .candgrp{margin:4px 0;padding:4px 6px;border-radius:4px;background:#f5f8ff}
 .candgrp.cur{background:#fff7e0;border:1px solid #f0d48a}
 .candgrp b.cur{color:#c0392b}
 .candhead{font-size:12px;font-weight:600;margin-bottom:2px}
 .candbody{max-height:150px;overflow:auto;background:#fff;border:1px solid #e0e6ef;border-radius:4px;padding:4px 6px;font-size:12px;white-space:pre-wrap;color:#333;margin-top:3px}
 .candbody mark{background:#ffe08a;text-decoration:underline}
 .okmsg{background:#d9f2df;border:1px solid #5fbf78;padding:8px 12px;border-radius:6px;color:#137a3a;font-weight:600}
 .nav{display:flex;gap:10px;margin:8px 0}
</style></head><body>
<p><a href="{{ url_for('index') }}">&larr; back to panel</a></p>
{% if saved %}<div class="okmsg">Saved ✓ — correction applied.</div>{% endif %}
<div class="nav">
 {% if prev_p %}<a href="{{ url_for('page', n=prev_p) }}">&larr; page {{ prev_p }}</a>{% endif %}
 {% if next_p %}<a href="{{ url_for('page', n=next_p) }}">page {{ next_p }} &rarr;</a>{% endif %}
</div>
<h3>Page {{ m.page.page }} <small>scan {{ m.page.scan_no }}</small></h3>
<div class="toprow">
  <div class="pane">
    <div class="panehead">OCR text (page)</div>
    <div class="panebody">{{ ocr }}</div>
  </div>
  <div class="pane">
    <div class="panehead">Best match: <b>{{ m.block.block_id if m.block else '—' }}</b> · {{ m.block.kind if m.block else '—' }} · petentenvita #{{ m.block.entry if m.block else '—' }} · score {{ '%.0f'|format(matched_score) if matched_score is not none else '—' }}</div>
    {% if block_html %}<div class="panebody">{{ block_html }}</div>{% else %}<div class="panebody">no regest/head matched for this page.</div>{% endif %}
    <div class="note" style="padding:4px 10px">Last lines → {{ last_desc }}</div>
  </div>
</div>
<div class="wrap">
 <img src="{{ url_for('image', n=m.page.page) }}" alt="annotated page">
 <form method="post" action="{{ url_for('save') }}">
  <input type="hidden" name="page" value="{{ m.page.page }}">
  <label>status</label>
  <select name="status">
   {% for s in STATUSES %}<option value="{{ s }}" {{ 'selected' if s==m.status }}>{{ s }}</option>{% endfor %}
  </select>
    <label>last regest/head (block_id)</label>
    <input type="text" name="last_block" list="blks" value="{{ m.block.block_id if m.block else '' }}">
   <label>search petentenvita no.</label>
   <input type="text" id="petSearch" placeholder="e.g. 1259" autocomplete="off">
   <div id="petRes" class="cands"></div>
    <datalist id="blks">
    {% for bid, kind, entry, sc in blk_opts %}<option value="{{ bid }}">{{ kind }} · {{ bid }} · #{{ entry }} · score {{ '%.0f'|format(sc) }}</option>{% endfor %}
   </datalist>
   <div class="cands">Candidates (top {{ cands|length }}):
    {% for c in cands %}<div class="candgrp {{ 'cur' if c.current else '' }}"><div class="candhead">{% if c.current %}<b class="cur">#{{ c.entry }} (current)</b>{% else %}<b>#{{ c.entry }}</b>{% endif %} · {{ c.kind }} · <b>{{ c.block_id }}</b> · score {{ '%.0f'|format(c.score) }}</div><div class="candbody">{{ c.text }}</div></div>{% endfor %}
   </div>
   <label>position</label>
    <select name="position">
     {% set sel = 'continues' if m.position in ('starts-here','continues') else 'ends-here' %}
     <option value="continues" {{ 'selected' if sel=='continues' }}>continues</option>
     <option value="ends-here" {{ 'selected' if sel=='ends-here' }}>ends here</option>
    </select>
  <label>score (0–100, optional)</label>
  <input type="number" name="score" min="0" max="100" step="0.1" placeholder="auto {{ '%.0f'|format(m.score) if m.status!='blank-page' else '—' }}">
  <div class="btns"><button type="submit">save review</button></div>
 </form>
 </div>
<script>
const petSearch = document.getElementById('petSearch');
const petRes = document.getElementById('petRes');
const lastBlock = document.querySelector('input[name="last_block"]');
function renderRes(blocks, entry){
  if(!blocks.length){ petRes.innerHTML = '<i>no. ' + entry + ' not found</i>'; return; }
  petRes.innerHTML = blocks.map(b =>
    '<div class="candgrp"><b>#' + b[0] + '</b>: <a href="javascript:void(0)" onclick="setBlk(\'' + b[1] + '\')">' + b[2] + ' · ' + b[1] + '</a></div>'
  ).join('');
}
window.setBlk = function(bid){ lastBlock.value = bid; petSearch.value = ''; petRes.innerHTML = ''; };
petSearch.addEventListener('input', function(){
  const v = this.value.replace(/\D/g, '');
  if(!v){ petRes.innerHTML = ''; return; }
  const entry = parseInt(v, 10);
  fetch('{{ url_for('api_blocks') }}?q=' + entry)
    .then(r => r.json())
    .then(blocks => renderRes(blocks, entry))
    .catch(function(){ petRes.innerHTML = '<i>search error</i>'; });
});
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------

def _make_volume_cfg(vol: int) -> dict:
    vc = rgpage.resolve_volume_config(vol)
    out_dir = vc["output_dir"]
    dbg = vc["debug_dir"] / f"rg{vol}"
    return {
        "vol": vol,
        "xml_path": vc["xml"],
        "cache": rgpage.cache_path(dbg),
        "pages_tsv": out_dir / f"rg{vol}_pages.tsv",
        "blocks_tsv": out_dir / f"rg{vol}_blocks.tsv",
        "review_json": out_dir / f"rg{vol}_review.json",
        "page_review_json": out_dir / f"rg{vol}_page_review.json",
        "live_dir": out_dir / f"rg{vol}_live",
        "images_dir": vc["images"],
        "first_scan": vc["first_scan_no"],
        "first_page": vc["first_logical_page"],
        "parts": vc["parts"],
    }


def build_app(cfg: dict, volumes: dict | None = None) -> Flask:
    volumes = volumes or {}
    vol_map = volumes.get("volumes", {})
    load_context(cfg)
    if cfg["pages_tsv"].exists():
        load_review_data(cfg)

    def _set_volume(vol: int) -> None:
        new_cfg = _make_volume_cfg(vol)
        cfg.clear()
        cfg.update(new_cfg)
        with LOCK:
            STATE.update({"running": False, "done": False, "stage": "idle",
                          "error": "", "progress": 0, "total": 0, "current": ""})
            MATCHES.clear()
            REVIEW.clear()
            PAGE_REVIEW.clear()
        load_context(cfg)
        if cfg["pages_tsv"].exists():
            load_review_data(cfg)

    def _serve_file(path: Path, placeholder: str) -> Response:
        if path.exists():
            return Response(path.read_bytes(), mimetype="image/jpeg")
        img = np.full((600, 400, 3), 200, np.uint8)
        cv2.putText(img, placeholder, (60, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (60, 60, 60), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img)
        return Response(buf.tobytes(), mimetype="image/jpeg")

    app = Flask(__name__)

    def _pos_ui(p: str) -> str:
        if p in ("starts-here", "continues"):
            return "continues to next page"
        if p in ("ends-here", "complete-on-page"):
            return "ends here"
        return p or "—"

    app.jinja_env.filters["pos"] = _pos_ui

    @app.route("/", methods=["GET"])
    def index():
        page_f = request.args.get("page", "")
        entry_f = request.args.get("entry", "")
        block_f = request.args.get("block", "")
        status_f = request.args.get("status", "")
        regest_entry_f = request.args.get("regest_entry", "")
        regest_block_f = request.args.get("regest_block", "")
        with LOCK:
            running = STATE["running"]
            stage = STATE["stage"]
            current = STATE["current"]
            error = STATE["error"]
            ms = [apply_overrides(m) for m in MATCHES]
        counts = {s: sum(1 for m in ms if m.status == s)
                  for s in ["", "ok", "low-score", "blank-page", "manual-review"]}
        counts[""] = len(ms)
        queue = [m for m in ms if m.status == "low-score"]
        queue.sort(key=lambda m: m.score)
        rows = [m for m in ms
                if match_filters(m, page_f, entry_f, block_f, status_f)]
        report = build_report() if ms else []
        report_groups = group_report(report)
        missing = [r for r in report if not r["pages"]]
        regests = [r for r in report if r["kind"] == "regest"]
        regest_total = len(regests)
        if regest_entry_f:
            if regest_entry_f.isdigit():
                regests = [r for r in regests if str(r["entry"]) == regest_entry_f]
            else:
                regests = [r for r in regests if regest_entry_f in r["block_id"]]
        if regest_block_f:
            regests = [r for r in regests if regest_block_f in r["block_id"]]
        else:
            regests = regests[:100]
        return render_template_string(
            INDEX_TMPL, vol=cfg["vol"], volumes=vol_map, running=running, stage=stage, current=current,
            error=error, matches=ms, queue=queue, rows=rows,
            counts=counts, page_f=page_f, entry_f=entry_f, block_f=block_f,
            status_f=status_f, shown=len(rows), report=report,
            report_groups=report_groups, entries_n=len(report_groups),
            missing=missing, regests=regests,
            regest_total=regest_total, regest_entry_f=regest_entry_f, regest_block_f=regest_block_f,
        )

    @app.route("/switch/<int:vol>")
    def switch_volume(vol):
        if str(vol) not in vol_map:
            return redirect(url_for("index"))
        _set_volume(vol)
        return redirect(url_for("index"))

    @app.route("/run", methods=["POST"])
    def run():
        with LOCK:
            if STATE["running"]:
                return redirect(url_for("index"))
            STATE["rebuild"] = bool(request.form.get("rebuild"))
            STATE["easyocr"] = False
            STATE["pages_limit"] = request.form.get("pages") or "all"
        threading.Thread(target=_run_job, args=(cfg,), daemon=True).start()
        return redirect(url_for("index"))

    @app.route("/api/status")
    def api_status():
        with LOCK:
            return jsonify({k: STATE[k] for k in
                            ("running", "done", "stage", "error", "progress",
                             "total", "current")})

    @app.route("/api/blocks")
    def api_blocks():
        q = request.args.get("q", "").strip()
        try:
            entry = int(q)
        except ValueError:
            return jsonify([])
        blocks = [[entry, d.block_id, d.kind]
                  for d in _DOCS_BY_ENTRY.get(entry, [])]
        return jsonify(blocks)

    @app.route("/latest.jpg")
    def latest():
        return _serve_file(cfg["live_dir"] / "latest.jpg", "waiting for OCR...")

    @app.route("/recent.jpg")
    def recent():
        return _serve_file(cfg["live_dir"] / "recent.jpg", "no frames")

    @app.route("/live")
    def live():
        return render_template_string(LIVE_TMPL, vol=cfg["vol"])

    def find_page(n):
        with LOCK:
            return next((apply_overrides(x) for x in MATCHES if x.page.page == n), None)

    @app.route("/page/<int:n>")
    def page(n):
        m = find_page(n)
        if m is None:
            return "page not found (run OCR first)", 404
        cand_entries = rgpage.candidate_entries_for_page(m.page.page, _ENTRY_PAGES,
                                                         page_entries=m.page.entries)
        docs_near = [d for e in cand_entries for d in _DOCS_BY_ENTRY.get(e, [])]
        text = " ".join(ln.text for ln in m.page.lines).strip()
        scored = {d.block_id: rgpage.fuzzy_score(text, d)[0] for d in docs_near}
        # make sure the matched block's entry appears even if outside the window
        guaranteed = {m.block.entry} if m.block else set()
        have = {d.block_id for d in docs_near}
        for e in guaranteed:
            for d in _DOCS_BY_ENTRY.get(e, []):
                if d.block_id not in have:
                    docs_near.append(d)
                    scored[d.block_id] = rgpage.fuzzy_score(text, d)[0]
                    have.add(d.block_id)
        # top candidates (blocks) by score: the matched block first, then best alts
        cands = []
        seen = set()
        if m.block:
            cands.append({"block_id": m.block.block_id, "kind": m.block.kind,
                          "entry": m.block.entry, "score": m.score,
                          "text": m.block.text, "current": True})
            seen.add(m.block.block_id)
        for d in sorted(docs_near, key=lambda d: scored[d.block_id], reverse=True):
            if d.block_id in seen:
                continue
            cands.append({"block_id": d.block_id, "kind": d.kind,
                          "entry": d.entry, "score": scored[d.block_id],
                          "text": d.text, "current": False})
            seen.add(d.block_id)
            if len(cands) >= 5:
                break
        blk_opts = sorted(
            ((d.block_id, d.kind, d.entry, scored[d.block_id]) for d in docs_near),
            key=lambda t: (-t[3], t[0]))
        bt = m.block.text if m.block else ""
        matched_score = m.score if m.status != "blank-page" else None
        # highlight the aligned portion of the matched text
        block_html = None
        if m.block:
            hl = rgpage.match_highlight(text, m.block)
            if hl:
                s, e = hl
                block_html = Markup("".join([escape(bt[:s]), "<mark>",
                                             escape(bt[s:e]), "</mark>",
                                             escape(bt[e:])]))
            else:
                block_html = Markup(escape(bt))
        with LOCK:
            saved = request.args.get("saved") == "1"
            all_pages = sorted(x.page.page for x in MATCHES)
        idx = all_pages.index(n) if n in all_pages else -1
        prev_p = all_pages[idx - 1] if idx > 0 else None
        next_p = all_pages[idx + 1] if 0 <= idx < len(all_pages) - 1 else None
        return render_template_string(
            PAGE_TMPL, vol=cfg["vol"], m=m, cands=cands, blk_opts=blk_opts,
            STATUSES=STATUSES, POSITIONS=POSITIONS, ocr=text,
            block_html=block_html, last_desc=last_block_desc(m),
            matched_score=matched_score, saved=saved,
            prev_p=prev_p, next_p=next_p,
        )

    @app.route("/image/<int:n>.png")
    def image(n):
        m = find_page(n)
        if m is None:
            return "not found", 404
        path = rgpage.find_scan_path(cfg["parts"], m.page.scan)
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "image not found", 404
        ok, buf = cv2.imencode(".png", rgpage.annotate(img, m.page, m))
        if not ok:
            return "encode error", 500
        return Response(buf.tobytes(), mimetype="image/png")

    @app.route("/scan/<int:n>.png")
    def scan(n):
        page = _PAGES_BY_SCAN.get(n)
        if page is None:
            return "scan not found", 404
        path = rgpage.find_scan_path(cfg["parts"], page.scan)
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "image not found", 404
        ok, buf = cv2.imencode(".png", rgpage.annotate(img, page))
        if not ok:
            return "encode error", 500
        return Response(buf.tobytes(), mimetype="image/png")

    @app.route("/pages")
    def page_review():
        pages = sorted(_PAGES_BY_SCAN.values(), key=lambda p: p.scan_no)
        def uncertain(p):
            return p.page_source != "tesseract"
        def low_conf(p):
            return p.mean_conf < 65
        flt = request.args.get("filter", "all")
        n_inferred = sum(1 for p in pages if uncertain(p))
        n_low = sum(1 for p in pages if low_conf(p))
        if flt == "inferred":
            shown = [p for p in pages if uncertain(p)]
        elif flt == "low":
            shown = [p for p in pages if low_conf(p)]
        else:
            shown = pages
        return render_template_string(PAGES_TMPL, vol=cfg["vol"], pages=shown,
                                      flt=flt, total=len(pages),
                                      n_inferred=n_inferred, n_low=n_low,
                                      saved=request.args.get("saved"))

    @app.route("/pages/save", methods=["POST"])
    def pages_save():
        global PAGE_REVIEW
        new: dict[int, int] = {}
        for key, value in request.form.items():
            if key.startswith("page_"):
                try:
                    scan = int(key.split("_", 1)[1])
                except ValueError:
                    continue
                if value.strip():
                    try:
                        new[scan] = int(value.strip())
                    except ValueError:
                        continue
        PAGE_REVIEW = new
        cfg["page_review_json"].parent.mkdir(parents=True, exist_ok=True)
        cfg["page_review_json"].write_text(
            json.dumps({str(k): v for k, v in new.items()}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        _rematch(cfg)
        return redirect(url_for("page_review", saved=1))

    @app.route("/rematch", methods=["POST"])
    def rematch():
        with LOCK:
            if STATE["running"]:
                return redirect(url_for("index"))
        _rematch(cfg)
        return redirect(url_for("index"))

    @app.route("/save", methods=["POST"])
    def save():
        if not cfg["pages_tsv"].exists():
            return "tables do not exist yet", 400
        n = int(request.form["page"])
        block_id = request.form.get("last_block", "").strip()
        if block_id and block_id not in _DOCS:
            return _unknown_block_msg(block_id), 400
        status = request.form.get("status", "")
        if status == "low-score" and block_id in _DOCS:
            status = "ok"
        score_raw = request.form.get("score", "").strip()
        if score_raw:
            score = float(score_raw)
        else:
            m = find_page(n)
            score = m.score if m else 0.0
        o = {
            "last_block": block_id,
            "position": request.form.get("position", ""),
            "status": status,
            "score": score,
            "reviewed": True,
        }
        with LOCK:
            REVIEW[str(n)] = o
        cfg["review_json"].parent.mkdir(parents=True, exist_ok=True)
        cfg["review_json"].write_text(json.dumps(REVIEW, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
        return redirect(url_for("page", n=n, saved=1))

    @app.route("/export")
    def export():
        if not cfg["pages_tsv"].exists():
            return "nothing to export", 400
        with LOCK:
            rows = [apply_overrides(m) for m in MATCHES]
            rgpage.build_spans(rows, list(_DOCS.values()), _ENTRY_PAGES)
        out = cfg["pages_tsv"].with_name(
            f"rg{cfg['vol']}_pages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
        rgpage.write_pages_tsv(rows, out, overridden={int(k) for k in REVIEW})
        return redirect(url_for("index", tab="export"))

    @app.route("/download/pages")
    def download_pages():
        if not cfg["pages_tsv"].exists():
            return "nothing to export", 400
        with LOCK:
            rows = [apply_overrides(m) for m in MATCHES]
            rgpage.build_spans(rows, list(_DOCS.values()), _ENTRY_PAGES)
        out = cfg["pages_tsv"].with_name(
            f"rg{cfg['vol']}_pages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
        rgpage.write_pages_tsv(rows, out, overridden={int(k) for k in REVIEW})
        return send_file(out, as_attachment=True,
                         download_name=out.name, mimetype="text/tab-separated-values")

    @app.route("/download/report")
    def download_report():
        rows = build_report()
        out = cfg["pages_tsv"].with_name(f"rg{cfg['vol']}_page_per_regest.tsv")
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["petentenvita", "last regest/head", "kind",
                        "page_start", "page_end", "span", "conf", "overridden"])
            for r in rows:
                w.writerow([r["entry"], r["block_id"], r["kind"],
                            r["page_start"], r["page_end"], r["span"],
                            r["score"], r["overridden"]])
        return send_file(out, as_attachment=True,
                         download_name=out.name, mimetype="text/tab-separated-values")

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse
    import webbrowser

    p = argparse.ArgumentParser(description="RG review web panel.")
    p.add_argument("--vol", type=int, default=8)
    p.add_argument("--images", type=Path)
    p.add_argument("--xml", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--pages-tsv", type=Path)
    p.add_argument("--blocks-tsv", type=Path)
    p.add_argument("--review-json", type=Path)
    p.add_argument("--live-dir", type=Path)
    p.add_argument("--first-scan", type=int)
    p.add_argument("--first-page", type=int)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parent
    vc = rgpage.resolve_volume_config(args.vol)
    cfg = {
        "vol": args.vol,
        "xml_path": args.xml or vc["xml"],
        "cache": args.cache or (vc["debug_dir"] / f"rg{args.vol}" / "ocr_pages.jsonl"),
        "pages_tsv": args.pages_tsv or (vc["output_dir"] / f"rg{args.vol}_pages.tsv"),
        "blocks_tsv": args.blocks_tsv or (vc["output_dir"] / f"rg{args.vol}_blocks.tsv"),
        "review_json": args.review_json or (vc["output_dir"] / f"rg{args.vol}_review.json"),
        "live_dir": args.live_dir or (vc["output_dir"] / f"rg{args.vol}_live"),
        "images_dir": args.images or vc["images"],
        "first_scan": args.first_scan or vc["first_scan_no"],
        "first_page": args.first_page or vc["first_logical_page"],
        "parts": vc["parts"],
    }
    app = build_app(cfg, volumes=rgpage.load_volumes())
    url = f"http://{args.host}:{args.port}"
    print(f"RG{args.vol} panel: {url}  (Ctrl-C to stop)")
    webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
