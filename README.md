# RGpageparser

For each page scan of a *Repertorium Germanicum* volume, answer two questions:

1. **Which printed page number is on the scan?**
2. **Which entry block is at the bottom of the page, and does it end there or
   continue on the next page?** The block can be the `head` of the entry or a
   regest (`rg1`, `rg2`, …).

`rgpage.py` is self-contained: it reads the scans and the RG XML directly, runs
Tesseract, and writes both tables and annotated images. It does **not** depend
on the old debug tables in `legacy/`.

## Commands

```bash
# OCR every scan and build the page/block tables
python rgpage.py run --vol 8

# same, with the EasyOCR safety net and a live OpenCV window
python rgpage.py run --vol 8 --easyocr --show

# only OCR and cache, no matching
python rgpage.py scan --vol 8

# rebuild tables from an existing cache (fast, no OCR)
python rgpage.py match --vol 8

# 20 random annotated pages from the cache
python rgpage.py match --vol 8 --random 20 --seed 42

# a small trial run: first 20 scans
python rgpage.py run --vol 8 --limit 20 --random 5
```

After `run` and `match` the **review panel opens automatically** in the
browser. It is the single interface: click **▶ Run OCR** to run the OCR from the
web, watch the live prediction, see the pages that need review, and correct
block / position / status / score (saved to `output/rgN_review.json`). Use
`--no-review` to skip it, or open it later on its own:

```bash
python rgpage.py review --vol 8        # open the panel (no re-run)
```

Defaults: images in `H4 6481 (8,1`, XML in `../RG_data/rg_xml/rg8.xml`,
first scan `71` = printed page `1`, cache in `debug/rg8/ocr_pages.jsonl`,
outputs in `output/`.

Useful options:

| Option | Meaning |
|---|---|
| `--from-scan N`, `--to-scan N`, `--limit N` | restrict the scan range |
| `--rebuild` | ignore the OCR cache and scan again |
| `--easyocr` | enable the optional EasyOCR fallback |
| `--show` | show the live annotated frame in an OpenCV window (ESC quits) |
| `--delay MS` | delay between live frames |
| `--random N`, `--seed N` | number of annotated samples and random seed |
| `--live-dir DIR` | where `latest.jpg` / `recent.jpg` are written |
| `--pages-out`, `--blocks-out`, `--images-out` | override output paths |
| `--no-review` | don't open the review panel after `run` / `match` |
| `--port N` | port for the review panel (default 5000) |

## Outputs

For `--vol 8`:

| File | Contents |
|---|---|
| `output/rg8_pages.tsv` | One row per scan: `page`, `scan`, `page_source`, `entries_on_page`, `last_entry`, `last_block`, `last_kind`, `position`, `position_en`, `span_start`, `span_end`, `status`, `score`, `overridden`, `ocr_last_lines`. |
| `output/rg8_blocks.tsv` | One row per `head`/`regest`: `entry`, `block_id`, `label`, `kind`, `page_start`, `page_end`, `pages`, `status` (`complete` / `continues`), `source`, `warning`. |
| `output/rg8_live/latest.jpg` | Most recent annotated scan (updated during `run`). |
| `output/rg8_live/recent.jpg` | Last few annotated scans in a strip. |
| `output/rg8_randomN/` | Annotated samples and `contact_sheet.jpg`. |
| `output/rg8_review.json` | Manual review overrides saved by the web app. |
| `output/rg8_pages_reviewed.tsv` | Written by the web app's `/export`: TSV + overrides. |
| `output/rg8_report.tsv` | Written by the web app's `/download/report`: per-block `entry`, `block_id`, `kind`, `page_start`, `page_end`, `span`, `conf`, `overridden`. |

Annotation colors:

- **green** = printed page number;
- **blue** = margin entry number;
- **orange** = last printed lines used for matching;
- **red** = the matched last block.

The black panel shows scan, page, entries on the page, last block/label,
position and score. The review panel is in English; the position dropdown is a
binary choice: **continues** (the block runs on to the next page) or
**ends here** (it stops on this page). The underlying values remain
`starts-here`, `continues`, `ends-here`, `complete-on-page`.

## How it works

1. **Page number.** Tesseract reads the page once; candidate page numbers are
   isolated in the bottom/top centre strip. The scan sequence is then
   normalized against `first_scan`/`first_page`, including gaps such as the
   missing scans 544–545, so a misread digit does not shift later pages.
2. **Entry numbers on the page.** Each column is split at the gutter and the
   margin strip is OCRed (PSM 6, then PSM 11 / lower confidence as fallback).
   A number is kept only if it exists in the XML and its nearby text looks like
   a name, not an archive reference. A per-page median filter removes outliers;
   same-page and neighbouring-page gaps are filled.
3. **Last block.** The right column is OCRed, handwriting-sized/low-confidence
   lines are dropped, and the last two printed lines are compared with the
   `<head>` / `<regest>` texts of entries near this page using
   `rapidfuzz.partial_ratio`. The alignment tells whether the OCR text sits at
   the start (`starts-here`), end (`ends-here`), both (`complete-on-page`) or
   middle (`continues`) of the block.
4. **Spans.** Matched blocks become anchors. A block's span starts at its first
   matched page and is extended forward by one page only when it
   `continues`/`starts-here`; it is never extended backwards. Unmatched blocks
   are inferred from their entry page. Spans are forced to be monotonic and
   truncated if they exceed six pages.
5. **EasyOCR fallback.** With `--easyocr`, EasyOCR runs only on the bottom 40%
   of pages whose Tesseract score is below 55. Its result replaces the
   Tesseract match only if it is more than 5 points better.
6. **Blank pages.** A page with mean OCR confidence below 50 and no margin
   entries is marked `blank-page`.

Runtime on the development Mac is about 5 s per scan (roughly 70 minutes for
839 scans). OCR results are cached after every 10 scans, so an interrupted
`run` can be resumed; `match` reuses the cache and takes seconds.

## Requirements

- Python 3.9+
- Tesseract (`brew install tesseract`)
- Python packages:

```bash
pip install opencv-python numpy pytesseract Pillow rapidfuzz
# optional, only for --easyocr:
pip install easyocr
```

## Code map

| File | Purpose |
|---|---|
| `rgpage.py` | The self-contained page/block locator and annotator. |
| `review_app.py` | The web review panel (review/correct/export) served by `rgpage.py review`. |
| `rg_xml.py` | Schema-aware XML reader (`<petentenvita>/<head>/<regest>`, legacy `<lemma>/<sublemma>`, milestone regests). |
| `imaging.py` | Image loading and deskew helpers. |
| `legacy/` | Archived five-stage pipeline. Not used by `rgpage.py`. See `legacy/README.md`. |

## RG XML model

Each volume is one file with `<petentenvita>` containers, each holding a
`<head>` and one or more `<regest>` elements:

```xml
<rg vol="8">
  <petentenvita id="10800001">
    <head id="10800001-0">…</head>
    <regest id="10800001-1">…</regest>
    <regest id="10800001-2">…</regest>
  </petentenvita>
</rg>
```

The entry number is `id[3:]` (`10800001` → entry `1`). Blocks are labelled
`head`, `rg1`, `rg2`, … according to the suffix. Milestone regests
(`<regest milestone="start" id="ov2"/>`) are supported by `rg_xml.py`.
