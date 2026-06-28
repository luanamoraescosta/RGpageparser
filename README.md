

# Overview

**RGpageparser** is an OCR-based pipeline developed to assign book page numbers to entries contained in *Regesta Pontificum* XML files. The system processes scanned book pages, extracts entry and page numbers, creates an entry-to-page mapping, and enriches the XML with page information.

# Repository Structure

| File                          | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `page_assignment.py`          | Main OCR pipeline and XML enrichment.                |
| `checkxml.py`                 | XML validation and consistency checks.               |
| `review_ambiguous_missing.py` | Manual and AI-assisted review of unresolved entries. |
| `rg8_enriched.xml`            | XML file enriched with page information.             |
| `debug/`                      | Diagnostic reports and quality-control files.        |

# Workflow Overview

The workflow starts by processing the scanned book images. Each image is first **deskewed** and split into its two text columns. OCR is then applied using multiple Tesseract configurations.

To identify page numbers, the system focuses on the **top and bottom margins** of the page, where printed page numbers are usually located. To identify entry numbers, it analyzes the **left margins of each column**, where new lemmas typically begin.

The OCR output is filtered using a series of **positional and lexical heuristics** in order to distinguish actual entry numbers from dates, archival references, and OCR noise. Common OCR mistakes are automatically corrected.

After extraction, the system builds an **entry-to-page mapping**, infers some missing entries when possible, and inserts `<page>` elements into the corresponding `<sublemma>` elements in the XML file.

```{mermaid}
flowchart TD
    A[Scanned Images] --> B[Image Preprocessing]
    B --> C[OCR Extraction]
    C --> D[Heuristic Filtering]
    D --> E[Entry-to-Page Mapping]
    E --> F[XML Enrichment]
    F --> G[Validation and Review]
```

# Installation

## Requirements

* Python ≥ 3.10
* Tesseract OCR

Install dependencies:

```bash
pip install opencv-python pillow numpy pytesseract openai
```

# Usage

Run the main pipeline:

```bash
python page_assignment.py
```

Validate the resulting XML:

```bash
python checkxml.py
```

Review ambiguous or missing entries:

```bash
python review_ambiguous_missing.py
```

For AI-assisted review:

```bash
python review_ambiguous_missing.py --use-qwen
```

# Outputs

The pipeline produces:

* `rg8_enriched.xml`: XML enriched with page numbers.
* `page_mapping.json`: entry-to-page mapping.
* `debug/*.tsv`: validation and diagnostic reports.

# Citation

If you use this software in research, please cite the repository appropriately.
