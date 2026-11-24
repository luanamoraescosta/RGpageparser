# Legacy five-stage pipeline (archived)

These files are the previous full pipeline:

- `stage1_build_page_mapping.py`
- `stage2_review_ambiguous_mapping.py`
- `stage2b_manual_entry_override.py`
- `stage3_locate_sublemmas.py`
- `stage3b_manual_sublemma_override.py`
- `stage4_build_enriched_xml.py`
- `stage5_check_enriched_xml.py`
- `rgpipeline.py`
- `interactive_display.py`

They are kept only to reproduce the already generated
`output/rg8_enriched_pages.xml` and the reports in `debug/rg8/`.

The new simple entry point is `../rgpage.py`; it answers the two questions we
actually need: the printed page number and the RG/head block at the bottom of
the page (with continuation).

To run an archived stage from the repository root (Python needs the root on
`sys.path`, which `-m` provides):

```bash
python -m legacy.stage1_build_page_mapping --volume 8
python -m legacy.stage3_locate_sublemmas --volume 8
```

Note: `legacy/rgpipeline.py` still imports the stage modules by their old
top-level names, so it is not runnable after the move. Use the individual
`-m legacy.stageN...` commands instead.
