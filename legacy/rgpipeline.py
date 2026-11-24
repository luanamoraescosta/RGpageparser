#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Single entry point for the RG page parser.

Instead of remembering seven numbered scripts, use this command:

    python rgpipeline.py run    --volume 8      # OCR + match + write + audit
    python rgpipeline.py review --volume 8      # manual ambiguous mapping + blocks
    python rgpipeline.py check  --volume 8      # audit only
    python rgpipeline.py status --volume 8      # show where the run stands
    python rgpipeline.py all    --volume 8      # run, then review

The individual stage modules remain available for debugging, but are hidden
behind these four commands for normal use.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import stage1_build_page_mapping as stage1
import stage2_review_ambiguous_mapping as stage2
import stage2b_manual_entry_override as stage2b
import stage3_locate_sublemmas as stage3
import stage3b_manual_sublemma_override as stage3b
import stage4_build_enriched_xml as stage4
import stage5_check_enriched_xml as stage5
from pipeline_config import CFG, add_common_args, configure_from_args

COMMON_FLAGS = {
    "volume": "--volume",
    "xml": "--xml",
    "images": "--images",
    "base_dir": "--base-dir",
    "debug_dir": "--debug-dir",
    "output": "--output",
    "tesseract": "--tesseract",
    "first_scan_no": "--first-scan-no",
    "first_logical_page": "--first-logical-page",
}


def common_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    for dest, flag in COMMON_FLAGS.items():
        value = getattr(args, dest, None)
        if value is not None:
            argv.extend([flag, str(value)])
    return argv


def _stage(name: str, module, extra: list[str], args: argparse.Namespace) -> int:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    return module.main([name] + common_argv(args) + extra)


def cmd_run(args: argparse.Namespace, *, announce: bool = True) -> int:
    stage1_extra: list[str] = []
    if args.rebuild:
        stage1_extra.append("--rebuild")
    if args.test is not None:
        stage1_extra += ["--test", str(args.test)]

    rc = _stage("run: OCR scans -> entry/page mapping", stage1, stage1_extra, args)
    if rc:
        return rc

    stage3_extra: list[str] = []
    if args.limit is not None:
        stage3_extra += ["--limit", str(args.limit)]
    if args.review:
        stage3_extra.append("--review")
    if getattr(args, "debug_images", 0):
        stage3_extra += ["--debug-images", str(args.debug_images)]

    rc = _stage("run: fuzzy match pages -> XML blocks", stage3, stage3_extra, args)
    if rc:
        return rc

    rc = _stage("run: write enriched XML", stage4, ["--write-xml"], args)
    if rc:
        return rc

    rc = _stage("run: audit enriched XML", stage5, [], args)
    if rc:
        return rc

    if announce:
        print(
            "\nAutomatic pipeline finished.\n"
            "Manual review (optional):  python rgpipeline.py review --volume "
            f"{CFG.volume if CFG.volume is not None else 'N'}"
        )
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    common = ["review"] + common_argv(args)

    if args.entry is not None and args.page is not None:
        return stage2b.main(common + ["--entry", str(args.entry), "--page", str(args.page)])

    if args.mapping:
        if args.page is not None:
            return stage2b.main(common + ["--page", str(args.page)])
        return stage2.main(common)

    if args.blocks:
        extra = ["--page", str(args.page)] if args.page is not None else ["--queue-unresolved"]
        return stage3b.main(common + extra)

    if args.page is not None:
        return stage3b.main(common + ["--page", str(args.page)])

    rc = stage2.main(common)
    if rc:
        return rc
    return stage3b.main(common + ["--queue-unresolved"])


def cmd_status(args: argparse.Namespace) -> int:
    configure_from_args(args)

    print(f"Volume      : {CFG.volume if CFG.volume is not None else '(not set)'}")
    print(f"Source XML  : {CFG.xml_input} {'[ok]' if CFG.xml_input.exists() else '[MISSING]'}")
    print(f"Images      : {CFG.image_dir}", end="")
    if CFG.image_dir.is_dir():
        count = sum(1 for p in CFG.image_dir.iterdir() if p.suffix.lower() in CFG.image_extensions)
        print(f" [{count} files]")
    else:
        print(" [MISSING]")
    print(f"Debug dir   : {CFG.debug_dir}")
    print(f"Output XML  : {CFG.xml_output} {'[ok]' if CFG.xml_output.exists() else '[not written]'}")
    print(f"First scan  : {CFG.first_scan_no} (logical page {CFG.first_logical_page})")

    artifacts = [
        ("entry -> page mapping", CFG.mapping_tsv),
        ("ambiguous entries", CFG.ambiguous_mapping_tsv),
        ("page matches", CFG.matches_tsv),
        ("reviewed matches", CFG.matches_reviewed_tsv),
        ("page spans", CFG.spans_tsv),
    ]
    print("\nProgress:")
    for label, path in artifacts:
        print(f"  {label:<24}: {'done' if path.exists() else 'missing'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RG page parser with a single command."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="OCR, match, write XML and audit")
    add_common_args(run)
    run.add_argument("--rebuild", action="store_true", help="Ignore the stage-1 cache and re-OCR.")
    run.add_argument("--test", type=int, default=None, help="Only OCR the first N scans (testing).")
    run.add_argument("--limit", type=int, default=None, help="Only match the first N scans (testing).")
    run.add_argument("--review", action="store_true", help="Open stage-3 interactive match review.")
    run.add_argument("--debug-images", type=int, default=0, help="Save annotated OCR images for the first N pages.")

    review = sub.add_parser("review", help="Manual corrections (mapping and/or block matches)")
    add_common_args(review)
    review.add_argument("--mapping", action="store_true", help="Review ambiguous entry -> page mappings.")
    review.add_argument("--blocks", action="store_true", help="Review unresolved page -> block matches.")
    review.add_argument("--page", type=int, default=None, help="Jump to a specific page.")
    review.add_argument("--entry", type=int, default=None, help="Set one entry/page pair directly.")

    check = sub.add_parser("check", help="Audit the enriched XML")
    add_common_args(check)

    status = sub.add_parser("status", help="Show paths and pipeline progress")
    add_common_args(status)

    all_cmd = sub.add_parser("all", help="Run the pipeline, then the manual reviews")
    add_common_args(all_cmd)
    all_cmd.add_argument("--rebuild", action="store_true")
    all_cmd.add_argument("--test", type=int, default=None)
    all_cmd.add_argument("--limit", type=int, default=None)
    all_cmd.add_argument("--review", action="store_true")
    all_cmd.add_argument("--debug-images", type=int, default=0)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:] if argv is not None else None)
    configure_from_args(args)

    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "review":
            return cmd_review(args)
        if args.command == "check":
            return stage5.main(["check"] + common_argv(args))
        if args.command == "status":
            return cmd_status(args)
        if args.command == "all":
            rc = cmd_run(args, announce=False)
            if rc:
                return rc
            return cmd_review(argparse.Namespace(
                **vars(args), mapping=False, blocks=False, page=None, entry=None
            ))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
