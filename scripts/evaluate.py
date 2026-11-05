#!/usr/bin/env python3
"""Evaluate a scored candidate table against the dev-set labels.

Uses the cross-validated probabilities (``p_cv``) of the dev-set rows so the
numbers are not inflated by training on the same senses.  Two views:

- ranking over *ambiguous* senses (two or more candidates): top-1 accuracy
  and mean reciprocal rank, both strict (the gold record) and lenient (any
  record in the gold's 小段落 or 段落, since WLSP lists the same headword more
  than once nearby and those are not wrong links)
- many-to-many: precision, recall and F1 of the links kept at each threshold,
  strict and lenient

Appends a block per run to ``dev/eval.md``.

Usage::

    uv run scripts/evaluate.py build/scored.parquet --name core
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

LENIENT = ("exact", "same_subpara", "same_para")


def with_hit(df: pl.DataFrame, lenient: bool) -> pl.DataFrame:
    """Add a ``hit`` column marking candidates that count as correct.

    Args:
        df: Dev-set rows (``gold_rel`` not null)
        lenient: Count same-小段落 and same-段落 candidates as correct too

    Returns:
        The table with an integer ``hit`` column
    """
    ok = LENIENT if lenient else ("exact",)
    return df.with_columns(pl.col("gold_rel").is_in(ok).cast(pl.Int8).alias("hit"))


def ranking_metrics(df: pl.DataFrame, score: str) -> dict[str, float]:
    """Top-1 accuracy and MRR over ambiguous dev senses.

    Args:
        df: Dev-set rows with a ``hit`` column
        score: Column to rank candidates by

    Returns:
        ``{"senses": n, "top1": acc, "mrr": mrr}``
    """
    ranked = (
        df.sort([pl.col("sense_id"), pl.col(score)], descending=[False, True])
        .with_columns(
            pl.int_range(pl.len()).over("sense_id").alias("rank"),
            pl.len().over("sense_id").alias("n"),
        )
        .filter((pl.col("n") > 1) & (pl.col("hit") == 1))
        .group_by("sense_id")
        .agg(pl.col("rank").min())
    )
    if ranked.is_empty():
        return {"senses": 0, "top1": 0.0, "mrr": 0.0}
    rank = ranked["rank"]
    return {
        "senses": len(ranked),
        "top1": float((rank == 0).mean()),
        "mrr": float((1.0 / (rank + 1)).mean()),
    }


def threshold_metrics(df: pl.DataFrame, score: str, threshold: float) -> dict[str, float]:
    """Precision, recall and F1 of pairs with ``score >= threshold``.

    Precision counts kept pairs that are hits; recall counts gold (exact)
    pairs that were kept.

    Args:
        df: Dev-set rows with a ``hit`` column
        score: Probability column
        threshold: Minimum score to keep a link

    Returns:
        ``{"precision": p, "recall": r, "f1": f}``
    """
    kept = df.filter(pl.col(score) >= threshold)
    tp = int((kept["hit"] == 1).sum())
    exact_kept = int((kept["gold_rel"] == "exact").sum())
    n_gold = int((df["gold_rel"] == "exact").sum())
    precision = tp / len(kept) if len(kept) else 0.0
    recall = exact_kept / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def report_line(label: str, df: pl.DataFrame, score: str, thresholds: list[float]) -> str:
    """One markdown table row for a scoring column.

    Args:
        label: Row label
        df: Dev-set rows with a ``hit`` column
        score: Column to evaluate
        thresholds: Thresholds for the P/R/F1 cells (skipped for the baseline)

    Returns:
        A markdown table row
    """
    rm = ranking_metrics(df, score)
    line = f"| {label} | {rm['senses']} | {rm['top1']:.3f} | {rm['mrr']:.3f} |"
    for t in thresholds:
        if score == "baseline":
            line += " |"
        else:
            tm = threshold_metrics(df, score, t)
            line += f" {tm['precision']:.2f}/{tm['recall']:.2f}/{tm['f1']:.2f} |"
    return line


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scored", type=Path, nargs="?", default=Path("build/scored.parquet"))
    ap.add_argument("--name", required=True, help="label for this run in the report")
    ap.add_argument("--report", type=Path, default=Path("dev/eval.md"))
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = pl.read_parquet(args.scored).filter(pl.col("gold_rel").is_not_null())
    header = (
        "| run | ambiguous senses | top-1 | MRR | "
        + " | ".join(f"P/R/F1 @{t}" for t in args.thresholds)
        + " |\n|---|---|---|---|"
        + "---|" * len(args.thresholds)
        + "\n"
    )
    lines = []
    for view, lenient in (("strict", False), ("lenient", True)):
        d = with_hit(df, lenient)
        for label, col in ((f"baseline {view}", "baseline"), (f"{args.name} {view}", "p_cv")):
            line = report_line(label, d, col, args.thresholds)
            lines.append(line)
            logger.info(line)

    new = not args.report.exists()
    with args.report.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("# Dev-set evaluation\n\n")
        fh.write(f"\n## {date.today().isoformat()} {args.name}\n\n{header}")
        fh.write("\n".join(lines) + "\n")
    logger.info("appended to %s", args.report)


if __name__ == "__main__":
    main()
