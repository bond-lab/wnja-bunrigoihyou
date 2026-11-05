#!/usr/bin/env python3
"""Evaluate a scored candidate table against the dev-set labels.

Uses the cross-validated probabilities (``p_cv``) of the labelled rows so the
numbers are not inflated by training on the same senses.  Reports, for the
model and for the hand-weighted baseline:

- top-1 accuracy and mean reciprocal rank over *ambiguous* senses (two or more
  candidates), which is where scoring matters
- precision, recall and F1 of the links kept at each threshold, over all
  labelled pairs (many-to-many view)

Appends a row per run to ``dev/eval.md``.

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


def ranking_metrics(df: pl.DataFrame, score: str) -> dict[str, float]:
    """Top-1 accuracy and MRR over ambiguous labelled senses.

    Args:
        df: Labelled rows (``gold`` not null)
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
        .filter((pl.col("n") > 1) & (pl.col("gold") == 1))
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

    Args:
        df: Labelled rows
        score: Probability column
        threshold: Minimum score to keep a link

    Returns:
        ``{"precision": p, "recall": r, "f1": f}``
    """
    kept = df.filter(pl.col(score) >= threshold)
    tp = int((kept["gold"] == 1).sum())
    n_pos = int((df["gold"] == 1).sum())
    precision = tp / len(kept) if len(kept) else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scored", type=Path, nargs="?", default=Path("build/scored.parquet"))
    ap.add_argument("--name", required=True, help="label for this run in the report")
    ap.add_argument("--report", type=Path, default=Path("dev/eval.md"))
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = pl.read_parquet(args.scored).filter(pl.col("gold").is_not_null())
    lines = []
    for label, col in (("baseline", "baseline"), (args.name, "p_cv")):
        rm = ranking_metrics(df, col)
        line = f"| {label} | {rm['senses']} | {rm['top1']:.3f} | {rm['mrr']:.3f} |"
        if col == "p_cv":
            for t in args.thresholds:
                tm = threshold_metrics(df, col, t)
                line += f" {tm['precision']:.2f}/{tm['recall']:.2f}/{tm['f1']:.2f} |"
        else:
            line += " |" * len(args.thresholds)
        lines.append(line)
        logger.info(line)

    header = (
        "| run | ambiguous senses | top-1 | MRR | "
        + " | ".join(f"P/R/F1 @{t}" for t in args.thresholds)
        + " |\n|---|---|---|---|"
        + "---|" * len(args.thresholds)
        + "\n"
    )
    new = not args.report.exists()
    with args.report.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("# Dev-set evaluation\n\n")
        fh.write(f"\n## {date.today().isoformat()} {args.name}\n\n{header}")
        fh.write("\n".join(lines) + "\n")
    logger.info("appended to %s", args.report)


if __name__ == "__main__":
    main()
