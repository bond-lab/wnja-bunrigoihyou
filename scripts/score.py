#!/usr/bin/env python3
"""Score candidate pairs and write the wnja to WLSP mapping with confidences.

A classifier (shallow gradient-boosted trees by default, or a logistic
regression with ``--model lr``) is trained on the labelled rows of the candidate table
(gold pairs from the dev set against the other candidates of the same senses),
using every feature column present: the structural features from
``candidates.py`` plus any tier-specific columns (``chive_*``, ``sent_*``,
``judge_*``) that later scripts have added.  Cross-validated probabilities
(``p_cv``, grouped by sense so a sense never sees itself in training) are
stored for honest evaluation, then the model is refit on all labelled rows and
applied to every candidate pair as ``p``.

Outputs: ``build/scored.parquet`` (all rows with ``p``), the mapping TSV for
pairs with ``p`` at or above the threshold, and a review TSV for borderline pairs.

Usage::

    uv run scripts/score.py build/candidates.parquet -o wnja-wlsp.tsv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ID_COLUMNS = (
    "sense_id",
    "synset_id",
    "ili",
    "lemma",
    "pos",
    "midashi_no",
    "record_id",
    "code",
    "labels",
    "wlsp_lemma",
    "pos_wlsp",
)
NON_FEATURES = set(ID_COLUMNS) | {"lex", "gold", "gold_rel", "p", "p_cv", "baseline", "evidence"}
# Priors about how many candidates or senses a word has.  They help top-1
# accuracy but penalise genuine many-to-many links, so they are off by default.
PRIORS = {"n_cands", "n_senses", "sense_rank", "inv_cands", "inv_senses"}
LEX_TYPES = ("lemma", "form", "suru", "yomi")
POS_PAIRS = tuple(f"{a}{b}" for a in "nvar" for b in "nvar")


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived feature columns: one-hot match type and POS pair, fractions, logs.

    Args:
        df: Candidate table

    Returns:
        The table with extra numeric columns
    """
    cols: list[pl.Expr] = [
        *[(pl.col("lex") == t).cast(pl.Int8).alias(f"lex_{t}") for t in LEX_TYPES],
        *[
            ((pl.col("pos") + pl.col("pos_wlsp")) == pp).cast(pl.Int8).alias(f"pp_{pp}")
            for pp in POS_PAIRS
        ],
        (1.0 / pl.col("n_cands")).alias("inv_cands"),
        (1.0 / pl.col("n_senses")).alias("inv_senses"),
    ]
    for c in df.columns:
        if c.startswith("ov_"):
            rel = c.split("_", 2)[1]
            cols.append((pl.col(c) / pl.max_horizontal(pl.col(f"n_{rel}"), 1)).alias(f"fr_{c[3:]}"))
            cols.append((pl.col(c) > 0).cast(pl.Int8).alias(f"any_{c[3:]}"))
        elif c.startswith("size_"):
            cols.append(pl.col(c).log1p().alias(f"log_{c}"))
    return df.with_columns(cols)


def feature_columns(df: pl.DataFrame, priors: bool = False) -> list[str]:
    """Numeric columns used as model features.

    Args:
        df: Candidate table after ``add_derived``
        priors: Whether to include the candidate-count and sense-count priors

    Returns:
        Column names, in table order
    """
    excluded = NON_FEATURES if priors else NON_FEATURES | PRIORS
    return [
        c
        for c, dtype in zip(df.columns, df.dtypes, strict=True)
        if c not in excluded and dtype.is_numeric()
    ]


def baseline_score(df: pl.DataFrame) -> pl.Series:
    """Hand-weighted overlap score, the pre-model baseline.

    Args:
        df: Candidate table

    Returns:
        A float series, higher is better
    """
    expr = (
        3 * pl.col("ov_syn_subpara")
        + 2 * pl.col("ov_syn_para")
        + pl.col("ov_syn_item")
        + pl.col("ov_hyper_item")
        + pl.col("ov_hypo_item")
        + pl.col("ov_sim_subpara")
        + 0.5 * pl.col("pos_ok")
    )
    return df.select(expr.alias("baseline"))["baseline"]


def make_model(kind: str = "gbt") -> Pipeline:
    """Build the scoring model.

    Args:
        kind: ``gbt`` for shallow gradient-boosted trees (default, best on dev)
            or ``lr`` for a scaled, regularised logistic regression

    Returns:
        An unfitted scikit-learn pipeline

    Raises:
        ValueError: If ``kind`` is not recognised
    """
    if kind == "gbt":
        return make_pipeline(
            GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
            )
        )
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
        )
    raise ValueError(f"unknown model kind {kind!r}")


def top_weights(model: Pipeline, features: list[str], n: int = 12) -> str:
    """Summarise the most influential features of a fitted model.

    Args:
        model: Fitted pipeline from ``make_model``
        features: Feature names in training order
        n: How many to list

    Returns:
        A comma-separated ``name=weight`` string
    """
    est = model[-1]
    weights = est.coef_[0] if hasattr(est, "coef_") else est.feature_importances_
    top = sorted(zip(features, weights, strict=True), key=lambda t: -abs(t[1]))[:n]
    return ", ".join(f"{f}={w:+.2f}" for f, w in top)


def fit_and_predict(
    df: pl.DataFrame, features: list[str], kind: str = "gbt", folds: int = 5
) -> pl.DataFrame:
    """Cross-validate on labelled rows, refit on all of them, predict every row.

    Args:
        df: Candidate table with derived features
        features: Feature column names
        kind: Model kind for ``make_model``
        folds: Number of GroupKFold folds (groups are senses)

    Returns:
        The table with ``p_cv`` (dev-set rows only) and ``p`` (all rows)
    """
    x_all = df.select(features).fill_null(0.0).to_numpy().astype(float)
    labelled = df["gold"].is_not_null().to_numpy()
    x = x_all[labelled]
    y = df["gold"].drop_nulls().to_numpy()
    groups = df.filter(pl.col("gold").is_not_null())["sense_id"].to_numpy()
    in_dev = df["gold_rel"].is_not_null().to_numpy() if "gold_rel" in df.columns else labelled
    sense_ids = df["sense_id"].to_numpy()
    p_cv = np.full(len(df), np.nan)
    for train, test in GroupKFold(n_splits=folds).split(x, y, groups):
        model = make_model(kind).fit(x[train], y[train])
        # predict every dev-set row (labelled or not) of the held-out senses
        held = in_dev & np.isin(sense_ids, np.unique(groups[test]))
        p_cv[held] = model.predict_proba(x_all[held])[:, 1]
    model = make_model(kind).fit(x, y)
    p = model.predict_proba(x_all)[:, 1]
    logger.info("most influential features: %s", top_weights(model, features))
    return df.with_columns(pl.Series("p_cv", p_cv), pl.Series("p", p))


def evidence(df: pl.DataFrame) -> pl.Series:
    """Summarise the non-zero overlap features of each row as text.

    Args:
        df: Candidate table

    Returns:
        A string series like ``lemma;syn_subpara=2;hyper_item=1``
    """
    ov = [c for c in df.columns if c.startswith("ov_")]
    parts = [pl.col("lex")] + [
        pl.when(pl.col(c) > 0).then(pl.lit(c[3:] + "=") + pl.col(c).cast(pl.Utf8)).otherwise(None)
        for c in ov
    ]
    return df.select(pl.concat_str(parts, separator=";", ignore_nulls=True).alias("evidence"))[
        "evidence"
    ]


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidates", type=Path, nargs="?", default=Path("build/candidates.parquet"))
    ap.add_argument("-o", "--output", type=Path, default=Path("wnja-wlsp.tsv"))
    ap.add_argument("--scored", type=Path, default=Path("build/scored.parquet"))
    ap.add_argument("--review", type=Path, default=Path("build/review.tsv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--review-threshold", type=float, default=0.2)
    ap.add_argument(
        "--priors", action="store_true", help="include candidate/sense count priors as features"
    )
    ap.add_argument("--model", choices=("gbt", "lr"), default="gbt")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = add_derived(pl.read_parquet(args.candidates))
    features = feature_columns(df, priors=args.priors)
    logger.info("%d rows, %d features", len(df), len(features))
    df = df.with_columns(baseline_score(df))
    df = fit_and_predict(df, features, args.model)
    df = df.with_columns(evidence(df), pl.col("p").round(3), pl.col("p_cv").round(3))
    df.write_parquet(args.scored)

    out_cols = [*ID_COLUMNS, "p", "evidence"]
    mapping = df.filter(pl.col("p") >= args.threshold).sort(
        ["sense_id", "p"], descending=[False, True]
    )
    mapping.select(out_cols).rename({"p": "confidence"}).write_csv(args.output, separator="\t")
    review = df.filter((pl.col("p") >= args.review_threshold) & (pl.col("p") < args.threshold))
    review.sort("p", descending=True).select(out_cols).write_csv(args.review, separator="\t")
    logger.info(
        "wrote %s (%d links for %d senses) and %s (%d borderline pairs)",
        args.output,
        len(mapping),
        mapping["sense_id"].n_unique(),
        args.review,
        len(review),
    )


if __name__ == "__main__":
    main()
