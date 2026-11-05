#!/usr/bin/env python3
"""Add chiVe word-vector similarity features to the candidate table (tier 1).

chiVe (Works Applications) gives 300-d word2vec vectors for Sudachi-normalised
Japanese words.  For each (wnja sense, WLSP sense) pair the features compare
the wnja word and its synset members with the WLSP sense's neighbours:

- ``chive_sub_max`` / ``chive_sub_mean``: cosine between the word and the other
  headwords of the candidate's 小段落
- ``chive_para_max`` / ``chive_para_mean``: the same for the 段落
- ``chive_syn_sub_max``: best cosine between any other synset member and any
  小段落 neighbour
- ``chive_syn_sub``, ``chive_syn_para``, ``chive_syn_item``: cosine between the
  centroid of the other synset members and the centroid of the 小段落, 段落
  and 分類項目 headwords
- ``chive_has``: 1 if the wnja word has a vector, else 0 (other features null)

Words missing from the vocabulary are tokenised with Sudachi and their token
vectors averaged.  Columns are added in place to the candidate parquet.

Usage::

    uv run scripts/features_chive.py build/candidates.parquet \
        --kv build/chive/chive-1.3-mc30_gensim/chive-1.3-mc30.kv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import wn
from gensim.models import KeyedVectors
from sudachipy import Dictionary, SplitMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wlsp import Wlsp

logger = logging.getLogger(__name__)

FEATURES = (
    "chive_sub_max",
    "chive_sub_mean",
    "chive_para_max",
    "chive_para_mean",
    "chive_syn_sub_max",
    "chive_syn_sub",
    "chive_syn_para",
    "chive_syn_item",
)


class Vectors:
    """Unit word vectors with a Sudachi fallback for out-of-vocabulary words."""

    def __init__(self, kv: KeyedVectors, tokenizer: object | None = None) -> None:
        """Wrap a gensim KeyedVectors.

        Args:
            kv: Loaded chiVe vectors
            tokenizer: A sudachipy tokenizer; None disables the fallback
        """
        self.kv = kv
        self.tokenizer = tokenizer
        self.cache: dict[str, np.ndarray | None] = {}
        self.n_direct = self.n_split = self.n_missing = 0

    def _unit(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n else v

    def get(self, word: str) -> np.ndarray | None:
        """Unit vector for ``word``, or None if nothing is known.

        Args:
            word: A written form

        Returns:
            A 1-d float32 array or None
        """
        if word in self.cache:
            return self.cache[word]
        vec: np.ndarray | None = None
        if word in self.kv:
            vec = self._unit(self.kv[word])
            self.n_direct += 1
        elif self.tokenizer is not None:
            toks = [m.normalized_form() for m in self.tokenizer.tokenize(word, SplitMode.C)]
            parts = [self.kv[t] for t in toks if t in self.kv]
            if parts:
                vec = self._unit(np.mean(parts, axis=0))
                self.n_split += 1
        if vec is None:
            self.n_missing += 1
        self.cache[word] = vec
        return vec


def centroid(vecs: list[np.ndarray]) -> np.ndarray | None:
    """Unit centroid of a list of unit vectors.

    Args:
        vecs: Possibly empty list of vectors

    Returns:
        The normalised mean, or None if the list is empty
    """
    if not vecs:
        return None
    m = np.mean(vecs, axis=0)
    n = np.linalg.norm(m)
    return m / n if n else m


def stack(vecs: list[np.ndarray]) -> np.ndarray | None:
    """Stack vectors into a matrix, or None if there are none.

    Args:
        vecs: Possibly empty list of vectors

    Returns:
        A 2-d array or None
    """
    return np.vstack(vecs) if vecs else None


class GroupVectors:
    """Per-group matrices and centroids of WLSP headword vectors."""

    def __init__(self, db: Wlsp, vectors: Vectors) -> None:
        """Compute vectors for every WLSP sense and group them.

        Args:
            db: The indexed WLSP database
            vectors: Vector lookup
        """
        self.db = db
        self.sense_vec: dict[str, np.ndarray | None] = {}
        for mid, rec in db.senses.items():
            vec = vectors.get(rec.lemma)
            if vec is None:
                for form in db.forms[mid]:
                    vec = vectors.get(form)
                    if vec is not None:
                        break
            self.sense_vec[mid] = vec
        self.members: dict[str, dict[object, list[str]]] = {}
        self.centroids: dict[str, dict[object, np.ndarray | None]] = {}
        for level in ("subpara", "para", "item"):
            members = defaultdict(list)
            for key, mids in db.groups[level].items():
                members[key] = [m for m in mids if self.sense_vec[m] is not None]
            self.members[level] = dict(members)
            self.centroids[level] = {
                key: centroid([self.sense_vec[m] for m in mids]) for key, mids in members.items()
            }

    def neighbour_matrix(self, mid: str, level: str) -> np.ndarray | None:
        """Matrix of vectors of the other senses in ``mid``'s group.

        Args:
            mid: 見出し番号 of the candidate sense
            level: ``subpara`` or ``para``

        Returns:
            A 2-d array or None if no neighbour has a vector
        """
        rec = self.db.senses[mid]
        mids = self.members[level][self.db.group_key(rec, level)]
        return stack([self.sense_vec[m] for m in mids if m != mid])

    def group_centroid(self, mid: str, level: str) -> np.ndarray | None:
        """Centroid of all senses in ``mid``'s group at ``level``."""
        rec = self.db.senses[mid]
        return self.centroids[level].get(self.db.group_key(rec, level))


def pair_features(
    word_vec: np.ndarray | None,
    syn_vecs: list[np.ndarray],
    mid: str,
    groups: GroupVectors,
) -> dict[str, float | None]:
    """Similarity features for one (wnja sense, WLSP sense) pair.

    Args:
        word_vec: Unit vector of the wnja word, or None
        syn_vecs: Unit vectors of the other members of the synset
        mid: 見出し番号 of the candidate sense
        groups: Group vectors of the WLSP database

    Returns:
        Feature name to value; None where a side has no vector
    """
    out: dict[str, float | None] = dict.fromkeys(FEATURES)
    sub = groups.neighbour_matrix(mid, "subpara")
    para = groups.neighbour_matrix(mid, "para")
    if word_vec is not None:
        if sub is not None:
            sims = sub @ word_vec
            out["chive_sub_max"], out["chive_sub_mean"] = float(sims.max()), float(sims.mean())
        if para is not None:
            sims = para @ word_vec
            out["chive_para_max"], out["chive_para_mean"] = float(sims.max()), float(sims.mean())
    syn = centroid(syn_vecs)
    if syn is not None:
        if sub is not None:
            out["chive_syn_sub_max"] = float((sub @ np.vstack(syn_vecs).T).max())
        for level, name in (
            ("subpara", "chive_syn_sub"),
            ("para", "chive_syn_para"),
            ("item", "chive_syn_item"),
        ):
            c = groups.group_centroid(mid, level)
            if c is not None:
                out[name] = float(c @ syn)
    return out


def add_features(df: pl.DataFrame, ja: wn.Wordnet, db: Wlsp, vectors: Vectors) -> pl.DataFrame:
    """Compute chiVe features for every row of the candidate table.

    Args:
        df: Candidate table from ``candidates.py``
        ja: The Japanese WordNet
        db: The indexed WLSP database
        vectors: Vector lookup

    Returns:
        The table with the ``chive_*`` columns replaced
    """
    t0 = time.time()
    groups = GroupVectors(db, vectors)
    logger.info(
        "WLSP vectors: %d senses with a vector of %d (%.0fs)",
        sum(v is not None for v in groups.sense_vec.values()),
        len(groups.sense_vec),
        time.time() - t0,
    )
    synset_vecs: dict[str, dict[str, np.ndarray]] = {}
    for ss in ja.synsets():
        d: dict[str, np.ndarray] = {}
        for w in ss.words():
            v = vectors.get(w.lemma())
            if v is not None:
                d[w.lemma()] = v
        synset_vecs[ss.id] = d
    logger.info(
        "wnja vectors: direct %d, split %d, missing %d (%.0fs)",
        vectors.n_direct,
        vectors.n_split,
        vectors.n_missing,
        time.time() - t0,
    )
    feats: dict[str, list[float | None]] = {f: [] for f in FEATURES}
    has: list[int] = []
    for lemma, synset_id, mid in df.select("lemma", "synset_id", "midashi_no").iter_rows():
        word_vec = vectors.get(lemma)
        others = [v for k, v in synset_vecs[synset_id].items() if k != lemma]
        row = pair_features(word_vec, others, mid, groups)
        for f in FEATURES:
            feats[f].append(row[f])
        has.append(int(word_vec is not None))
    logger.info("features for %d pairs (%.0fs)", len(df), time.time() - t0)
    df = df.drop([c for c in df.columns if c.startswith("chive_")])
    return df.with_columns(
        *[pl.Series(f, feats[f], dtype=pl.Float32) for f in FEATURES],
        pl.Series("chive_has", has, dtype=pl.Int8),
    )


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidates", type=Path, nargs="?", default=Path("build/candidates.parquet"))
    ap.add_argument(
        "--kv", type=Path, default=Path("build/chive/chive-1.3-mc30_gensim/chive-1.3-mc30.kv")
    )
    ap.add_argument("--wlsp", type=Path, default=Path("data/WLSP/bunruidb.txt"))
    ap.add_argument("--wn-data", type=Path, default=Path(".wn_data"))
    ap.add_argument("--lexicon", default="wnja:2.1")
    ap.add_argument("--no-sudachi", action="store_true", help="disable the OOV fallback")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kv = KeyedVectors.load(str(args.kv))
    logger.info("loaded %s: %d words, %d dims", args.kv, len(kv), kv.vector_size)
    tokenizer = None if args.no_sudachi else Dictionary(dict="core").create()
    vectors = Vectors(kv, tokenizer)
    wn.config.data_directory = args.wn_data.resolve()
    ja = wn.Wordnet(lexicon=args.lexicon)
    db = Wlsp.load(args.wlsp)

    df = add_features(pl.read_parquet(args.candidates), ja, db, vectors)
    df.write_parquet(args.candidates)
    logger.info("wrote %s with %d chive columns", args.candidates, len(FEATURES) + 1)


if __name__ == "__main__":
    main()
