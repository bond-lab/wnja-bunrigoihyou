#!/usr/bin/env python3
"""Add sentence-embedding similarity features to the candidate table (tier 2).

Each wnja synset is rendered as text (Japanese definition, members, examples,
optionally the English gloss from omw-en through ILI) and each WLSP sense as
three texts of increasing specificity:

- ``cat``: the category path (類/部門/中項目/分類項目 labels, plus the English
  中項目 name from the TUFS ``bunrui_names.tsv`` when available)
- ``sub``: the headword with its 小段落 neighbours
- ``ctx``: the category path with 小段落 and 段落 neighbours

Both sides are embedded with a sentence-transformers model and the cosine
between the synset and each WLSP text becomes a feature, named
``sent_<tag>_cat``, ``sent_<tag>_sub`` and ``sent_<tag>_ctx`` where ``<tag>``
is derived from the model name, so several models can live side by side in
the table.  Embeddings are cached under ``build/emb/<tag>/`` and reused.

Usage::

    uv run scripts/features_sent.py --model BAAI/bge-m3 --device mps
    uv run scripts/features_sent.py --model intfloat/multilingual-e5-small \\
        --query-prefix "query: " --doc-prefix "passage: " --max-texts 500   # smoke test

Note on TUFS glosses: ``--tufs-gloss`` adds the TUFS English gloss to the
~800 WLSP records that are TUFS concepts.  Those records are the dev set, so
the feature leaks into the evaluation; use it for the final mapping only.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import wn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wlsp import Wlsp, WlspEntry

logger = logging.getLogger(__name__)

VARIANTS = ("cat", "sub", "ctx")


def model_tag(model: str) -> str:
    """Short column-safe tag for a model name.

    Args:
        model: Hugging Face model ID or path, e.g. ``BAAI/bge-m3``

    Returns:
        The last path component in lower case with non-alphanumerics as ``_``
    """
    return re.sub(r"[^a-z0-9]+", "_", model.rstrip("/").split("/")[-1].lower()).strip("_")


def read_category_names(path: Path | None) -> dict[str, str]:
    """English names of WLSP 中項目 codes from the TUFS ``bunrui_names.tsv``.

    Args:
        path: TSV with columns ``Code``, ``en``, ... (``1.10``, ``Matters``), or None

    Returns:
        Mapping from code (``1.10``) to English name; empty if no file
    """
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["Code"]: row["en"] for row in csv.DictReader(fh, delimiter="\t")}


def read_tufs_glosses(path: Path | None) -> dict[str, str]:
    """English glosses of WLSP records from the TUFS vocabulary.

    Takes the English lemma of each concept and the ``【意味】`` field of its
    Japanese row (both keyed by the TUFS concept ID, which is the WLSP record
    ID without zero padding).

    Args:
        path: ``tufs-vocab.tsv`` or None

    Returns:
        Mapping from zero-padded WLSP record ID to gloss text
    """
    if path is None or not path.exists():
        return {}
    glosses: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["cid"] == "None":
                continue
            rid = row["cid"].zfill(6)
            if row["lang"] == "en":
                glosses.setdefault(rid, set()).add(row["lemma"])
            elif row["lang"] == "ja":
                m = re.search(r"【意味】;\s*([^【]+)", row["comment"])
                if m:
                    glosses.setdefault(rid, set()).update(
                        g.strip() for g in m.group(1).split(";") if g.strip()
                    )
    return {rid: ", ".join(sorted(g)) for rid, g in glosses.items() if g}


def synset_text(ss: wn.Synset, en_gloss: str = "") -> str:
    """Render a wnja synset as one line of text.

    Args:
        ss: A wnja synset
        en_gloss: English definition to append, or empty

    Returns:
        Members, definition(s) and examples, e.g. ``会う, 逢う: 人と顔を合わせる。 例: ...``
    """
    lemmas = "、".join(w.lemma() for w in ss.words())
    defs = " ".join(ss.definitions())
    examples = " ".join(ss.examples()[:2])
    text = f"{lemmas}: {defs}"
    if examples:
        text += f" 例: {examples}"
    if en_gloss:
        text += f" ({en_gloss})"
    return text


def wlsp_texts(
    rec: WlspEntry,
    db: Wlsp,
    cat_names: dict[str, str],
    gloss: str = "",
    para_limit: int = 20,
) -> dict[str, str]:
    """Render a WLSP sense as the ``cat``, ``sub`` and ``ctx`` texts.

    Args:
        rec: Canonical entry of the sense
        db: The indexed WLSP database
        cat_names: English 中項目 names by code
        gloss: TUFS English gloss to append to ``sub`` and ``ctx``, or empty
        para_limit: Maximum 段落 neighbours in ``ctx``

    Returns:
        Mapping from variant name to text
    """
    labels = rec.labels.replace("/", " ")
    en = cat_names.get(rec.middle_key)
    cat = f"{labels} ({en})" if en else labels
    sub_n = "、".join(db.neighbours(rec, "subpara"))
    para_n = "、".join(db.neighbours(rec, "para", para_limit))
    sub = f"{rec.lemma}: {sub_n}" if sub_n else rec.lemma
    ctx = f"{cat}: {rec.lemma}; {sub_n}; {para_n}"
    if gloss:
        sub += f" ({gloss})"
        ctx += f" ({gloss})"
    return {"cat": cat, "sub": sub, "ctx": ctx}


class Embedder:
    """Sentence-transformers wrapper with an on-disk cache per text."""

    def __init__(
        self,
        model: str,
        cache_dir: Path,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        """Load the model and any cached embeddings.

        Args:
            model: Hugging Face model ID or local path
            cache_dir: Directory for ``cache.npz`` (texts and vectors)
            device: ``cuda``, ``mps``, ``cpu`` or None for auto-detection
            batch_size: Encoding batch size
            max_length: Token limit per text
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device, trust_remote_code=True)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size
        self.cache_path = cache_dir / "cache.npz"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            z = np.load(self.cache_path, allow_pickle=False)
            self.cache = dict(zip(z["texts"].tolist(), z["vectors"], strict=True))
            logger.info("loaded %d cached embeddings from %s", len(self.cache), self.cache_path)

    def encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Embed texts (with a prefix for the model), using and filling the cache.

        Args:
            texts: Texts to embed, in order
            prefix: Instruction or role prefix the model expects (``query: ``)

        Returns:
            A (len(texts), dim) float32 array of unit vectors
        """
        keyed = [prefix + t for t in texts]
        todo = sorted({k for k in keyed if k not in self.cache})
        if todo:
            t0 = time.time()
            vecs = self.model.encode(
                todo,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            self.cache.update(zip(todo, vecs, strict=True))
            logger.info("embedded %d new texts in %.0fs", len(todo), time.time() - t0)
        return np.vstack([self.cache[k] for k in keyed])

    def save(self) -> None:
        """Write the cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        texts = np.array(list(self.cache), dtype=object)
        vectors = np.vstack(list(self.cache.values())) if self.cache else np.zeros((0, 0))
        np.savez(self.cache_path, texts=texts.astype(str), vectors=vectors)


def add_features(
    df: pl.DataFrame,
    ja: wn.Wordnet,
    db: Wlsp,
    embedder: Embedder,
    tag: str,
    texts_for: dict[str, object],
    prefixes: tuple[str, str],
    max_texts: int = 0,
) -> pl.DataFrame:
    """Compute the ``sent_<tag>_*`` cosine features for the candidate table.

    Args:
        df: Candidate table
        ja: The Japanese WordNet
        db: The indexed WLSP database
        embedder: Model wrapper
        tag: Column tag for this model
        texts_for: Options: ``cat_names`` (dict), ``glosses`` (dict),
            ``en_gloss`` (dict synset_id -> English definition)
        prefixes: ``(query_prefix, doc_prefix)`` for synset and WLSP texts
        max_texts: If > 0, restrict to the first N synsets (smoke test)

    Returns:
        The table with the three feature columns (null for rows outside the subset)
    """
    synset_ids = df["synset_id"].unique().sort().to_list()
    if max_texts:
        synset_ids = synset_ids[:max_texts]
    en_gloss: dict[str, str] = texts_for.get("en_gloss", {})
    s_texts = [synset_text(ja.synset(sid), en_gloss.get(sid, "")) for sid in synset_ids]
    s_vec = embedder.encode(s_texts, prefixes[0])
    s_index = {sid: i for i, sid in enumerate(synset_ids)}

    sub = df.filter(pl.col("synset_id").is_in(synset_ids))
    mids = sub["midashi_no"].unique().sort().to_list()
    per_variant = {v: [] for v in VARIANTS}
    for mid in mids:
        texts = wlsp_texts(
            db.senses[mid],
            db,
            texts_for.get("cat_names", {}),
            texts_for.get("glosses", {}).get(db.senses[mid].record_id, ""),
        )
        for v in VARIANTS:
            per_variant[v].append(texts[v])
    w_vec = {v: embedder.encode(per_variant[v], prefixes[1]) for v in VARIANTS}
    w_index = {mid: i for i, mid in enumerate(mids)}
    embedder.save()

    si = np.array([s_index.get(s, -1) for s in df["synset_id"]])
    wi = np.array([w_index.get(m, -1) for m in df["midashi_no"]])
    ok = (si >= 0) & (wi >= 0)
    cols = []
    for v in VARIANTS:
        vals = np.full(len(df), np.nan, dtype=np.float32)
        vals[ok] = np.einsum("ij,ij->i", s_vec[si[ok]], w_vec[v][wi[ok]])
        cols.append(pl.Series(f"sent_{tag}_{v}", vals, dtype=pl.Float32).fill_nan(None))
    df = df.drop([c for c in df.columns if c.startswith(f"sent_{tag}_")])
    return df.with_columns(cols)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidates", type=Path, nargs="?", default=Path("build/candidates.parquet"))
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--query-prefix", default="", help="prefix for synset texts (e5: 'query: ')")
    ap.add_argument("--doc-prefix", default="", help="prefix for WLSP texts (e5: 'passage: ')")
    ap.add_argument("--en-gloss", action="store_true", help="append the omw-en definition")
    ap.add_argument("--en-lexicon", default="omw-en:1.4")
    ap.add_argument(
        "--category-names", type=Path, default=Path("../tufs/tufsdata/bunrui_names.tsv")
    )
    ap.add_argument("--tufs-gloss", type=Path, default=None, help="tufs-vocab.tsv (leaks into dev)")
    ap.add_argument("--max-texts", type=int, default=0, help="smoke test on the first N synsets")
    ap.add_argument("--wlsp", type=Path, default=Path("data/WLSP/bunruidb.txt"))
    ap.add_argument("--wn-data", type=Path, default=Path(".wn_data"))
    ap.add_argument("--lexicon", default="wnja:2.1")
    ap.add_argument("--emb-dir", type=Path, default=Path("build/emb"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tag = model_tag(args.model)
    wn.config.data_directory = args.wn_data.resolve()
    ja = wn.Wordnet(lexicon=args.lexicon)
    db = Wlsp.load(args.wlsp)
    df = pl.read_parquet(args.candidates)

    texts_for: dict[str, object] = {
        "cat_names": read_category_names(args.category_names),
        "glosses": read_tufs_glosses(args.tufs_gloss),
    }
    if args.en_gloss:
        en = wn.Wordnet(lexicon=args.en_lexicon)
        en_gloss = {}
        for sid in df["synset_id"].unique():
            ss = ja.synset(sid)
            if ss.ili:
                for es in en.synsets(ili=ss.ili):
                    en_gloss[sid] = es.definition() or ""
        texts_for["en_gloss"] = en_gloss
        logger.info("English glosses for %d synsets", len(en_gloss))
    if texts_for["glosses"]:
        logger.warning(
            "TUFS glosses enabled for %d records: dev evaluation will leak",
            len(texts_for["glosses"]),
        )

    embedder = Embedder(
        args.model, args.emb_dir / tag, args.device, args.batch_size, args.max_length
    )
    logger.info("model %s on %s, tag %s", args.model, embedder.model.device, tag)
    df = add_features(
        df, ja, db, embedder, tag, texts_for, (args.query_prefix, args.doc_prefix), args.max_texts
    )
    df.write_parquet(args.candidates)
    done = int(df[f"sent_{tag}_ctx"].is_not_null().sum())
    logger.info("wrote %s: sent_%s_* set on %d of %d rows", args.candidates, tag, done, len(df))


if __name__ == "__main__":
    main()
