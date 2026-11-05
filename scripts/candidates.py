#!/usr/bin/env python3
"""Generate (wnja sense, WLSP sense) candidate pairs with structural features.

For every word in the Japanese WordNet, candidate WLSP senses are those whose
written form matches the word's lemma or one of its forms; kana-only forms
fall back to matching the WLSP reading.  Each sense of the word is paired with
each candidate, and features describe how well the synset's neighbourhood in
wordnet (other members, hypernyms, hyponyms, co-hyponyms, similar and antonym
synsets) overlaps with the candidate's neighbourhood in WLSP (小段落, 段落,
分類項目, 中項目).  Hierarchy comes from the English wordnet through ILI.

Rows for senses that appear in the dev set carry a ``gold`` label (1 for the
gold WLSP sense, 0 for the other candidates); all other rows have a null label.

Usage::

    uv run scripts/candidates.py -o build/candidates.parquet
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import wn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wlsp import Wlsp, WlspEntry, fold

logger = logging.getLogger(__name__)

LEVELS = ("subpara", "para", "item", "middle")
RELATIONS = ("syn", "hyper", "hypo", "sib", "sim", "ant")
# Which (relation, level) overlap counts to compute.
OVERLAPS = (
    ("syn", "subpara"),
    ("syn", "para"),
    ("syn", "item"),
    ("hyper", "para"),
    ("hyper", "item"),
    ("hyper", "middle"),
    ("hypo", "para"),
    ("hypo", "item"),
    ("hypo", "middle"),
    ("sib", "para"),
    ("sib", "item"),
    ("sim", "subpara"),
    ("sim", "para"),
    ("sim", "item"),
    ("ant", "para"),
    ("ant", "item"),
)
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


def is_kana(text: str) -> bool:
    """Return True if ``text`` consists only of kana and the long-vowel mark.

    Args:
        text: Any string

    Returns:
        Whether every character is hiragana, katakana or ー
    """
    return bool(text) and all("ぁ" <= c <= "ゟ" or "゠" <= c <= "ヿ" or c == "ー" for c in text)


@dataclass
class SynsetContext:
    """Folded written forms of a synset's members and of its related synsets."""

    forms: set[str] = field(default_factory=set)
    related: dict[str, set[str]] = field(default_factory=lambda: {r: set() for r in RELATIONS})


def word_forms(word: wn.Word) -> set[str]:
    """Folded lemma and non-kana written forms of a wnja word.

    Args:
        word: A wnja word

    Returns:
        Set of folded forms; kana-only variants other than the lemma are left
        out, since they are readings rather than spellings
    """
    lemma = word.lemma()
    out = {fold(lemma)}
    out.update(fold(f) for f in word.forms() if not is_kana(f) or is_kana(lemma))
    return out


def kana_forms(word: wn.Word) -> set[str]:
    """Kana-only forms of a wnja word, folded to hiragana (its readings).

    Args:
        word: A wnja word

    Returns:
        Set of hiragana strings, possibly empty
    """
    return {fold(f) for f in [word.lemma(), *word.forms()] if is_kana(f)}


def synset_contexts(ja: wn.Wordnet, en: wn.Wordnet) -> dict[str, SynsetContext]:
    """Collect member forms and related-synset forms for every wnja synset.

    Hypernyms, hyponyms and co-hyponyms are taken from the English wordnet
    (through ILI) and mapped back to wnja synsets; similar, also and antonym
    links come from wnja itself.

    Args:
        ja: The Japanese WordNet
        en: The English wordnet sharing ILIs with it

    Returns:
        Mapping from wnja synset ID to its context
    """
    contexts: dict[str, SynsetContext] = {}
    by_ili: dict[str, str] = {}
    for ss in ja.synsets():
        ctx = SynsetContext()
        for w in ss.words():
            ctx.forms |= word_forms(w)
        contexts[ss.id] = ctx
        if ss.ili:
            by_ili[ss.ili] = ss.id

    def ja_forms(en_synsets: list[wn.Synset]) -> set[str]:
        out: set[str] = set()
        for es in en_synsets:
            jid = by_ili.get(es.ili or "")
            if jid:
                out |= contexts[jid].forms
        return out

    for ss in ja.synsets():
        ctx = contexts[ss.id]
        for rel, names in (("sim", ("similar", "also")), ("ant", ("antonym",))):
            for name in names:
                for other in ss.get_related(name):
                    ctx.related[rel] |= contexts[other.id].forms
        if not ss.ili:
            continue
        for es in en.synsets(ili=ss.ili):
            hypers = es.hypernyms() + es.get_related("instance_hypernym")
            ctx.related["hyper"] |= ja_forms(hypers)
            ctx.related["hypo"] |= ja_forms(es.hyponyms())
            for h in hypers:
                ctx.related["sib"] |= ja_forms([s for s in h.hyponyms() if s.id != es.id])
        for rel in ("hyper", "hypo", "sib"):
            ctx.related[rel] -= ctx.forms
    return contexts


def find_candidates(word: wn.Word, db: Wlsp) -> dict[str, str]:
    """Candidate WLSP senses for a word, with how each was matched.

    Args:
        word: A wnja word
        db: The indexed WLSP database

    Returns:
        Mapping from 見出し番号 to match type: ``lemma``, ``form``, ``suru``
        (verb lemma with する stripped) or ``yomi`` (reading only)
    """
    found: dict[str, str] = {}
    lemma = fold(word.lemma())
    for mid in db.by_form.get(lemma, []):
        found[mid] = "lemma"
    for f in word_forms(word) - {lemma}:
        for mid in db.by_form.get(f, []):
            found.setdefault(mid, "form")
    if word.pos == "v" and lemma.endswith("する"):
        for mid in db.by_form.get(lemma[:-2], []):
            found.setdefault(mid, "suru")
    if not found:
        for y in kana_forms(word):
            for mid in db.by_yomi.get(y, []):
                found.setdefault(mid, "yomi")
    return found


def pair_features(ctx: SynsetContext, own: set[str], rec: WlspEntry, db: Wlsp) -> dict[str, int]:
    """Overlap counts between a synset's neighbourhood and a WLSP sense's groups.

    Args:
        ctx: Context of the synset
        own: Folded forms of the word being linked (excluded from member overlap)
        rec: Canonical WLSP entry of the candidate sense
        db: The indexed WLSP database

    Returns:
        Feature name to count: ``ov_<relation>_<level>``, ``n_<relation>`` sizes
        and ``size_<level>`` group sizes
    """
    groups = {level: db.group_forms(rec, level) for level in LEVELS}
    sets = dict(ctx.related)
    sets["syn"] = ctx.forms - own
    out: dict[str, int] = {}
    for rel, level in OVERLAPS:
        out[f"ov_{rel}_{level}"] = len(sets[rel] & groups[level])
    for rel in RELATIONS:
        out[f"n_{rel}"] = len(sets[rel])
    for level in LEVELS:
        out[f"size_{level}"] = len(db.groups[level][db.group_key(rec, level)])
    return out


def read_gold(path: Path) -> dict[str, set[str]]:
    """Gold WLSP senses per wnja sense from the dev set.

    Args:
        path: Path to ``dev/devset.tsv``

    Returns:
        Mapping from wnja sense ID to the set of gold 見出し番号s
    """
    gold: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["wnja_sense"] and row["status"] != "hyponym":
                gold[row["wnja_sense"]].add(row["wlsp_midashi_no"])
    return dict(gold)


def build(ja: wn.Wordnet, en: wn.Wordnet, db: Wlsp, gold: dict[str, set[str]]) -> pl.DataFrame:
    """Build the candidate table.

    Args:
        ja: The Japanese WordNet
        en: The English wordnet sharing ILIs with it
        db: The indexed WLSP database
        gold: Gold 見出し番号s per wnja sense (dev set)

    Returns:
        One row per (wnja sense, WLSP sense) candidate pair
    """
    t0 = time.time()
    contexts = synset_contexts(ja, en)
    logger.info("synset contexts for %d synsets in %.0fs", len(contexts), time.time() - t0)
    rows: list[dict[str, object]] = []
    n_words = n_matched = 0
    for word in ja.words():
        n_words += 1
        cands = find_candidates(word, db)
        if not cands:
            continue
        n_matched += 1
        own = word_forms(word)
        senses = word.senses()
        for i, sense in enumerate(senses):
            ss = sense.synset()
            ctx = contexts[ss.id]
            for mid, how in cands.items():
                rec = db.senses[mid]
                row: dict[str, object] = {
                    "sense_id": sense.id,
                    "synset_id": ss.id,
                    "ili": ss.ili or "",
                    "lemma": word.lemma(),
                    "pos": ss.pos,
                    "midashi_no": mid,
                    "record_id": rec.record_id,
                    "code": rec.code,
                    "labels": rec.labels,
                    "wlsp_lemma": rec.lemma,
                    "pos_wlsp": rec.pos,
                    "lex": how,
                    "pos_ok": int(ss.pos == rec.pos),
                    "n_cands": len(cands),
                    "n_senses": len(senses),
                    "sense_rank": i,
                    "go_no": int(rec.go_no),
                }
                row.update(pair_features(ctx, own, rec, db))
                if sense.id in gold:
                    row["gold"] = int(mid in gold[sense.id])
                else:
                    row["gold"] = None
                rows.append(row)
    logger.info(
        "%d of %d words matched; %d candidate pairs in %.0fs",
        n_matched,
        n_words,
        len(rows),
        time.time() - t0,
    )
    return pl.DataFrame(rows).with_columns(pl.col("gold").cast(pl.Int8))


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wlsp", type=Path, default=Path("data/WLSP/bunruidb.txt"))
    ap.add_argument("--wn-data", type=Path, default=Path(".wn_data"))
    ap.add_argument("--lexicon", default="wnja:2.1")
    ap.add_argument("--en-lexicon", default="omw-en:1.4")
    ap.add_argument("--dev", type=Path, default=Path("dev/devset.tsv"))
    ap.add_argument("-o", "--output", type=Path, default=Path("build/candidates.parquet"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wn.config.data_directory = args.wn_data.resolve()
    ja = wn.Wordnet(lexicon=args.lexicon)
    en = wn.Wordnet(lexicon=args.en_lexicon)
    db = Wlsp.load(args.wlsp)
    gold = read_gold(args.dev) if args.dev.exists() else {}

    df = build(ja, en, db, gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    labelled = df.filter(pl.col("gold").is_not_null())
    logger.info(
        "wrote %s: %d rows, %d labelled (%d positive); gold senses with a candidate: %d of %d",
        args.output,
        len(df),
        len(labelled),
        int(labelled["gold"].sum()),
        labelled["sense_id"].n_unique(),
        len(gold),
    )


if __name__ == "__main__":
    main()
