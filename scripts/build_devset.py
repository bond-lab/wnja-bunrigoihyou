#!/usr/bin/env python3
"""Build the sense-level dev set linking WLSP entries to Japanese WordNet senses.

The TUFS basic vocabulary (Bond et al. 2020) uses the 分類語彙表 (WLSP) record
ID as its concept ID, and ``tufs-omw-map.tsv`` links those concepts to
Princeton WordNet 3.0 synsets.  Composing the two gives, for each Japanese
basic-vocabulary item, a WLSP record (a sense) and a wordnet synset.  This
script looks the WLSP lemma up in that synset in the Japanese WordNet (wnja)
to get a wnja sense ID, and writes one row per (WLSP record, synset) pair.

Usage::

    uv run scripts/build_devset.py --tufs ../tufs --wlsp data/WLSP/bunruidb.txt \
        --wn-data .wn_data -o dev/devset.tsv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import wn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wlsp import WlspEntry, hira_to_kata, read_wlsp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TufsEntry:
    """Japanese lemma and reading of a TUFS concept."""

    cid: str
    lemma: str
    yomi: str
    english: str


_INDEX = re.compile(r"\(\d+\)$")


def polite_form(tufs_form: str) -> str:
    """Return the TUFS polite (ます) form of a verb, or an empty string.

    TUFS lists verbs in the polite form (会います) while WLSP lists the
    dictionary form (会う).  A trailing disambiguation index such as
    ``(2)`` is removed; only the first ``;``-separated variant is used.

    Args:
        tufs_form: Lemma string from ``tufs-vocab.tsv``

    Returns:
        The polite form, e.g. ``会います``, or ``""`` if the form is not a ます form
    """
    form = _INDEX.sub("", tufs_form.split(";")[0].strip())
    return form if form.endswith("ます") else ""


def _yomi_from_comment(comment: str) -> str:
    """Extract the reading from a TUFS Japanese comment field."""
    m = re.search(r"【よみ】;\s*([^;【]+)", comment)
    return m.group(1).strip() if m else ""


def read_tufs(path: Path) -> dict[str, TufsEntry]:
    """Read the Japanese and English rows of ``tufs-vocab.tsv`` keyed by concept ID.

    Concepts with no Japanese row get an empty lemma and reading.

    Args:
        path: Path to ``tufs-vocab.tsv``

    Returns:
        Mapping from TUFS concept ID (= WLSP record ID without padding) to entry
    """
    english: dict[str, str] = {}
    japanese: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["cid"] == "None":
                continue
            if row["lang"] == "en":
                english[row["cid"]] = row["lemma"]
            elif row["lang"] == "ja":
                japanese[row["cid"]] = (row["lemma"], _yomi_from_comment(row["comment"]))
    return {
        cid: TufsEntry(cid, *japanese.get(cid, ("", "")), english.get(cid, ""))
        for cid in english.keys() | japanese.keys()
    }


def read_omw_map(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Read ``tufs-omw-map.tsv`` (cid, relation, PWN 3.0 synset).

    Args:
        path: Path to the mapping file

    Returns:
        Mapping from concept ID to list of (relation, synset) pairs
    """
    links: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as fh:
        for cid, rel, synset in csv.reader(fh, delimiter="\t"):
            links[cid].append((rel, synset))
    return links


def find_sense(synset: wn.Synset, candidates: list[tuple[str, str]]) -> tuple[str, str, str]:
    """Find the wnja sense in ``synset`` whose lemma or form matches a candidate.

    Args:
        synset: A wnja synset
        candidates: ``(method, string)`` pairs tried in order

    Returns:
        ``(sense_id, matched_lemma, method)``; empty strings if nothing matched
    """
    senses = synset.senses()
    for method, cand in candidates:
        if not cand:
            continue
        for sense in senses:
            word = sense.word()
            forms = {word.lemma(), *word.forms()}
            if cand in forms or hira_to_kata(cand) in forms:
                return sense.id, word.lemma(), method
    return "", "", ""


def row_status(relation: str, synset_found: bool, sense_found: bool) -> str:
    """Classify a dev-set row by how far the wordnet side could be resolved.

    Args:
        relation: Link type from ``tufs-omw-map.tsv`` (``synonym``, ``hyponym``, ``multi``)
        synset_found: Whether the synset exists in wnja
        sense_found: Whether a wnja sense with the WLSP lemma was found in it

    Returns:
        ``sense`` (sense-level link), ``synset-only`` (lemma missing from the wnja
        synset), ``no-synset`` (synset absent from wnja) or ``hyponym`` (the WLSP
        sense is narrower than the synset, so no sense is expected)
    """
    if relation == "hyponym":
        return "hyponym"
    if not synset_found:
        return "no-synset"
    return "sense" if sense_found else "synset-only"


def tufs_sense_id(tufs_wn: wn.Wordnet | None, pwn: str, lemma: str) -> str:
    """Find the sense of ``lemma`` in synset ``pwn`` in the TUFS Japanese wordnet.

    Args:
        tufs_wn: The TUFS Japanese wordnet, or None to skip the lookup
        pwn: Princeton WordNet 3.0 synset ID, e.g. ``02428924-v``
        lemma: TUFS Japanese lemma (possibly ``;``-separated variants)

    Returns:
        The TUFS sense ID, or an empty string if the synset or lemma is absent
    """
    if tufs_wn is None or not lemma:
        return ""
    try:
        synset = tufs_wn.synset(f"tufs-ja-{pwn}")
    except wn.Error:
        return ""
    variants = {v.strip() for v in lemma.split(";")}
    for sense in synset.senses():
        if sense.word().lemma() in variants:
            return sense.id
    return ""


def build_rows(
    wlsp: dict[str, WlspEntry],
    tufs: dict[str, TufsEntry],
    links: dict[str, list[tuple[str, str]]],
    wnja: wn.Wordnet,
    tufs_wn: wn.Wordnet | None = None,
) -> list[dict[str, str]]:
    """Compose WLSP records, TUFS entries and synset links into dev-set rows.

    Args:
        wlsp: WLSP entries by padded record ID
        tufs: TUFS Japanese entries by concept ID
        links: Synset links by concept ID
        wnja: The Japanese WordNet
        tufs_wn: The TUFS Japanese wordnet, used to fill ``tufs_sense``

    Returns:
        One row per (concept, synset) pair, sorted by concept ID
    """
    rows: list[dict[str, str]] = []
    for cid in sorted(links, key=int):
        record = wlsp.get(cid.zfill(6))
        if record is None:
            logger.warning("cid %s has a synset link but is not a WLSP record", cid)
            continue
        entry = tufs.get(cid, TufsEntry(cid, "", "", ""))
        candidates = [
            ("wlsp_lemma", record.lemma),
            ("tufs_lemma", entry.lemma.split(";")[0].strip()),
            ("wlsp_yomi", record.yomi),
            ("tufs_yomi", entry.yomi),
        ]
        for rel, pwn in links[cid]:
            synset_id = f"wnja-{pwn}"
            try:
                synset = wnja.synset(synset_id)
            except wn.Error:
                logger.warning("%s not in wnja (cid %s)", synset_id, cid)
                synset = None
            sense_id, matched, method = find_sense(synset, candidates) if synset else ("", "", "")
            status = row_status(rel, synset is not None, bool(sense_id))
            rows.append(
                {
                    "cid": cid,
                    "wlsp_record": record.record_id,
                    "wlsp_midashi_no": record.midashi_no,
                    "wlsp_code": record.code,
                    "wlsp_labels": record.labels,
                    "wlsp_midashi": record.midashi,
                    "lemma": record.lemma,
                    "polite": polite_form(entry.lemma),
                    "wlsp_yomi": record.yomi,
                    "tufs_form": entry.lemma,
                    "tufs_en": entry.english,
                    "relation": rel,
                    "pwn30": pwn,
                    "wnja_synset": synset_id if synset else "",
                    "ili": (synset.ili or "") if synset else "",
                    "wnja_sense": sense_id,
                    "wnja_lemma": matched,
                    "match": method,
                    "status": status,
                    "tufs_sense": tufs_sense_id(tufs_wn, pwn, entry.lemma),
                    "wnja_definition": (synset.definition() or "") if synset else "",
                }
            )
    return rows


def write_tsv(rows: list[dict[str, str]], path: Path) -> None:
    """Write rows as a UTF-8 TSV with a header.

    Args:
        rows: Dicts sharing the same keys
        path: Output path (parent directory must exist)
    """
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--tufs",
        type=Path,
        default=Path("../tufs"),
        help="TUFS wordnet repo (tufs-vocab.tsv, tufs-omw-map.tsv)",
    )
    ap.add_argument("--wlsp", type=Path, default=Path("data/WLSP/bunruidb.txt"))
    ap.add_argument("--wn-data", type=Path, default=Path(".wn_data"))
    ap.add_argument("--lexicon", default="wnja:2.1")
    ap.add_argument(
        "--tufs-lexicon",
        default="tufs-ja",
        help="TUFS Japanese wordnet lexicon in the wn database; '' to skip",
    )
    ap.add_argument("-o", "--output", type=Path, default=Path("dev/devset.tsv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wn.config.data_directory = args.wn_data.resolve()
    wnja = wn.Wordnet(lexicon=args.lexicon)
    tufs_wn = wn.Wordnet(lexicon=args.tufs_lexicon) if args.tufs_lexicon else None
    wlsp = read_wlsp(args.wlsp)
    tufs = read_tufs(args.tufs / "tufs-vocab.tsv")
    links = read_omw_map(args.tufs / "tufs-omw-map.tsv")

    rows = build_rows(wlsp, tufs, links, wnja, tufs_wn)
    write_tsv(rows, args.output)

    by_status = Counter(r["status"] for r in rows)
    by_rel = Counter(r["relation"] for r in rows)
    logger.info(
        "wrote %d rows for %d concepts to %s", len(rows), len({r["cid"] for r in rows}), args.output
    )
    logger.info("relations: %s", dict(by_rel))
    logger.info("status: %s", dict(by_status))
    logger.info(
        "rows with a TUFS sense: %d; with neither wnja nor TUFS sense: %d",
        sum(1 for r in rows if r["tufs_sense"]),
        sum(1 for r in rows if not r["tufs_sense"] and not r["wnja_sense"]),
    )
    logger.info(
        "concepts with a WLSP record but no synset link: %d",
        sum(1 for cid in tufs if cid not in links and tufs[cid].lemma),
    )


if __name__ == "__main__":
    main()
