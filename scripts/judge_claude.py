#!/usr/bin/env python3
"""Ask Claude to adjudicate uncertain wnja-to-WLSP links, one word at a time.

For each selected word the prompt lists its wnja senses (synset members,
Japanese definition, examples, optional English gloss) and its candidate WLSP
senses (code, category labels, 小段落 and 段落 neighbours, and the 岩波国語辞典
definition when ``build/iwanami_defs.tsv`` is present).  Claude returns, as
JSON, the links it judges to hold with a relation (``same``, ``narrower``,
``broader``, ``related``) and a confidence.  Verdicts are cached in
``build/judge.jsonl`` keyed by word and prompt hash, and ``--apply`` turns
them into ``judge_same``, ``judge_rel`` and ``judge_conf`` columns of the
candidate table for the scorer.

Words are selected when any of their pairs has a scorer probability in the
uncertain band, or when the top two candidates of a sense are close.

Modes (each a separate run)::

    uv run scripts/judge_claude.py --dry-run           # write prompts, estimate size
    uv run scripts/judge_claude.py --sync --max-words 20   # small direct run
    uv run scripts/judge_claude.py --submit            # Message Batches (50% price)
    uv run scripts/judge_claude.py --collect           # fetch finished batches
    uv run scripts/judge_claude.py --apply             # add judge_* columns

Requires the ``judge`` extra (``uv sync --extra judge``) and an API key in
``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import wn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wlsp import Wlsp

logger = logging.getLogger(__name__)

RELATIONS = ("same", "narrower", "broader", "related")

SYSTEM = """\
You are a Japanese lexicographer aligning two sense inventories: the Japanese \
WordNet (synsets with Japanese members, a definition and examples) and the \
分類語彙表 (WLSP), a thesaurus where each sense of a headword is a position in \
a category, shown here by its category path and the neighbouring headwords in \
the same 小段落 (closest) and 段落 (related).

For one headword you are given all of its WordNet senses and all of its WLSP \
senses. Decide which pairs denote the same sense. Report every pair that holds \
with a relation: "same" (same meaning), "narrower" (the WordNet sense is more \
specific than the WLSP sense), "broader" (more general), or "related" (clearly \
connected but neither covers the other). Leave out pairs that do not hold. A \
WordNet sense may match several WLSP senses and vice versa; WLSP often lists \
the same headword more than once in nearby positions, and such near-duplicates \
should all be linked. Give a confidence between 0 and 1 for each link. Be \
precise about meaning: the category path and neighbours define the WLSP sense, \
not the headword alone.\
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "wordnet": {"type": "string", "description": "WordNet sense label, e.g. W1"},
                    "wlsp": {"type": "string", "description": "WLSP sense label, e.g. B2"},
                    "relation": {"type": "string", "enum": list(RELATIONS)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["wordnet", "wlsp", "relation", "confidence"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string", "description": "optional short remark"},
    },
    "required": ["links", "note"],
    "additionalProperties": False,
}


@dataclass
class WordCase:
    """Everything the judge sees for one headword."""

    lemma: str
    senses: list[dict[str, str]]  # label, sense_id, text
    wlsp: list[dict[str, str]]  # label, midashi_no, text
    prompt: str = ""
    key: str = ""
    pairs: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)

    def finish(self) -> None:
        """Render the prompt and compute the cache key and label maps."""
        lines = [f"Headword: {self.lemma}", "", "WordNet senses:"]
        lines += [f"{s['label']}. {s['text']}" for s in self.senses]
        lines += ["", "WLSP senses:"]
        lines += [f"{w['label']}. {w['text']}" for w in self.wlsp]
        lines += ["", "Which WordNet senses correspond to which WLSP senses?"]
        self.prompt = "\n".join(lines)
        self.key = hashlib.sha256((SYSTEM + self.prompt).encode("utf-8")).hexdigest()[:16]
        self.pairs = {
            (s["label"], w["label"]): (s["sense_id"], w["midashi_no"])
            for s in self.senses
            for w in self.wlsp
        }


def read_iwanami(defs_path: Path, map_path: Path) -> dict[str, list[str]]:
    """Iwanami definitions per WLSP record ID, via the wlsp2iwanami table.

    Args:
        defs_path: TSV ``senseid<TAB>definition`` produced at TUAT (may be absent)
        map_path: ``WLSP2iwanami.txt`` from NINJAL (Shift_JIS; column 3 holds the
            WLSP record as a quoted CSV string whose first field is the record ID)

    Returns:
        Mapping from zero-padded WLSP record ID to definitions; empty if no defs file
    """
    if not defs_path.exists() or not map_path.exists():
        return {}
    defs: dict[str, str] = {}
    with defs_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 2 and row[0]:
                defs[row[0]] = row[1]
    out: dict[str, list[str]] = defaultdict(list)
    with map_path.open(encoding="shift_jis", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3 or not row[2] or row[0] not in defs:
                continue
            record_id = row[2].strip('"').split(",")[0]
            out[record_id].append(defs[row[0]])
    return dict(out)


def select_words(
    df: pl.DataFrame,
    low: float,
    high: float,
    margin: float,
    single_sample: int = 0,
    seed: int = 0,
) -> list[str]:
    """Words with an uncertain pair or a close top-2 decision, plus a sample of lone links.

    Words whose every wnja sense has exactly one WLSP candidate always score
    high, because the dev set holds no negatives of that kind; a random sample
    of them lets the judge estimate how often such lone links are wrong and
    supplies labels the scorer lacks.

    Args:
        df: Scored candidate table (needs ``p``)
        low: Lower end of the uncertain band
        high: Upper end of the uncertain band
        margin: Top-2 probability gap below which a sense counts as undecided
        single_sample: Number of single-candidate words to add at random
        seed: Random seed for that sample

    Returns:
        Sorted list of lemmas
    """
    band = df.filter((pl.col("p") >= low) & (pl.col("p") <= high))["lemma"]
    ranked = (
        df.sort(["sense_id", "p"], descending=[False, True])
        .with_columns(pl.int_range(pl.len()).over("sense_id").alias("rank"))
        .filter(pl.col("rank") < 2)
        .group_by("sense_id")
        .agg(pl.col("p").max() - pl.col("p").min(), pl.len().alias("n"), pl.col("lemma").first())
        .filter((pl.col("n") == 2) & (pl.col("p") < margin))["lemma"]
    )
    chosen = set(band.to_list()) | set(ranked.to_list())
    if single_sample:
        lone = (
            df.group_by("lemma")
            .agg(pl.col("n_cands").max().alias("m"))
            .filter(pl.col("m") == 1)["lemma"]
            .to_list()
        )
        rng = random.Random(seed)
        chosen |= set(rng.sample(lone, min(single_sample, len(lone))))
    return sorted(chosen)


def build_cases(
    df: pl.DataFrame,
    words: list[str],
    ja: wn.Wordnet,
    db: Wlsp,
    iwanami: dict[str, list[str]],
    en_gloss: dict[str, str],
) -> list[WordCase]:
    """Build one prompt per word from its candidate pairs.

    Args:
        df: Candidate table
        words: Lemmas to include
        ja: The Japanese WordNet
        db: The indexed WLSP database
        iwanami: Iwanami definitions per WLSP record ID (may be empty)
        en_gloss: English definition per synset ID (may be empty)

    Returns:
        Cases in the order of ``words``
    """
    sub = df.filter(pl.col("lemma").is_in(words)).select(
        "lemma", "sense_id", "synset_id", "midashi_no"
    )
    by_word: dict[str, tuple[dict[str, str], set[str]]] = {}
    for lemma, sense_id, synset_id, mid in sub.unique().iter_rows():
        senses, mids = by_word.setdefault(lemma, ({}, set()))
        senses[sense_id] = synset_id
        mids.add(mid)
    cases = []
    for lemma in words:
        if lemma not in by_word:
            continue
        senses, mids = by_word[lemma]
        case = WordCase(lemma=lemma, senses=[], wlsp=[])
        for i, (sense_id, synset_id) in enumerate(sorted(senses.items()), 1):
            ss = ja.synset(synset_id)
            text = f"{'、'.join(w.lemma() for w in ss.words())}: {' '.join(ss.definitions())}"
            if ss.examples():
                text += f" 例: {' / '.join(ss.examples()[:2])}"
            if en_gloss.get(synset_id):
                text += f" (en: {en_gloss[synset_id]})"
            case.senses.append({"label": f"W{i}", "sense_id": sense_id, "text": text})
        for j, mid in enumerate(sorted(mids, key=lambda m: db.senses[m].code), 1):
            rec = db.senses[mid]
            text = f"[{rec.code}] {rec.labels}: {rec.lemma}"
            sub_n = "、".join(db.neighbours(rec, "subpara"))
            para_n = "、".join(db.neighbours(rec, "para", 15))
            if sub_n:
                text += f"; 小段落: {sub_n}"
            if para_n:
                text += f"; 段落: {para_n}"
            if iwanami.get(rec.record_id):
                text += f"; 岩波: {' / '.join(iwanami[rec.record_id])}"
            case.wlsp.append({"label": f"B{j}", "midashi_no": mid, "text": text})
        case.finish()
        cases.append(case)
    return cases


def request_params(case: WordCase, model: str, effort: str, max_tokens: int) -> dict[str, object]:
    """Messages API parameters for one case (shared by sync and batch modes).

    Args:
        case: The word case
        model: Model ID
        effort: ``output_config.effort`` level
        max_tokens: Output token cap

    Returns:
        Keyword arguments for ``messages.create``
    """
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": case.prompt}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }


def parse_verdict(text: str, case: WordCase) -> list[dict[str, object]]:
    """Turn the model's JSON into (sense_id, midashi_no, relation, confidence) rows.

    Args:
        text: JSON text from the response
        case: The case it answers (for the label maps)

    Returns:
        One dict per valid link; unknown labels are dropped
    """
    data = json.loads(text)
    out = []
    for link in data.get("links", []):
        ids = case.pairs.get((link.get("wordnet", ""), link.get("wlsp", "")))
        if ids is None or link.get("relation") not in RELATIONS:
            continue
        out.append(
            {
                "sense_id": ids[0],
                "midashi_no": ids[1],
                "relation": link["relation"],
                "confidence": float(link.get("confidence", 0.0)),
            }
        )
    return out


def load_cache(path: Path) -> dict[str, dict[str, object]]:
    """Read cached verdicts keyed by prompt hash.

    Args:
        path: ``build/judge.jsonl``

    Returns:
        Mapping from key to the stored record
    """
    cache: dict[str, dict[str, object]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                cache[rec["key"]] = rec
    return cache


def append_cache(path: Path, records: list[dict[str, object]]) -> None:
    """Append verdict records to the JSONL cache.

    Args:
        path: ``build/judge.jsonl``
        records: Records with at least ``key``, ``lemma``, ``links``
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def response_text(message: object) -> str:
    """First text block of a Messages API response, or an empty string."""
    return next((b.text for b in message.content if b.type == "text"), "")


def run_sync(cases: list[WordCase], args: argparse.Namespace, cache_path: Path) -> None:
    """Judge cases one request at a time (small runs, testing).

    Args:
        cases: Cases still needing a verdict
        args: Parsed command-line arguments
        cache_path: ``build/judge.jsonl``
    """
    import anthropic

    client = anthropic.Anthropic()
    for case in cases:
        params = request_params(case, args.model, args.effort, args.max_tokens)
        try:
            message = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
            )
        except anthropic.RateLimitError:
            logger.warning("rate limited; sleeping 60s before %s", case.lemma)
            time.sleep(60)
            message = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
            )
        if message.stop_reason == "refusal":
            logger.warning("refused: %s", case.lemma)
            continue
        links = parse_verdict(response_text(message), case)
        append_cache(
            cache_path,
            [{"key": case.key, "lemma": case.lemma, "model": message.model, "links": links}],
        )
        logger.info("%s: %d links (%s)", case.lemma, len(links), message.usage.output_tokens)


def run_submit(cases: list[WordCase], args: argparse.Namespace, batches_path: Path) -> None:
    """Submit cases as Message Batches and record the batch IDs.

    Args:
        cases: Cases still needing a verdict
        args: Parsed command-line arguments
        batches_path: JSON file listing submitted batches and their cases
    """
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    record = json.loads(batches_path.read_text(encoding="utf-8")) if batches_path.exists() else []
    for start in range(0, len(cases), args.batch_size):
        chunk = cases[start : start + args.batch_size]
        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=c.key,
                    params=MessageCreateParamsNonStreaming(
                        **request_params(c, args.model, args.effort, args.max_tokens)
                    ),
                )
                for c in chunk
            ]
        )
        record.append(
            {
                "batch_id": batch.id,
                "model": args.model,
                "cases": {
                    c.key: {"lemma": c.lemma, "pairs": list(map(list, c.pairs.items()))}
                    for c in chunk
                },
                "collected": False,
            }
        )
        logger.info("submitted batch %s with %d cases", batch.id, len(chunk))
    batches_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


def run_collect(args: argparse.Namespace, batches_path: Path, cache_path: Path) -> None:
    """Fetch results of finished batches into the cache.

    Args:
        args: Parsed command-line arguments
        batches_path: JSON file written by ``run_submit``
        cache_path: ``build/judge.jsonl``
    """
    import anthropic

    client = anthropic.Anthropic()
    record = json.loads(batches_path.read_text(encoding="utf-8"))
    for entry in record:
        if entry["collected"]:
            continue
        batch = client.messages.batches.retrieve(entry["batch_id"])
        if batch.processing_status != "ended":
            logger.info("%s still %s", batch.id, batch.processing_status)
            continue
        out = []
        for result in client.messages.batches.results(batch.id):
            info = entry["cases"].get(result.custom_id)
            if info is None or result.result.type != "succeeded":
                logger.warning("%s: %s", result.custom_id, result.result.type)
                continue
            message = result.result.message
            if message.stop_reason == "refusal":
                continue
            case = WordCase(lemma=info["lemma"], senses=[], wlsp=[])
            case.pairs = {tuple(k): tuple(v) for k, v in info["pairs"]}
            links = parse_verdict(response_text(message), case)
            out.append(
                {
                    "key": result.custom_id,
                    "lemma": info["lemma"],
                    "model": message.model,
                    "links": links,
                }
            )
        append_cache(cache_path, out)
        entry["collected"] = True
        logger.info("collected %d verdicts from %s", len(out), batch.id)
    batches_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


def apply_verdicts(
    df: pl.DataFrame, cache: dict[str, dict[str, object]], cases: list[WordCase]
) -> pl.DataFrame:
    """Add ``judge_same``, ``judge_rel`` and ``judge_conf`` columns from cached verdicts.

    Pairs of a judged word that the judge did not list get ``judge_same`` 0 and
    confidence 0; pairs of words never judged stay null.

    Args:
        df: Candidate table
        cache: Verdicts by prompt key
        cases: Cases whose prompts identify the cache entries

    Returns:
        The table with the three columns replaced
    """
    verdict: dict[tuple[str, str], tuple[str, float]] = {}
    judged_words: set[str] = set()
    for case in cases:
        rec = cache.get(case.key)
        if rec is None:
            continue
        judged_words.add(case.lemma)
        for link in rec["links"]:
            verdict[(link["sense_id"], link["midashi_no"])] = (link["relation"], link["confidence"])
    same, rel, conf = [], [], []
    for lemma, sense_id, mid in df.select("lemma", "sense_id", "midashi_no").iter_rows():
        if lemma not in judged_words:
            same.append(None), rel.append(None), conf.append(None)
            continue
        r, c = verdict.get((sense_id, mid), ("none", 0.0))
        same.append(int(r == "same")), rel.append(r), conf.append(c if r != "none" else 0.0)
    df = df.drop([c for c in df.columns if c.startswith("judge_")])
    return df.with_columns(
        pl.Series("judge_same", same, dtype=pl.Int8),
        pl.Series("judge_rel", rel, dtype=pl.Utf8),
        pl.Series("judge_conf", conf, dtype=pl.Float32),
    )


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scored", type=Path, nargs="?", default=Path("build/scored.parquet"))
    ap.add_argument("--candidates", type=Path, default=Path("build/candidates.parquet"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--sync", action="store_true")
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--collect", action="store_true")
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--low", type=float, default=0.3)
    ap.add_argument("--high", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--max-words", type=int, default=0, help="limit the number of words (0 = all)")
    ap.add_argument(
        "--single-sample",
        type=int,
        default=0,
        help="also judge this many random words whose senses have a single candidate",
    )
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--en-gloss", action="store_true", help="add omw-en definitions to synsets")
    ap.add_argument("--en-lexicon", default="omw-en:1.4")
    ap.add_argument("--iwanami-defs", type=Path, default=Path("build/iwanami_defs.tsv"))
    ap.add_argument("--iwanami-map", type=Path, default=Path("data/wlsp2iwanami/WLSP2iwanami.txt"))
    ap.add_argument("--wlsp", type=Path, default=Path("data/WLSP/bunruidb.txt"))
    ap.add_argument("--wn-data", type=Path, default=Path(".wn_data"))
    ap.add_argument("--lexicon", default="wnja:2.1")
    ap.add_argument("--cache", type=Path, default=Path("build/judge.jsonl"))
    ap.add_argument("--batches", type=Path, default=Path("build/judge_batches.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.collect:
        run_collect(args, args.batches, args.cache)
        return

    wn.config.data_directory = args.wn_data.resolve()
    ja = wn.Wordnet(lexicon=args.lexicon)
    db = Wlsp.load(args.wlsp)
    df = pl.read_parquet(args.scored)
    words = select_words(df, args.low, args.high, args.margin, args.single_sample)
    if args.max_words:
        words = words[: args.max_words]
    en_gloss: dict[str, str] = {}
    if args.en_gloss:
        en = wn.Wordnet(lexicon=args.en_lexicon)
        for sid in df.filter(pl.col("lemma").is_in(words))["synset_id"].unique():
            ss = ja.synset(sid)
            for es in en.synsets(ili=ss.ili) if ss.ili else []:
                en_gloss[sid] = es.definition() or ""
    iwanami = read_iwanami(args.iwanami_defs, args.iwanami_map)
    if iwanami:
        logger.info("Iwanami definitions for %d WLSP records", len(iwanami))
    cases = build_cases(df, words, ja, db, iwanami, en_gloss)
    cache = load_cache(args.cache)
    todo = [c for c in cases if c.key not in cache]
    logger.info("%d words selected, %d cases, %d not yet judged", len(words), len(cases), len(todo))

    if args.apply:
        cand = pl.read_parquet(args.candidates)
        cand = apply_verdicts(cand, cache, cases)
        cand.write_parquet(args.candidates)
        logger.info(
            "wrote judge_* columns: %d rows judged", int(cand["judge_same"].is_not_null().sum())
        )
        return
    if args.dry_run:
        out = args.cache.with_name("judge_prompts.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for c in todo:
                fh.write(
                    json.dumps(
                        {"key": c.key, "lemma": c.lemma, "prompt": c.prompt}, ensure_ascii=False
                    )
                    + "\n"
                )
        chars = sum(len(c.prompt) for c in todo)
        logger.info(
            "wrote %d prompts to %s; about %.1fM characters (~%.1fM input tokens plus the system prompt)",
            len(todo),
            out,
            chars / 1e6,
            chars / 1e6 * 0.7,
        )
        if todo:
            print(todo[0].prompt)
        return
    if args.sync:
        run_sync(todo, args, args.cache)
    elif args.submit:
        run_submit(todo, args, args.batches)


if __name__ == "__main__":
    main()
