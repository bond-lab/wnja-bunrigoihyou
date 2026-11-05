"""Tests for the prompt building and verdict parsing in scripts/judge_claude.py."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from judge_claude import WordCase, apply_verdicts, parse_verdict, read_iwanami, select_words


@pytest.fixture
def case() -> WordCase:
    c = WordCase(
        lemma="会う",
        senses=[
            {"label": "W1", "sense_id": "s1", "text": "会う: 人と顔を合わせる"},
            {"label": "W2", "sense_id": "s2", "text": "会う、集まる: 一つの場所に集まる"},
        ],
        wlsp=[
            {"label": "B1", "midashi_no": "m1", "text": "[2.1550-09-01-02] 用/関係/作用: 会う"},
            {"label": "B2", "midashi_no": "m2", "text": "[2.3520-01-01-02] 用/活動/交わり: 会う"},
        ],
    )
    c.finish()
    return c


def test_prompt_and_key(case: WordCase) -> None:
    assert case.prompt.startswith("Headword: 会う")
    assert "W2. 会う、集まる" in case.prompt and "B2. [2.3520" in case.prompt
    assert len(case.key) == 16
    assert case.pairs[("W1", "B2")] == ("s1", "m2")


def test_parse_verdict_drops_unknown_labels(case: WordCase) -> None:
    text = (
        '{"links": [{"wordnet": "W1", "wlsp": "B2", "relation": "same", "confidence": 0.9},'
        ' {"wordnet": "W2", "wlsp": "B1", "relation": "related", "confidence": 0.4},'
        ' {"wordnet": "W9", "wlsp": "B1", "relation": "same", "confidence": 1.0},'
        ' {"wordnet": "W1", "wlsp": "B1", "relation": "maybe", "confidence": 0.5}], "note": ""}'
    )
    links = parse_verdict(text, case)
    assert [(k["sense_id"], k["midashi_no"], k["relation"]) for k in links] == [
        ("s1", "m2", "same"),
        ("s2", "m1", "related"),
    ]


def test_apply_verdicts(case: WordCase) -> None:
    df = pl.DataFrame(
        {
            "lemma": ["会う", "会う", "会う", "会う", "犬"],
            "sense_id": ["s1", "s1", "s2", "s2", "s9"],
            "midashi_no": ["m1", "m2", "m1", "m2", "m9"],
        }
    )
    cache = {
        case.key: {
            "key": case.key,
            "lemma": "会う",
            "links": [
                {"sense_id": "s1", "midashi_no": "m2", "relation": "same", "confidence": 0.9},
                {"sense_id": "s2", "midashi_no": "m1", "relation": "related", "confidence": 0.4},
            ],
        }
    }
    out = apply_verdicts(df, cache, [case])
    assert out["judge_same"].to_list() == [0, 1, 0, 0, None]
    assert out["judge_rel"].to_list() == ["none", "same", "related", "none", None]
    assert out["judge_conf"].to_list()[1] == pytest.approx(0.9)
    assert out["judge_conf"].to_list()[0] == 0.0


def test_select_words_band_and_margin() -> None:
    df = pl.DataFrame(
        {
            "lemma": ["a", "a", "b", "b", "c", "d"],
            "sense_id": ["a1", "a1", "b1", "b1", "c1", "d1"],
            "p": [0.95, 0.1, 0.9, 0.8, 0.5, 0.99],
        }
    )
    # a: confident and far apart; b: close top-2; c: in the band; d: confident
    assert select_words(df, 0.3, 0.7, 0.15) == ["b", "c"]


def test_read_iwanami(tmp_path: Path) -> None:
    defs = tmp_path / "iwanami_defs.tsv"
    defs.write_text("00001.0.1\tある状態に近いさま。\n", encoding="utf-8")
    mapping = tmp_path / "WLSP2iwanami.txt"
    mapping.write_text(
        "senseid\ttype\twlsp\tnote\t\n"
        '00001.0.1\t1-1\t"090962,86443,A,相,関係,真偽,本体・代理,3.1040,04,02,01,亜,亜,あ,あ"\t\t\n'
        "00002.0.1\t1-1\t\t\t\n",
        encoding="shift_jis",
    )
    assert read_iwanami(defs, mapping) == {"090962": ["ある状態に近いさま。"]}
    assert read_iwanami(tmp_path / "missing.tsv", mapping) == {}
