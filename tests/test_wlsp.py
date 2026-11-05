"""Tests for scripts/wlsp.py on a small Shift_JIS fixture."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wlsp import Wlsp, fold, hira_to_kata, kata_to_hira, normalise_midashi

ROWS = [
    "000001,00001,A,用,関係,作用,合体・出会い・集合など,2.1550,09,01,01,出会う,出会う,であう,うあで",
    "000002,00002,B,用,関係,作用,合体・出会い・集合など,2.1550,09,01,02,あう（会・遭）,あう,あう,うあ",
    "000003,00002,1,用,関係,作用,合体・出会い・集合など,2.1550,09,01,02,会う,会う,あう,うあ",
    "000004,00002,2,用,関係,作用,合体・出会い・集合など,2.1550,09,01,02,遭う,遭う,あう,うあ",
    "000005,00003,A,用,関係,作用,合体・出会い・集合など,2.1550,09,02,01,出くわす,出くわす,でくわす,すわくで",
    "000006,00004,A,用,活動,交わり,応接・送迎,2.3520,01,01,01,会う,会う,あう,うあ",
    "000007,00005,A,体,自然,自然,色,1.5020,12,01,01,青,青,あお,おあ",
]


@pytest.fixture
def db(tmp_path: Path) -> Wlsp:
    path = tmp_path / "bunruidb.txt"
    path.write_text("\r\n".join(ROWS) + "\r\n", encoding="shift_jis")
    return Wlsp.load(path)


def test_normalise_strips_hints() -> None:
    assert normalise_midashi("行（い）く") == "行く"
    assert normalise_midashi("浴びる［一風呂〜］") == "浴びる"
    assert normalise_midashi("会う") == "会う"


def test_kana_conversion_and_fold() -> None:
    assert hira_to_kata("あう") == "アウ"
    assert kata_to_hira("アウ") == "あう"
    assert fold("コーヒー") == fold("こーひー")
    assert fold("ＡＢＣ") == "ABC"


def test_senses_are_keyed_by_midashi_no(db: Wlsp) -> None:
    assert set(db.senses) == {"00001", "00002", "00003", "00004", "00005"}
    assert db.senses["00002"].record_type == "B"
    assert db.forms["00002"] == {"あう", "会う", "遭う"}


def test_lookup_uses_child_forms(db: Wlsp) -> None:
    assert db.lookup("遭う") == ["00002"]
    assert sorted(db.lookup("会う")) == ["00002", "00004"]
    assert db.lookup("なし") == []


def test_groups_and_neighbours(db: Wlsp) -> None:
    rec = db.senses["00002"]
    assert db.neighbours(rec, "subpara") == ["出会う"]
    assert db.neighbours(rec, "para") == ["出会う", "出くわす"]
    assert db.group_forms(rec, "item") >= {"出会う", "出くわす", "会う"}
    assert db.senses["00004"].middle_key == "2.35"
    assert db.senses["00005"].pos == "n"


def test_context_text(db: Wlsp) -> None:
    text = db.context_text(db.senses["00004"])
    assert text.startswith("用/活動/交わり/応接・送迎: 会う")
