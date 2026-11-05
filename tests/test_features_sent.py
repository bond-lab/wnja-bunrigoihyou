"""Tests for the text rendering in scripts/features_sent.py (no model needed)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from features_sent import model_tag, read_category_names, read_tufs_glosses, wlsp_texts
from wlsp import Wlsp

ROWS = [
    "000001,00001,A,用,関係,作用,合体・出会い・集合など,2.1550,09,01,01,出会う,出会う,であう,うあで",
    "000002,00002,A,用,関係,作用,合体・出会い・集合など,2.1550,09,01,02,会う,会う,あう,うあ",
    "000003,00003,A,用,関係,作用,合体・出会い・集合など,2.1550,09,02,01,出くわす,出くわす,でくわす,すわくで",
]


@pytest.fixture
def db(tmp_path: Path) -> Wlsp:
    path = tmp_path / "bunruidb.txt"
    path.write_text("\r\n".join(ROWS) + "\r\n", encoding="shift_jis")
    return Wlsp.load(path)


def test_model_tag() -> None:
    assert model_tag("BAAI/bge-m3") == "bge_m3"
    assert model_tag("intfloat/multilingual-e5-small") == "multilingual_e5_small"
    assert model_tag("/models/Qwen3-Embedding-4B/") == "qwen3_embedding_4b"


def test_wlsp_texts(db: Wlsp) -> None:
    texts = wlsp_texts(db.senses["00002"], db, {"2.15": "Action"})
    assert texts["cat"] == "用 関係 作用 合体・出会い・集合など (Action)"
    assert texts["sub"] == "会う: 出会う"
    assert (
        texts["ctx"]
        == "用 関係 作用 合体・出会い・集合など (Action): 会う; 出会う; 出会う、出くわす"
    )
    with_gloss = wlsp_texts(db.senses["00002"], db, {}, gloss="meet")
    assert with_gloss["sub"] == "会う: 出会う (meet)"
    assert with_gloss["cat"] == "用 関係 作用 合体・出会い・集合など"


def test_read_category_names_and_glosses(tmp_path: Path) -> None:
    names = tmp_path / "bunrui_names.tsv"
    names.write_text("Code\ten\tms\tmy\tja\n2.15\tAction\tx\ty\t作用\n", encoding="utf-8")
    assert read_category_names(names) == {"2.15": "Action"}
    assert read_category_names(tmp_path / "missing.tsv") == {}
    vocab = tmp_path / "tufs-vocab.tsv"
    vocab.write_text(
        "cid\tlang\twid\tlemma\tcomment\tiids\texamples\tis_basic\tscenes\tbunrui\n"
        "84044\tja\t903\t会います\t【よみ】; あいます;【意味】; meet; see\t1\t\t1\t\t2.3520\n"
        "84044\ten\t1\tmeet\t\t\t\t1\t\t\n"
        "None\tja\t2\tx\t【意味】; y\t\t\t1\t\t\n",
        encoding="utf-8",
    )
    assert read_tufs_glosses(vocab) == {"084044": "meet, see"}
