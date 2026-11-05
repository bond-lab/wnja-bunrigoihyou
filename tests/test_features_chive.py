"""Tests for scripts/features_chive.py with a tiny fake vector table."""

import sys
from pathlib import Path

import numpy as np
import pytest
from gensim.models import KeyedVectors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from features_chive import GroupVectors, Vectors, centroid, pair_features
from wlsp import Wlsp

ROWS = [
    "000001,00001,A,用,関係,作用,合体・出会い・集合など,2.1550,09,01,01,出会う,出会う,であう,うあで",
    "000002,00002,A,用,関係,作用,合体・出会い・集合など,2.1550,09,01,02,会う,会う,あう,うあ",
    "000003,00003,A,用,関係,作用,合体・出会い・集合など,2.1550,09,02,01,出くわす,出くわす,でくわす,すわくで",
    "000004,00004,A,用,活動,交わり,応接・送迎,2.3520,01,01,01,会う,会う,あう,うあ",
    "000005,00005,A,用,活動,交わり,応接・送迎,2.3520,01,01,02,会見する,会見する,かいけんする,るすんけいか",
]


@pytest.fixture
def db(tmp_path: Path) -> Wlsp:
    path = tmp_path / "bunruidb.txt"
    path.write_text("\r\n".join(ROWS) + "\r\n", encoding="shift_jis")
    return Wlsp.load(path)


@pytest.fixture
def vectors() -> Vectors:
    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(
        ["会う", "出会う", "出くわす", "会見する", "面会"],
        np.array(
            [[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0], [0, 1, 0], [0.1, 0.9, 0]], dtype=np.float32
        ),
    )
    return Vectors(kv, tokenizer=None)


def test_vectors_are_unit_and_cached(vectors: Vectors) -> None:
    v = vectors.get("出会う")
    assert v is not None and abs(np.linalg.norm(v) - 1) < 1e-6
    assert vectors.get("不明") is None
    assert vectors.n_missing == 1
    assert vectors.get("不明") is None and vectors.n_missing == 1


def test_centroid_empty() -> None:
    assert centroid([]) is None


def test_pair_features_prefer_the_right_group(db: Wlsp, vectors: Vectors) -> None:
    groups = GroupVectors(db, vectors)
    word = vectors.get("会う")
    others = [vectors.get("面会")]
    meet = pair_features(word, others, "00002", groups)  # 2.1550 with 出会う
    visit = pair_features(word, others, "00004", groups)  # 2.3520 with 会見する
    assert meet["chive_sub_max"] > visit["chive_sub_max"]
    assert visit["chive_syn_sub"] > meet["chive_syn_sub"]  # 面会 is near 会見する
    assert meet["chive_para_mean"] is not None
    assert pair_features(None, [], "00002", groups)["chive_sub_max"] is None
