"""Tests for the pure helpers in scripts/candidates.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from candidates import SynsetContext, find_candidates, is_kana, pair_features
from wlsp import Wlsp

ROWS = [
    "000001,00001,A,用,関係,作用,合体・出会い・集合など,2.1550,09,01,01,出会う,出会う,であう,うあで",
    "000002,00002,A,用,関係,作用,合体・出会い・集合など,2.1550,09,01,02,会う,会う,あう,うあ",
    "000003,00003,A,用,関係,作用,合体・出会い・集合など,2.1550,09,02,01,出くわす,出くわす,でくわす,すわくで",
    "000004,00004,A,用,活動,交わり,応接・送迎,2.3520,01,01,01,会う,会う,あう,うあ",
    "000005,00005,A,用,活動,交わり,応接・送迎,2.3520,01,01,02,会見する,会見する,かいけんする,るすんけいか",
    "000006,00006,A,体,活動,心,学習・習慣・記憶,1.3050,01,01,01,勉強,勉強,べんきょう,うょきんべ",
]


class FakeWord:
    """Minimal stand-in for wn.Word."""

    def __init__(self, lemma: str, pos: str, forms: list[str] | None = None) -> None:
        self._lemma = lemma
        self.pos = pos
        self._forms = forms or [lemma]

    def lemma(self) -> str:
        return self._lemma

    def forms(self) -> list[str]:
        return self._forms


@pytest.fixture
def db(tmp_path: Path) -> Wlsp:
    path = tmp_path / "bunruidb.txt"
    path.write_text("\r\n".join(ROWS) + "\r\n", encoding="shift_jis")
    return Wlsp.load(path)


def test_is_kana() -> None:
    assert is_kana("あう")
    assert is_kana("コーヒー")
    assert not is_kana("会う")
    assert not is_kana("")


def test_find_candidates_by_lemma_and_form(db: Wlsp) -> None:
    word = FakeWord("会う", "v", ["会う", "アウ", "逢う"])
    assert find_candidates(word, db) == {"00002": "lemma", "00004": "lemma"}
    word = FakeWord("逢う", "v", ["逢う", "会う"])
    assert find_candidates(word, db) == {"00002": "form", "00004": "form"}


def test_find_candidates_suru_and_yomi(db: Wlsp) -> None:
    assert find_candidates(FakeWord("勉強する", "v"), db) == {"00006": "suru"}
    assert find_candidates(FakeWord("でくわす", "v"), db) == {"00003": "yomi"}
    assert find_candidates(FakeWord("存在しない", "n"), db) == {}


def test_pair_features_counts_overlap(db: Wlsp) -> None:
    ctx = SynsetContext(forms={"会う", "出会う", "出くわす"})
    ctx.related["sim"] = {"会見する"}
    rec = db.senses["00002"]  # 会う in 2.1550
    feats = pair_features(ctx, {"会う"}, rec, db)
    assert feats["ov_syn_subpara"] == 1  # 出会う
    assert feats["ov_syn_para"] == 2  # 出会う, 出くわす
    assert feats["ov_sim_item"] == 0
    assert feats["n_syn"] == 2
    assert feats["size_subpara"] == 2
    rec = db.senses["00004"]  # 会う in 2.3520
    feats = pair_features(ctx, {"会う"}, rec, db)
    assert feats["ov_syn_subpara"] == 0
    assert feats["ov_sim_subpara"] == 1  # 会見する


def test_gold_relation_and_reading_match(db: Wlsp) -> None:
    from candidates import gold_relation, reading_match

    assert gold_relation("00002", {"00002"}, db) == "exact"
    assert gold_relation("00001", {"00002"}, db) == "same_subpara"  # 出会う, same 小段落
    assert gold_relation("00003", {"00002"}, db) == "same_para"  # 出くわす, next 小段落
    assert gold_relation("00004", {"00002"}, db) == "other"  # 2.3520
    assert reading_match(FakeWord("会う", "v", ["会う", "アウ"]), "00002", db) == 1
    assert reading_match(FakeWord("会う", "v", ["会う", "カイ"]), "00002", db) == 0
    assert reading_match(FakeWord("会う", "v", ["会う"]), "00002", db) is None


def test_pick_decoys_avoid_own_word_and_categories(db: Wlsp) -> None:
    import random

    from candidates import pick_decoys

    # word 会う (v): its candidates occupy 2.1550 and 2.3520; only 会見する is left,
    # which is in 2.3520, so no decoy is possible
    assert pick_decoys({"会う"}, {"2.1550", "2.3520"}, "v", db, 2, random.Random(0)) == []
    # with only 2.1550 taken, decoys must come from 2.3520 and not be 会う itself
    decoys = pick_decoys({"会う"}, {"2.1550"}, "v", db, 2, random.Random(0))
    assert decoys and all(db.senses[m].item_key == "2.3520" for m in decoys)
    assert "00004" not in decoys  # 会う in 2.3520 has the word's own form
    assert pick_decoys({"青"}, set(), "n", db, 1, random.Random(0)) == ["00006"]
