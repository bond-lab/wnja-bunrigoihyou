"""Tests for the pure helpers in scripts/build_devset.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_devset import hira_to_kata, normalise_midashi, polite_form


def test_normalise_strips_reading_hint() -> None:
    assert normalise_midashi("行（い）く") == "行く"
    assert normalise_midashi("明日（あした）") == "明日"


def test_normalise_strips_usage_note() -> None:
    assert normalise_midashi("浴びる［一風呂〜］") == "浴びる"


def test_normalise_leaves_plain_headword() -> None:
    assert normalise_midashi("会う") == "会う"
    assert normalise_midashi("あたたかい（暖・温）") == "あたたかい"


def test_hira_to_kata() -> None:
    assert hira_to_kata("あう") == "アウ"
    assert hira_to_kata("会う") == "会ウ"
    assert hira_to_kata("ABC") == "ABC"


def test_row_status() -> None:
    from build_devset import row_status

    assert row_status("synonym", True, True) == "sense"
    assert row_status("synonym", True, False) == "synset-only"
    assert row_status("synonym", False, False) == "no-synset"
    assert row_status("hyponym", True, False) == "hyponym"


def test_polite_form() -> None:
    assert polite_form("会います") == "会います"
    assert polite_form("入れます(2)") == "入れます"
    assert polite_form("あります(2);ございます") == "あります"
    assert polite_form("会う") == ""
    assert polite_form("あの(1)") == ""
    assert polite_form("") == ""
