"""Shared access to the 分類語彙表 (WLSP) database.

The database (``bunruidb.txt``, Shift_JIS CSV) has one record per headword
occurrence.  A headword listed with several readings or spellings is a type
``B`` record followed by numbered child records (types ``1``, ``2``, ...)
that share its 見出し番号.  Here a *sense* is identified by the 見出し番号,
with the parent (or the sole ``A`` record) as its canonical entry and the
children contributing extra written forms and readings.

Position in the thesaurus is given by the 分類番号 (類.部門中項目分類項目,
e.g. ``2.3520``) plus 段落番号, 小段落番号 and 語番号.  Records in the same
小段落 are close synonyms; the same 段落 groups related senses; the same
分類項目 is a topical category.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Columns of bunruidb.txt (see kaisetsu.txt in the WLSP distribution).
WLSP_FIELDS = (
    "record_id",
    "midashi_no",
    "record_type",
    "rui",
    "bumon",
    "chukoumoku",
    "bunrui_koumoku",
    "bunrui_no",
    "danraku",
    "shodanraku",
    "go_no",
    "midashi",
    "midashi_hontai",
    "yomi",
    "gyakuyomi",
)

# 類 (first digit of the 分類番号) to wordnet part of speech.
RUI_TO_POS = {"1": "n", "2": "v", "3": "a", "4": "r"}

_BRACKETS = re.compile(r"（[^）]*）|\[[^\]]*\]|［[^］]*］|〔[^〕]*〕")


@dataclass(frozen=True)
class WlspEntry:
    """One record of the WLSP database (a sense of a headword)."""

    record_id: str
    midashi_no: str
    record_type: str
    bunrui_no: str
    danraku: str
    shodanraku: str
    go_no: str
    midashi: str
    midashi_hontai: str
    yomi: str
    labels: str

    @property
    def code(self) -> str:
        """Full WLSP position code, e.g. ``2.3520-01-01-02``."""
        return f"{self.bunrui_no}-{self.danraku}-{self.shodanraku}-{self.go_no}"

    @property
    def lemma(self) -> str:
        """Headword with reading hints and usage notes stripped."""
        return normalise_midashi(self.midashi_hontai)

    @property
    def pos(self) -> str:
        """Wordnet part of speech implied by the 類 digit."""
        return RUI_TO_POS.get(self.bunrui_no[0], "x")

    @property
    def item_key(self) -> str:
        """Key of the 分類項目 group."""
        return self.bunrui_no

    @property
    def middle_key(self) -> str:
        """Key of the 中項目 group (類.部門中項目), e.g. ``2.35``."""
        return self.bunrui_no[:4]

    @property
    def para_key(self) -> tuple[str, str]:
        """Key of the 段落 group."""
        return (self.bunrui_no, self.danraku)

    @property
    def subpara_key(self) -> tuple[str, str, str]:
        """Key of the 小段落 group."""
        return (self.bunrui_no, self.danraku, self.shodanraku)


def normalise_midashi(midashi: str) -> str:
    """Strip reading hints and usage notes from a WLSP headword.

    Args:
        midashi: Headword as given in WLSP, e.g. ``行（い）く`` or ``浴びる［一風呂〜］``

    Returns:
        The bare headword, e.g. ``行く`` or ``浴びる``
    """
    return _BRACKETS.sub("", midashi).strip()


def hira_to_kata(text: str) -> str:
    """Convert hiragana to katakana, leaving other characters unchanged.

    Args:
        text: Any string

    Returns:
        The string with hiragana (U+3041 to U+3096) shifted to katakana
    """
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def kata_to_hira(text: str) -> str:
    """Convert katakana to hiragana, leaving other characters unchanged.

    Args:
        text: Any string

    Returns:
        The string with katakana (U+30A1 to U+30F6) shifted to hiragana
    """
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def fold(text: str) -> str:
    """Normalise a written form for matching: NFKC, katakana folded to hiragana.

    Args:
        text: A lemma or headword

    Returns:
        The folded form, so that ``コーヒー`` and ``こーひー`` compare equal
    """
    return kata_to_hira(unicodedata.normalize("NFKC", text))


def read_wlsp(path: Path) -> dict[str, WlspEntry]:
    """Read the WLSP database keyed by record ID.

    Args:
        path: Path to ``bunruidb.txt``

    Returns:
        Mapping from zero-padded record ID to entry, in file order
    """
    entries: dict[str, WlspEntry] = {}
    with path.open(encoding="shift_jis", newline="") as fh:
        for row in csv.reader(fh):
            rec = dict(zip(WLSP_FIELDS, row, strict=True))
            entries[rec["record_id"]] = WlspEntry(
                record_id=rec["record_id"],
                midashi_no=rec["midashi_no"],
                record_type=rec["record_type"],
                bunrui_no=rec["bunrui_no"],
                danraku=rec["danraku"],
                shodanraku=rec["shodanraku"],
                go_no=rec["go_no"],
                midashi=rec["midashi"],
                midashi_hontai=rec["midashi_hontai"],
                yomi=rec["yomi"],
                labels="/".join(
                    (rec["rui"], rec["bumon"], rec["chukoumoku"], rec["bunrui_koumoku"])
                ),
            )
    return entries


@dataclass
class Wlsp:
    """The WLSP database indexed by sense (見出し番号), form, reading and position.

    Attributes:
        records: All records by record ID
        senses: Canonical entry per 見出し番号 (the ``A`` or ``B`` parent record)
        forms: Folded written forms (parent and children) per 見出し番号
        readings: Readings per 見出し番号
        by_form: 見出し番号s per folded written form
        by_yomi: 見出し番号s per reading
        by_pos: 見出し番号s per wordnet part of speech (from the 類 digit)
        groups: For each level (``subpara``, ``para``, ``item``, ``middle``),
            the 見出し番号s in each group key
    """

    records: dict[str, WlspEntry]
    senses: dict[str, WlspEntry] = field(default_factory=dict)
    forms: dict[str, set[str]] = field(default_factory=dict)
    readings: dict[str, set[str]] = field(default_factory=dict)
    by_form: dict[str, list[str]] = field(default_factory=dict)
    by_yomi: dict[str, list[str]] = field(default_factory=dict)
    by_pos: dict[str, list[str]] = field(default_factory=dict)
    groups: dict[str, dict[object, list[str]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Wlsp:
        """Read ``bunruidb.txt`` and build the indices.

        Args:
            path: Path to ``bunruidb.txt``

        Returns:
            The indexed database
        """
        db = cls(records=read_wlsp(path))
        forms: dict[str, set[str]] = defaultdict(set)
        readings: dict[str, set[str]] = defaultdict(set)
        for rec in db.records.values():
            if rec.record_type in ("A", "B"):
                db.senses[rec.midashi_no] = rec
            forms[rec.midashi_no].add(fold(rec.lemma))
            readings[rec.midashi_no].add(rec.yomi)
        db.forms = dict(forms)
        db.readings = dict(readings)
        by_form: dict[str, list[str]] = defaultdict(list)
        by_yomi: dict[str, list[str]] = defaultdict(list)
        by_pos: dict[str, list[str]] = defaultdict(list)
        groups: dict[str, dict[object, list[str]]] = {
            level: defaultdict(list) for level in ("subpara", "para", "item", "middle")
        }
        for mid, rec in db.senses.items():
            for f in db.forms[mid]:
                by_form[f].append(mid)
            for y in db.readings[mid]:
                by_yomi[y].append(mid)
            by_pos[rec.pos].append(mid)
            groups["subpara"][rec.subpara_key].append(mid)
            groups["para"][rec.para_key].append(mid)
            groups["item"][rec.item_key].append(mid)
            groups["middle"][rec.middle_key].append(mid)
        db.by_form = dict(by_form)
        db.by_yomi = dict(by_yomi)
        db.by_pos = dict(by_pos)
        db.groups = {level: dict(g) for level, g in groups.items()}
        return db

    def lookup(self, form: str) -> list[str]:
        """Find senses whose written form matches, falling back to the reading.

        Args:
            form: A lemma or written form

        Returns:
            Matching 見出し番号s (possibly empty)
        """
        return self.by_form.get(fold(form), [])

    def group_key(self, rec: WlspEntry, level: str) -> object:
        """Return the group key of ``rec`` at ``level``.

        Args:
            rec: A WLSP entry
            level: ``subpara``, ``para``, ``item`` or ``middle``

        Returns:
            The key used in ``self.groups[level]``
        """
        return {
            "subpara": rec.subpara_key,
            "para": rec.para_key,
            "item": rec.item_key,
            "middle": rec.middle_key,
        }[level]

    def group_forms(self, rec: WlspEntry, level: str) -> set[str]:
        """Folded written forms of all senses in the same group as ``rec``.

        Args:
            rec: A WLSP entry
            level: ``subpara``, ``para``, ``item`` or ``middle``

        Returns:
            Set of folded forms, including those of ``rec`` itself
        """
        out: set[str] = set()
        for mid in self.groups[level][self.group_key(rec, level)]:
            out |= self.forms[mid]
        return out

    def neighbours(self, rec: WlspEntry, level: str, limit: int = 0) -> list[str]:
        """Headwords of the other senses in the same group as ``rec``, in order.

        Args:
            rec: A WLSP entry
            level: ``subpara``, ``para``, ``item`` or ``middle``
            limit: Maximum number to return; 0 for all

        Returns:
            Bare headwords, excluding ``rec`` itself
        """
        out = [
            self.senses[mid].lemma
            for mid in self.groups[level][self.group_key(rec, level)]
            if mid != rec.midashi_no
        ]
        return out[:limit] if limit else out

    def context_text(self, rec: WlspEntry, para_limit: int = 20) -> str:
        """Render a record's position as text for embedding or prompting.

        Args:
            rec: A WLSP entry
            para_limit: Maximum number of 段落 neighbours to include

        Returns:
            One line: category labels, then 小段落 and 段落 neighbours
        """
        sub = "、".join(self.neighbours(rec, "subpara"))
        para = "、".join(self.neighbours(rec, "para", para_limit))
        return f"{rec.labels}: {rec.lemma}; 同義: {sub}; 関連: {para}"
