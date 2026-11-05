# wnja-bunrigoihyou

Mapping Japanese WordNet (wnja) senses to 分類語彙表 (Word List by Semantic
Principles, WLSP) senses.

## Data

Cloned into `data/` (not committed; all CC BY-NC-SA 3.0, NINJAL):

| Directory | Source | Contents |
|---|---|---|
| `data/WLSP` | <https://github.com/masayu-a/WLSP> | 分類語彙表増補改訂版データベース ver 1.0.1 (`bunruidb.txt`, Shift_JIS, 101,067 records) |
| `data/wlsp2unidic` | <https://github.com/masayu-a/wlsp2unidic> | WLSP number to UniDic lemma ID |
| `data/wlsp2iwanami` | <https://github.com/masayu-a/wlsp2iwanami> | WLSP to 岩波国語辞典 senses |

```bash
mkdir -p data && cd data
for r in WLSP wlsp2unidic wlsp2iwanami; do git clone --depth 1 https://github.com/masayu-a/$r; done
```

The Japanese WordNet (`../wnja/build/wnja-2.1.xml`) and the TUFS Japanese
wordnet (`../tufs/build/lmf/tufs-ja.xml`) are loaded into a project-local
`wn` database in `.wn_data/` (not committed):

```bash
uv sync
uv run python -c "
import wn, pathlib
wn.config.data_directory = pathlib.Path('.wn_data').resolve()
wn.add('../wnja/build/wnja-2.1.xml')
wn.add('../tufs/build/lmf/tufs-ja.xml')"
```

## Dev set

The TUFS basic vocabulary (`../tufs`) uses the WLSP record ID as its concept
ID, and `tufs-omw-map.tsv` links those concepts to Princeton WordNet 3.0
synsets. Composing the two gives a sense-level gold standard: each WLSP
record is one sense of a headword, and the script looks that headword up in
the linked synset in wnja to get the wnja sense ID.

```bash
uv run scripts/build_devset.py --tufs ../tufs -o dev/devset.tsv
```

`dev/devset.tsv` has one row per (WLSP record, synset) pair:

| Column | Meaning |
|---|---|
| `cid` | TUFS concept ID = WLSP record ID |
| `wlsp_record`, `wlsp_midashi_no` | レコードID, 見出し番号 |
| `wlsp_code` | 分類番号-段落-小段落-語番号, e.g. `2.3520-01-01-02` |
| `wlsp_labels` | 類/部門/中項目/分類項目 |
| `wlsp_midashi`, `wlsp_yomi` | 見出し and 読み as given in WLSP |
| `lemma` | dictionary form: the WLSP 見出し本体 with reading hints and notes stripped, e.g. 会う |
| `polite` | TUFS polite form of a verb, e.g. 会います; empty for other words |
| `tufs_form`, `tufs_en` | raw TUFS Japanese string (may carry a `(2)` index) and English gloss |
| `relation` | link type from TUFS: `synonym`, `hyponym`, `multi` |
| `pwn30`, `wnja_synset`, `ili` | synset identifiers |
| `wnja_sense`, `wnja_lemma`, `match` | wnja sense found in the synset, the lemma it was found under, and which string matched |
| `status` | `sense`, `synset-only`, `no-synset` or `hyponym` (see below) |
| `tufs_sense` | sense ID in the TUFS Japanese wordnet, which has an entry for every linked TUFS Japanese lemma |
| `wnja_definition` | Japanese gloss of the synset |

Status of the 734 rows (503 WLSP records) in the current build:

| Status | Rows | Meaning |
|---|---|---|
| `sense` | 599 | WLSP record linked to a wnja sense |
| `synset-only` | 97 | synset is in wnja but lacks the WLSP headword: candidate new wnja senses |
| `no-synset` | 27 | synset has no Japanese entry in wnja 2.1 (or is a TUFS-only synset) |
| `hyponym` | 11 | WLSP sense is narrower than the synset, so no sense is expected |

Where wnja lacks the sense, the TUFS wordnet usually has it: 648 rows carry a
`tufs_sense`, and only 26 rows (concepts with no Japanese TUFS row) have
neither a wnja nor a TUFS sense. The `synset-only` and `no-synset` rows are
therefore candidates for adding TUFS senses to wnja.

A further 343 Japanese TUFS entries have a WLSP record but no synset link.

## Development

```bash
uv run ruff check scripts tests
uv run pytest tests
```

## Related work

No published mapping between the Japanese WordNet and WLSP was found (search
of ACL Anthology, 言語処理学会, NINJAL, Consensus; 2026-11-05). The closest
work is NINJAL's WLSP alignments to UniDic (Kondo and Tanaka 2017; Katayama
et al. 2024) and to 岩波国語辞典 senses (Wu et al. 2019), the WLSP corpus
annotations (Kato et al. 2018; Asahara et al. 2022), and the WordNet to
Goi-Taikei alignments (Asanoma 2001; Bond et al. 2009).
