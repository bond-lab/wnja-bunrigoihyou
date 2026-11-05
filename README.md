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

## Mapping pipeline

Each step adds columns to one candidate table in `build/` (gitignored), so the
tiers can run on different machines and the table can be copied between them.

```bash
# core (laptop): candidate pairs with structural features, then score and evaluate
uv run scripts/candidates.py                    # -> build/candidates.parquet
uv run scripts/score.py                         # -> wnja-wlsp.tsv, build/scored.parquet, build/review.tsv
uv run scripts/evaluate.py --name core          # -> dev/eval.md

# tier 1 (laptop): chiVe word-vector similarity features
curl -o build/chive/chive-1.3-mc30_gensim.tar.gz \
  https://sudachi.s3-ap-northeast-1.amazonaws.com/chive/chive-1.3-mc30_gensim.tar.gz
tar xzf build/chive/chive-1.3-mc30_gensim.tar.gz -C build/chive
uv run scripts/features_chive.py                # adds chive_* columns in place
uv run scripts/score.py && uv run scripts/evaluate.py --name chive
```

`candidates.py` pairs every wnja sense with the WLSP senses (見出し番号) whose
written form matches the word's lemma or a form, falling back to the reading
for kana words, and records how the synset's neighbourhood (members,
hypernyms, hyponyms, co-hyponyms, similar and antonym synsets, hierarchy via
ILI into omw-en 1.4) overlaps the candidate's 小段落, 段落, 分類項目 and 中項目.
`score.py` fits a logistic regression on the dev-set labels (gold pair against
the other candidates of the same sense) with grouped cross-validation, and
writes `wnja-wlsp.tsv` with one row per link: wnja sense, synset, ILI, lemma,
WLSP 見出し番号, record, code, labels, `confidence` and an `evidence` summary.
Links are many-to-many: every pair above the threshold is kept.

Dev-set results so far (303 ambiguous senses; strict = the gold record,
lenient = any record in the gold's 小段落 or 段落, since WLSP lists the same
headword more than once nearby; P/R at confidence 0.5, lenient):

| features | top-1 strict | top-1 lenient | MRR lenient | P / R |
|---|---|---|---|---|
| hand-weighted baseline | 0.627 | 0.725 | 0.843 | |
| core (structural) | 0.696 | 0.741 | 0.857 | 0.81 / 0.80 |
| core + chiVe (tier 1) | 0.733 | 0.767 | 0.871 | 0.83 / 0.86 |
| + multilingual-e5-small (CPU preview of tier 2) | 0.706 | 0.754 | 0.863 | 0.83 / 0.86 |
| + within-sense relative similarities | 0.686 | 0.761 | 0.865 | 0.87 / 0.89 |

Embedding cosines sit in a narrow band (e5-small: 0.85 to 0.88), so the raw
values were ignored by the tree model; expressed relative to the best
candidate of the same sense (`rel_*` columns, added in `score.py`) the e5
category similarity becomes the most used feature and the many-to-many
precision/recall at 0.5 rises to 0.87/0.89 (0.92/0.82 at 0.7). Top-1 is
unchanged within the noise of 303 senses.

Links are scored pair by pair without candidate-count priors, and dev
candidates in the same group as the gold are left unlabelled rather than
treated as negatives, so genuine many-to-many links are not penalised. Note
the dev set itself has almost no wnja-sense-to-several-records links (a TUFS
concept is one WLSP record), so that direction is unmeasured.

### Tier 2: sentence embeddings (GPU or Apple silicon)

```bash
uv sync --extra sent                              # torch + sentence-transformers
# copy build/candidates.parquet from the laptop, then, per model:
uv run scripts/features_sent.py --model BAAI/bge-m3 --device mps
uv run scripts/features_sent.py --model intfloat/multilingual-e5-small \
    --query-prefix "query: " --doc-prefix "passage: "
uv run scripts/score.py && uv run scripts/evaluate.py --name bge-m3
```

`features_sent.py` embeds each synset (members, Japanese definition,
examples; `--en-gloss` appends the omw-en definition) and three texts per
WLSP sense: the category path with the English 中項目 name from the TUFS
`bunrui_names.tsv` (`cat`), the headword with its 小段落 neighbours (`sub`),
and the path with 小段落 and 段落 neighbours (`ctx`). The cosines become
`sent_<model>_cat/sub/ctx`, so several models can be compared in one table.
Embeddings are cached in `build/emb/<model>/`. `--max-texts 300` is a smoke
test. `--tufs-gloss ../tufs/tufs-vocab.tsv` adds TUFS English glosses to the
~800 WLSP records that are TUFS concepts; those are the dev set, so only use
it for the final mapping, never for evaluation.

### Tier 2b: Claude as judge for uncertain words

```bash
uv sync --extra judge                    # anthropic SDK; needs ANTHROPIC_API_KEY
uv run scripts/judge_claude.py --dry-run             # prompts + size estimate, no API calls
uv run scripts/judge_claude.py --sync --max-words 20 # small direct run to check output
uv run scripts/judge_claude.py --submit              # Message Batches (half price)
uv run scripts/judge_claude.py --collect             # later: fetch finished batches
uv run scripts/judge_claude.py --apply               # judge_* columns -> candidates.parquet
uv run scripts/score.py && uv run scripts/evaluate.py --name judge
```

Words are selected when a pair's confidence falls in the uncertain band
(0.3 to 0.7) or a sense's top two candidates are within 0.15. One prompt per
word lists all its wnja senses and all its WLSP senses (code, labels, 小段落
and 段落 neighbours, and the 岩波国語辞典 definition when
`build/iwanami_defs.tsv` exists, see tier 3) and asks for the links that
hold, each with a relation (same, narrower, broader, related) and a
confidence, as JSON via structured outputs. Verdicts are cached in
`build/judge.jsonl`. Default model `claude-sonnet-5-5` at medium effort;
`--model claude-opus-5-5` for a stronger judge.

### Tier 3: 岩波国語辞典 definitions (TUAT)

`data/wlsp2iwanami` links 60k WLSP records to Iwanami sense IDs. On a machine
with the 岩波国語辞典第五版タグ付きコーパス, write `build/iwanami_defs.tsv`
(`senseid<TAB>definition`, UTF-8); the judge picks it up automatically and a
definition-to-definition embedding feature can be added in `features_sent.py`.

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
