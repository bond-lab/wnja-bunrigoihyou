# Script explainers

One self-contained HTML explainer per script, generated from the code with the
`code-explainer` skill (background, intuition, code walkthrough with
`file:line` references, quiz). Each `<name>_explainer.html` is a symlink to
the latest version `<name>_explainer.vN.html`; `<name>_explainer.notes.md`
records what produced each version.

| Script | Explainer |
|---|---|
| `wlsp.py` | [wlsp_explainer.html](wlsp_explainer.html) |
| `build_devset.py` | [build_devset_explainer.html](build_devset_explainer.html) |
| `candidates.py` | [candidates_explainer.html](candidates_explainer.html) |
| `score.py` | [score_explainer.html](score_explainer.html) |
| `evaluate.py` | [evaluate_explainer.html](evaluate_explainer.html) |
| `features_chive.py` | [features_chive_explainer.html](features_chive_explainer.html) |
| `features_sent.py` | [features_sent_explainer.html](features_sent_explainer.html) |
| `judge_claude.py` | [judge_claude_explainer.html](judge_claude_explainer.html) |

The pages are AI-generated from the code at a given commit (stamped in each
page) and can be wrong or stale: verify against the source. To lint one
against the current code:

```sh
uv run --no-project ~/.claude/skills/code-explainer/explainer_check.py --no-net \
    scripts/docs/wlsp_explainer.html scripts/wlsp.py
```
