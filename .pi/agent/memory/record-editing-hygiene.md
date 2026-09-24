---
name: record-editing-hygiene
description: Rules for editing .pi/agent/memory markdown - use the edit tool not python scripts on long table cells, and prefer heading-per-item structure where cells would need paragraphs
category: workflow
priority: 8
keep_updated: true
status: active
---

The experiments records under `.pi/agent/memory/experiments/` are mostly one-line markdown table cells
with long prose. Editing them from `python3 - <<EOF` scripts caused real damage three times in one
session:

- an `assert` failure mid-script left a file written and a commit made anyway (the commit was chained on a
  newline, not `&&`), so a record landed with an unreviewed mixture of edits;
- heredoc lines containing parentheses/quotes got mangled or raised `SyntaxError`, and one script that
  mixed escaped quotes inside a triple-quoted string produced a `SyntaxError` only at run time;
- a replacement anchored on a *tag* like `"| prefill, decode |\n| H10 |"` landed mid-sentence once other
  rows were inserted between them, which left a backlog cell containing three contradictory generations of
  prose and a broken sentence.

Rule: use the `edit` tool with a short unique `oldText` for these files, one call per file, and read the
cell back before committing. When a row needs a wholesale rewrite, replace the whole row rather than
patching a fragment of it. If a script is genuinely needed (multi-file analysis), have it *print* the
proposed change instead of writing it.

**Structure beats care here.** `experiments/plans/backlog.md` was converted to one `###` heading per
item with the fields as bullets underneath, because a cell that needs paragraphs cannot be edited safely
by exact-match text at all - the two failures above both came from appending a correction where a
replacement belonged. Files that are still wide tables carry the same risk, `experiments/INDEX.md` most
of all, since its `result` cells hold multi-sentence prose: when editing one, replace the whole row.

Related: `experiment-protocol.md` for the directory conventions, `qwen4exp-rdna4-project.md` for the project state.
