# Legacy raw-text path

Steps `translation`, `translation_qa` and `markdown` translate the raw OCR text
(`paths.input_dir`) instead of the curated Markdown corpus. The code lives in
`automation/src/legacy/`, and the CLI still accepts these step names (for example
`--steps translation,translation_qa,markdown`). They are not part of the default chain.
Helpers shared with the Markdown path are in `automation/src/common.py`.

## Translation QA + re-align (`translation_qa` step)

Runs right after `translation`, using a different model than translation by default (`gpt-5.2-translation-qa` vs. the translation model) as an independent cross-check. See [`results/translation_qa/`](#translation_qa) above for what it checks and writes. Idempotent like the other LLM steps: skips a policy whose `translation_qa/<stem>.json` report already exists unless `--force` is passed.

## Markdown rendering (`markdown` step)

`results/cleaned_text/<policy>.cleaned.txt`, `results/cleaned_markdown/<policy>.cleaned.md`,
and `results/translation_markdown/<policy>.md` are written by their own
pipeline step, `markdown` — independent of `translation`, and idempotent: it
only (re)renders a policy whose outputs don't exist yet, so it's safe to run
anytime (`--steps markdown`) to backfill whatever is missing, without
re-running translation.

- `results/cleaned_text/<policy>.cleaned.txt` and
  `results/cleaned_markdown/<policy>.cleaned.md` are both rendered straight
  from the raw original-language input (`data/raw/localpolicies/<policy>.txt`)
  — they have no dependency on translation having run. `cleaned_text` is
  plain text (mirrors `translation/<policy>.txt`); `cleaned_markdown` is the
  same content rendered as Markdown (mirrors `translation_markdown`).
- `results/translation_markdown/<policy>.md` is rendered from
  `results/translation/<policy>.txt` (the translated English output), so a
  given policy is only rendered once its translation exists; policies not
  yet translated are silently skipped, not treated as failures.
- `results/translation/<policy>.txt` itself always stays plain text — it's
  what the storage/RAG and evaluation steps read, and reformatting it as
  Markdown in place would change both.

The Markdown artifacts are rendered by `chunking.render_markdown` on top of
`chunking.clean_text` (OCR-noise removal, paragraph reflow, structural-marker
detection — `automation/src/chunking.py:clean_text`): structural markers
become headings nested by keyword (`BAB`/`Chapter` → `#`,
`Bagian`/`Part`/`Section` → `##`, `Paragraf`/`Paragraph` → `###`,
`Pasal`/`Article` → `####`, with an unrecognized/foreign-language marker
nested one level under the most recent keyword heading), and
numbered/lettered/roman-numeral list starts become proper
ordered/nested-bullet Markdown lists. The plain-text artifacts
(`cleaned_text`, `translation`) skip that rendering step entirely.

`markdown` belongs to the raw-OCR-text path and is **not** part of the
default chain any more — run it explicitly with `--steps markdown` alongside
`--steps translation`. See `run_markdown_step` in
`automation/src/legacy/translate.py`. In the Markdown path the source side needs no
rendering at all (the input is already Markdown, and is snapshotted to
`results/source_markdown/`), and the translated side *is* Markdown.

## Chunked translation (`translation.chunking`)

Every chunked translation run cleans and structurally splits each source file
first — strips OCR noise lines (including stray lone-punctuation residue like
a leftover `.` on its own line), reflows broken lines back into paragraphs,
and keeps structural markers (`BAB` / `Pasal` / `Bagian` / `Paragraf` /
all-caps titles) on their own line (`automation/src/chunking.py:clean_text`) —
this is internal to chunking (it decides the chunk boundaries) and isn't
persisted on its own; the `markdown` step above re-derives the same cleaning
from the raw input when it renders `results/cleaned_text/*.cleaned.txt` and
`results/cleaned_markdown/*.cleaned.md`.

Some source policy files are large enough that a single translation call risks
a timeout. When `translation.chunking.enabled` is `true`, `translate.py` runs
each file through the rest of the chunk -> translate -> combine pipeline
instead of a single API call:

1. **Cleaning** — see above.
2. **Structural split** — cuts the cleaned text into sections at structural
   markers (no overlap; these are the document's natural boundaries).
   Consecutive markers with no body text between them (e.g. a `BAB` line
   followed immediately by its all-caps title and then `Pasal 1`) merge into
   the same section instead of each becoming a one-line section — a new
   section only starts once a marker follows actual body content.
3. **Fallback split** — only for a section that still exceeds `safe_limit`
   characters (e.g. a Lampiran/appendix): splits further on sentence
   boundaries (`.` / `;`), carrying the previous sub-chunk's trailing text as
   non-translated context so the model can resolve continuations.
4. **Translate** — structural chunks are translated directly with the normal
   translation prompt; fallback chunks use
   `automation/prompts/translation/chunking/fallback_prompt.txt`, which
   instructs the model to translate only the marked target text.
5. **Combine** — sub-chunks within a section are concatenated directly;
   sections are joined with a blank line.

```yaml
translation:
  model: claude-sonnet-4-5
  prompt_version: v2
  chunking:
    enabled: true
    safe_limit: 32000   # chars; default 32000, well under the ~50,000 timeout ceiling
```

Chunking is disabled by default — existing runs are unaffected unless `enabled: true` is set.

When chunking is enabled, `mid_product/chunks/<policy>.chunks.json` is also always written — the chunk list (section 2 output): `chunk_index`, `section_id`, `type` (`structural`/`fallback`), `context`, `text`, `translated_text`, in translation order. The RAG storage step reuses it instead of re-chunking the merged English output.

## Sliding-window evaluation

The sliding-window evaluation method (`evaluation.method: sliding_window`) is deprecated. Its
code lives in `automation/src/legacy/` (`run_eval_sliding_window.py`, `sliding_window/`) and its
tests in `automation/tests/legacy/`. RAG is the default; translation is chunk-and-combine only.
