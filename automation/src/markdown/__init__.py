"""Markdown-in / Markdown-out translation path.

Parallel to the raw-text path in `automation.src.translate` /
`automation.src.translation_qa`: the input is the curated Markdown corpus in
`data/processed/localpolicies/cleaned_markdown/`, whose structure (BAB /
Bagian / Paragraf / Pasal headings) is already explicit, so nothing needs to
re-detect it with regexes on OCR text.

Modules:
- `chunking`  -- parse headings, pack sections into translation units, verify
                 that a translation kept the source's structure.
- `translate` -- the `translation_md` pipeline step.
- `qa`        -- the `translation_qa_md` pipeline step.
- `to_text`   -- the `md_to_text` step: translated Markdown -> the plain-text
                 artifact every downstream stage reads, plus the per-heading
                 retrieval chunks that RAG storage consumes unchanged.
"""
