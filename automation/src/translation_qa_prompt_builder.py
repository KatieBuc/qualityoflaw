"""Build the {{ORIGINAL_TEXT}} / {{TRANSLATED_TEXT}} / {{CONTEXT}} prompt for
the translation_qa step.

Pure formatting only — no I/O, no LLM calls. Mirrors the role of
`automation/src/diagnose_prompt_builder.py` for the discrepancy_diagnosis step.
"""


def build_translation_qa_prompt(
    original_text: str,
    translated_text: str,
    template_text: str,
    context: str | None = None,
) -> str:
    context_block = (
        f"### Preceding Context (already-translated tail of the previous chunk, for reference only — do not re-fix or duplicate it)\n{context}\n"
        if context
        else ""
    )
    return (
        template_text.replace("{{ORIGINAL_TEXT}}", original_text)
        .replace("{{TRANSLATED_TEXT}}", translated_text)
        .replace("{{CONTEXT}}", context_block)
    )
