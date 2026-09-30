"""Build the {{ORIGINAL_TEXT}} / {{DISCREPANCIES}} prompt for the
discrepancy_diagnosis step.

Pure formatting only — no I/O, no LLM calls. Mirrors the role of
`automation/src/rag/prompt_builder.py` for the evaluation step's prompt.
"""


def _candidate_tag(indicator_id: str, chunk_id: int) -> str:
    return f"{indicator_id}-{chunk_id}"


def format_discrepancies_block(
    discrepancy_rows: list[dict],
    evaluation_results: dict,
    candidates_by_indicator: dict[str, list[dict]],
) -> str:
    blocks: list[str] = []
    for row in discrepancy_rows:
        indicator_id = row["indicator_id"]
        eval_entry = evaluation_results.get(indicator_id, {})
        candidates = candidates_by_indicator.get(indicator_id, [])

        blocks.append(f"--- Discrepancy [{indicator_id}]: {row.get('indicator_value', '')} ---")
        blocks.append(f"Dimension: {row.get('dimension', 'unknown')}")
        blocks.append(
            f"Golden label: {row.get('golden_label')} | "
            f"Predicted label: {row.get('pred_label')} | "
            f"Error type: {row.get('error_type')}"
        )
        snippet = eval_entry.get("evidence") or "(none — judge predicted No)"
        blocks.append(f"Evidence snippet cited by the judge: {snippet}")
        rationale = eval_entry.get("rationale") or "(no rationale recorded)"
        blocks.append(f"Judge's own rationale: {rationale}")
        blocks.append("Candidate passages retrieved for this indicator at evaluation time:")
        if not candidates:
            blocks.append("  (none retrieved)")
        else:
            for candidate in candidates:
                tag = _candidate_tag(indicator_id, candidate["chunk_id"])
                blocks.append(f"  [{tag}] {candidate['text']}")
        blocks.append("")
    return "\n".join(blocks)


def build_diagnosis_prompt(
    original_text: str,
    discrepancy_rows: list[dict],
    evaluation_results: dict,
    candidates_by_indicator: dict[str, list[dict]],
    template_text: str,
) -> str:
    return template_text.replace("{{ORIGINAL_TEXT}}", original_text).replace(
        "{{DISCREPANCIES}}",
        format_discrepancies_block(discrepancy_rows, evaluation_results, candidates_by_indicator),
    )
