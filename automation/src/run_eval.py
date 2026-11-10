import json
import logging
import threading
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    RetrievalConfig,
    get_output_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.criteria import (
    EXPECTED_INDICATOR_IDS,
    validate_criteria_files,
    extract_schema_from_prompts,
    list_criteria_files,
)
from automation.src.failure_log import clear_failure, record_failure
from automation.src.llm.confidence import ConfidenceSpec
from automation.src.llm.wrapper import AzureLLMWrapper, ContextLengthExceeded, LLMCallError, format_api_error
from automation.src.policy_files import filter_policy_files
from automation.src.rag.embedder import AzureEmbedder
from automation.src.rag.embedding_align import AlignerUnavailable, warm_up
from automation.src.rag.evidence_check import (
    resolve_evidence_citation,
    resolve_evidence_original_alignment,
    resolve_evidence_source_text,
)
from automation.src.rag.prompt_builder import (
    CHARS_PER_TOKEN,
    TrimReport,
    build_candidate_lookup,
    build_source_alignment_lookup,
    build_source_text_lookup,
    cited_candidates,
    format_criteria_list,
    format_criteria_with_candidates,
    parse_criteria_lines,
)
from automation.src.rag.retriever import RetrievedChunk, load_policy_store, retrieve_evidence_candidates
from automation.src.rag.store import store_path_for

logger = logging.getLogger(__name__)


class CriterionResult(BaseModel):
    id: str = Field(description="The ID of the indicator being evaluated. Only contain numbers and a single dot for seperation.")
    indicator: str = Field(description="The name of the indicator being evaluated.")
    included: str = Field(description="Must be 'Yes' or 'No'.")
    evidence: Optional[List[str]] = Field(default=None, description="One or more citation tags (e.g. [\"1.3-4.0\", \"1.3-4.2\"]) of the sentences that support this indicator if included is 'Yes', otherwise null. Never transcribe sentence text here.")
    rationale: str = Field(description="Explanation of how the text addresses or fails to address this indicator.")


class PolicyEvaluationResponse(BaseModel):
    evaluation_results: List[CriterionResult]


class VerbalizedCriterionResult(CriterionResult):
    self_reported_confidence: float = Field(
        description="Your confidence that this 'included' answer is correct, from 0.0 (coin flip) to 1.0 (certain)."
    )


class VerbalizedPolicyEvaluationResponse(BaseModel):
    evaluation_results: List[VerbalizedCriterionResult]


# With `evaluation.coverage` on, an indicator answered Yes also says how it is covered. The field is only in the
# schema when asked for, so runs without it send byte-identical requests to before.
class CoverageCriterionResult(CriterionResult):
    coverage: Optional[str] = Field(
        default=None,
        description="When included is 'Yes': how the policy covers it, exactly one of the allowed coverage values. Null when included is 'No'.",
    )


class CoveragePolicyEvaluationResponse(BaseModel):
    evaluation_results: List[CoverageCriterionResult]


class VerbalizedCoverageCriterionResult(VerbalizedCriterionResult):
    coverage: Optional[str] = Field(
        default=None,
        description="When included is 'Yes': how the policy covers it, exactly one of the allowed coverage values. Null when included is 'No'.",
    )
    coverage_self_reported_confidence: Optional[float] = Field(
        default=None,
        description="Your confidence that this 'coverage' value is correct, from 0.0 (coin flip) to 1.0 (certain). Null when coverage is null.",
    )


class VerbalizedCoveragePolicyEvaluationResponse(BaseModel):
    evaluation_results: List[VerbalizedCoverageCriterionResult]


def response_model_for(confidence: ConfidenceSpec) -> type[BaseModel]:
    """Pick the response schema for the configured confidence methods and the coverage answer.

    The verbalized and coverage fields are only in the schema when they are asked for, so runs without them
    produce byte-identical requests to before the confidence module existed -- the v3 baseline stays comparable.
    """
    if confidence.has_coverage:
        if confidence.needs_verbalized:
            return VerbalizedCoveragePolicyEvaluationResponse
        return CoveragePolicyEvaluationResponse
    if confidence.needs_verbalized:
        return VerbalizedPolicyEvaluationResponse
    return PolicyEvaluationResponse


def indicator_ids_in(criteria_dir: Path) -> frozenset[str]:
    """Every indicator id the criteria files of a prompt version define: what an evaluation must answer."""
    schema = extract_schema_from_prompts(criteria_dir)
    return frozenset(indicator_id for indicators in schema.values() for indicator_id in indicators)


def validate_completeness(
    evaluation_results: dict, expected_ids: frozenset[str] | set[str] | None = None
) -> tuple[bool, list[str]]:
    """Did the judge answer every indicator? `expected_ids` are the ones in the prompt that was used; without them
    the active project's schema is the reference."""
    expected = frozenset(expected_ids) if expected_ids is not None else EXPECTED_INDICATOR_IDS
    found_ids = set(evaluation_results.keys())
    missing = sorted(expected - found_ids)
    extra = sorted(found_ids - expected)
    issues: list[str] = []
    if missing:
        issues.append(f"Missing indicators ({len(missing)}): {', '.join(missing)}")
    if extra:
        issues.append(f"Unexpected indicators ({len(extra)}): {', '.join(extra)}")
    if len(found_ids) != len(expected):
        issues.append(f"Expected {len(expected)} indicators, found {len(found_ids)}")
    return len(issues) == 0, issues


def save_report(report: dict, output_folder: str, policy_path: str) -> str:
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    basename = Path(policy_path).name.replace(".txt", ".json")
    output_path = str(Path(output_folder) / basename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_path


def finalize_and_save_report(
    final_report: dict,
    policy_path: str,
    output_folder: str,
    deployment_name: str,
    allow_partial: bool,
    expected_ids: frozenset[str] | None = None,
) -> tuple[bool, str | None]:
    is_complete, completeness_issues = validate_completeness(
        final_report["evaluation_results"], expected_ids
    )
    has_failures = bool(final_report["failed_dimensions"])

    if not is_complete:
        for issue in completeness_issues:
            print(f"Completeness issue: {issue}", file=sys.stderr)

    if is_complete and not has_failures:
        output_path = save_report(
            final_report, output_folder, policy_path
        )
        print(f"Evaluation completed successfully: {output_path}")
        return True, output_path

    if allow_partial:
        output_path = save_report(
            final_report, output_folder, policy_path
        )
        print(f"Partial evaluation saved: {output_path}", file=sys.stderr)
        return False, output_path

    print(
        "Evaluation incomplete; report not saved. Use --allow-partial to save anyway.",
        file=sys.stderr,
    )
    return False, None


@dataclass
class DimensionResult:
    criteria_file: str
    batch_evals: dict
    candidates_by_id: dict[str, list[RetrievedChunk]] | None = None
    error: str | None = None
    error_type: str | None = None
    error_details: dict | None = None
    #: How many model calls the dimension took: 1 unless its prompt was too long for one (see `_judge_in_parts`).
    prompt_parts: int = 1
    #: What had to be left out of the prompt of an indicator that was too long even alone (see `TrimReport`).
    prompt_trim: dict | None = None


def _has_eval_report(output_dir: Path, policy_name: str) -> bool:
    if not output_dir.is_dir():
        return False
    for json_path in output_dir.glob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("policy_file") == policy_name:
            return True
        basename = policy_name.replace(".txt", ".json")
        if json_path.name.endswith(f"-{basename}"):
            return True
    return False


def _failure_entry_from_exc(
    policy_file: str,
    exc: Exception,
    *,
    failed_dimensions: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict:
    if isinstance(exc, LLMCallError):
        return {
            "policy_file": policy_file,
            "failed_dimensions": failed_dimensions or [],
            "errors": errors or [str(exc)],
            "error_type": exc.error_type,
            "message": str(exc),
            "details": exc.details,
            "attempts": exc.attempts,
        }
    error_type, details = format_api_error(exc)
    return {
        "policy_file": policy_file,
        "failed_dimensions": failed_dimensions or [],
        "errors": errors or [str(exc)],
        "error_type": error_type,
        "message": str(exc),
        "details": details,
        "attempts": 1,
    }


def _failure_entry_from_report(final_report: dict) -> dict:
    errors = final_report.get("errors", [])
    message = errors[0] if errors else "evaluation incomplete"
    return {
        "policy_file": final_report["policy_file"],
        "failed_dimensions": final_report.get("failed_dimensions", []),
        "errors": errors,
        "error_type": "EvaluationError",
        "message": message,
        "details": {},
        "attempts": 1,
    }


#: A prompt that is refused as too long is rebuilt smaller at most this many times (for one indicator that cannot be split).
_MAX_SHRINKS = 2


@dataclass
class _Judged:
    batch_evals: dict
    calls: int = 0
    trim: TrimReport = field(default_factory=TrimReport)

    def absorb(self, other: "_Judged") -> None:
        self.batch_evals.update(other.batch_evals)
        self.calls += other.calls
        self.trim.dropped_candidates += other.trim.dropped_candidates
        self.trim.truncated_candidates += other.trim.truncated_candidates
        self.trim.dropped_sentences += other.trim.dropped_sentences
        self.trim.indicators.extend(other.trim.indicators)


def _split_in_two(rows: list, weights: dict[str, int]) -> tuple[list, list]:
    """Two runs of consecutive indicators whose prompts weigh about the same (never an empty side)."""
    total = sum(weights[row.id] for row in rows)
    best_index, best_gap, running = 1, None, 0
    for index in range(1, len(rows)):
        running += weights[rows[index - 1].id]
        gap = abs(total - 2 * running)
        if best_gap is None or gap < best_gap:
            best_index, best_gap = index, gap
    return rows[:best_index], rows[best_index:]


def _judge_in_parts(
    criteria_rows: list,
    candidates_by_id: dict[str, list[RetrievedChunk]],
    template_text: str,
    complete_fn: Callable[[str], dict],
    max_prompt_tokens: int,
    label: str = "",
) -> _Judged:
    """Ask the judge about a dimension's indicators, in as few calls as fit the model's context.

    One call normally carries the whole dimension. When its prompt would exceed `max_prompt_tokens` -- or the model
    refuses it as too long anyway -- the indicators are split into two runs (balanced by prompt size) and each is judged
    on its own, recursively, so nothing is dropped: every indicator keeps ALL its retrieved candidates. Only an
    indicator that is too long even alone has its candidates trimmed (lowest-ranked first, see
    `format_criteria_with_candidates`), because there is nothing left to split.

    The answers of the parts are merged by indicator id; the tags the judge cites resolve against the full lookup as
    before.
    """
    budget_chars = int(max_prompt_tokens * CHARS_PER_TOKEN)

    def render(rows: list, max_chars: int | None = None, trim: TrimReport | None = None) -> str:
        block = format_criteria_with_candidates(rows, candidates_by_id, max_chars, trim)
        return template_text.replace("{{CRITERIA_WITH_CANDIDATES}}", block)

    weights = {
        row.id: len(format_criteria_with_candidates([row], candidates_by_id)) for row in criteria_rows
    }

    def split(rows: list, why: str) -> _Judged:
        left, right = _split_in_two(rows, weights)
        logger.info(
            "%s prompt %s: judging %d + %d indicators separately",
            label, why, len(left), len(right),
        )
        judged = run(left)
        judged.absorb(run(right))
        return judged

    def run(rows: list) -> _Judged:
        prompt = render(rows)
        if len(prompt) > budget_chars and len(rows) > 1:
            return split(rows, f"is about {int(len(prompt) / CHARS_PER_TOKEN)} tokens (budget {max_prompt_tokens})")

        trim = TrimReport()
        chars = budget_chars - (len(template_text) - len("{{CRITERIA_WITH_CANDIDATES}}"))
        if len(prompt) > budget_chars:                      # one indicator, still too long: trim its candidates
            prompt = render(rows, chars, trim)
            logger.warning("%s indicator %s alone exceeds the budget; candidates trimmed", label, rows[0].id)

        for attempt in range(_MAX_SHRINKS + 1):
            try:
                result = complete_fn(prompt)
            except ContextLengthExceeded as exc:
                if len(rows) > 1:
                    return split(rows, "was refused as too long")
                if attempt == _MAX_SHRINKS:
                    raise
                # a single indicator: cut its budget by what the error says, with a margin
                ratio = (exc.limit / exc.actual) if exc.limit and exc.actual else 0.6
                chars = int(min(chars, len(prompt)) * min(ratio, 1.0) * 0.9)
                trim = TrimReport()
                prompt = render(rows, chars, trim)
                logger.warning("%s indicator %s refused as too long; retrying with a smaller prompt", label, rows[0].id)
                continue
            judged = _Judged(result.get("evaluation_results", {}), calls=1, trim=trim)
            return judged
        raise AssertionError("unreachable")                 # the loop returns or raises

    return run(criteria_rows)


def _evaluate_dimension(
    policy_path: Path,
    criteria_file: str,
    criteria_folder: Path,
    template_text: str,
    store_chunks: list[dict],
    embedder: AzureEmbedder | None,
    retrieval_config: RetrievalConfig,
    complete_fn: Callable[[str], dict],
) -> DimensionResult:
    try:
        criteria_rows = parse_criteria_lines(str(criteria_folder / criteria_file))

        if not retrieval_config.enabled:
            criteria_list_str = format_criteria_list(criteria_rows)
            final_prompt = template_text.replace("{{CRITERIA_LIST}}", criteria_list_str).replace(
                "{{POLICY_TEXT}}", policy_path.read_text(encoding="utf-8")
            )
            batch_result = complete_fn(final_prompt)
            batch_evals = batch_result.get("evaluation_results", {})
            for item in batch_evals.values():
                item["evidence_verified"] = None
                # No retrieval, so no chunk to pair the original-language text
                # against; full-policy mode is a legacy fallback anyway.
                item["evidence_original"] = None
                item["evidence_original_aligned"] = None

            logger.info("[%s] %s completed (full-policy mode)", policy_path.name, criteria_file)
            return DimensionResult(criteria_file=criteria_file, batch_evals=batch_evals)

        # Retrieval is queried with the question alone, deliberately: the rubric
        # is guidance for deciding the answer, and folding its "Code No if …"
        # exclusions into the embedding would pull retrieval toward the very
        # passages the criterion exists to rule out.
        question_embeddings = embedder.embed_texts(
            [criterion.question for criterion in criteria_rows]
        )

        candidates_by_id = {
            criterion.id: retrieve_evidence_candidates(
                query=criterion.question,
                query_embedding=query_embedding,
                store_chunks=store_chunks,
                config=retrieval_config,
            )
            for criterion, query_embedding in zip(criteria_rows, question_embeddings)
        }

        judged = _judge_in_parts(
            criteria_rows,
            candidates_by_id,
            template_text,
            complete_fn,
            retrieval_config.max_prompt_tokens,
            label=f"[{policy_path.name}] {criteria_file}",
        )
        batch_evals = judged.batch_evals
        candidate_lookup = build_candidate_lookup(candidates_by_id)
        # Original-language text is shown only for the chunks the judge cited, so only those are aligned (and not at
        # all when evidence verification is off, which is the only reader of these lookups).
        to_align = (
            cited_candidates(batch_evals, candidates_by_id)
            if retrieval_config.evidence_verification_enabled
            else {}
        )
        source_lookup = build_source_text_lookup(to_align, retrieval_config.alignment_method)
        alignment_lookup = build_source_alignment_lookup(to_align, retrieval_config.alignment_method)
        _resolve_batch_evidence(
            batch_evals,
            candidate_lookup,
            source_lookup,
            alignment_lookup,
            retrieval_config,
            policy_path.name,
        )

        logger.info("[%s] %s completed", policy_path.name, criteria_file)
        return DimensionResult(
            criteria_file=criteria_file,
            batch_evals=batch_evals,
            candidates_by_id=candidates_by_id,
            prompt_parts=judged.calls,
            prompt_trim=judged.trim.as_dict() if judged.trim.trimmed else None,
        )
    except Exception as exc:
        error_type, details = format_api_error(exc)
        logger.error(
            "[%s] %s failed [%s]: %s | status=%s",
            policy_path.name,
            criteria_file,
            error_type,
            str(exc),
            details.get("status_code", "n/a"),
        )
        return DimensionResult(
            criteria_file=criteria_file,
            batch_evals={},
            error=f"{criteria_file}: {exc}",
            error_type=error_type,
            error_details=details,
        )


def _resolve_batch_evidence(
    batch_evals: dict,
    candidate_lookup: dict[str, str],
    source_lookup: dict[str, str],
    alignment_lookup: dict[str, str],
    retrieval_config: RetrievalConfig,
    policy_name: str,
) -> None:
    """Replace each "Yes" item's evidence (a list of LLM-cited sentence tags,
    e.g. ["1.3-4.0", "1.3-4.2"]) with the real sentence text copied from
    `candidate_lookup` — the temporary tag -> text table built for this
    dimension call. Since the final evidence is copied verbatim from
    retrieved sentences rather than transcribed by the LLM, it can never
    diverge from the real document.

    Also sets `evidence_original` to the same citations' original-language
    text (from `source_lookup`), so the report can show translated and
    original text side by side, and `evidence_original_aligned` to whether
    that pairing was an exact per-sentence match versus a line/whole-chunk
    fallback (from `alignment_lookup`; see sentence_align.py). Both are
    best-effort (None when the source couldn't be aligned at all) and never
    affect `evidence`/`evidence_verified`, which stay governed by the
    translated text alone.
    """
    for cid, item in batch_evals.items():
        if item.get("included") != "Yes":
            item["evidence_verified"] = None
            item["evidence_original"] = None
            item["evidence_original_aligned"] = None
            continue

        if not retrieval_config.evidence_verification_enabled:
            item["evidence_verified"] = None
            item["evidence_original"] = None
            item["evidence_original_aligned"] = None
            continue

        citations = item.get("evidence")
        resolved_sentences = resolve_evidence_citation(citations, candidate_lookup)
        if resolved_sentences is not None:
            item["evidence"] = "\n".join(resolved_sentences)
            item["evidence_verified"] = True
            item["evidence_original"] = resolve_evidence_source_text(citations, source_lookup)
            item["evidence_original_aligned"] = resolve_evidence_original_alignment(
                citations, alignment_lookup
            )
        else:
            logger.warning(
                "[%s] %s evidence citation %r did not match any retrieved candidate; nulling.",
                policy_name,
                cid,
                citations,
            )
            item["evidence"] = None
            item["evidence_verified"] = False
            item["evidence_original"] = None
            item["evidence_original_aligned"] = None
            note = "[unresolved evidence citation removed]"
            item["rationale"] = f"{item.get('rationale', '')} {note}".strip()


def _merge_dimension_results(
    policy_path: Path,
    deployment_name: str,
    prompt_version: str,
    dimension_results: list[DimensionResult | BaseException],
) -> tuple[dict, dict[str, list[RetrievedChunk]]]:
    evaluated_at = datetime.now(timezone.utc).isoformat()
    final_report = {
        "policy_file": policy_path.name,
        "model": deployment_name,
        "evaluated_at": evaluated_at,
        "prompt_version": prompt_version,
        "completed_dimensions": [],
        "failed_dimensions": [],
        "errors": [],
        "evaluation_results": {},
    }
    all_candidates_by_id: dict[str, list[RetrievedChunk]] = {}

    for result in dimension_results:
        if isinstance(result, BaseException):
            error_type, details = format_api_error(result)
            final_report["errors"].append(f"{error_type}: {result}")
            final_report["failed_dimensions"].append("unknown")
            continue

        if result.error:
            final_report["failed_dimensions"].append(result.criteria_file)
            final_report["errors"].append(result.error)
            continue

        for key, item in result.batch_evals.items():
            if key in final_report["evaluation_results"]:
                logger.warning(
                    "[%s] ID '%s' (%s) already exists; overwriting with new outcome.",
                    policy_path.name,
                    key,
                    item.get("indicator", ""),
                )
        final_report["evaluation_results"].update(result.batch_evals)
        final_report["completed_dimensions"].append(result.criteria_file)
        if result.candidates_by_id:
            all_candidates_by_id.update(result.candidates_by_id)

    split = {r.criteria_file: r.prompt_parts for r in dimension_results if not isinstance(r, BaseException) and r.prompt_parts > 1}
    trimmed = {r.criteria_file: r.prompt_trim for r in dimension_results if not isinstance(r, BaseException) and r.prompt_trim}
    if split:
        final_report["split_dimensions"] = split          # dimension -> model calls it took (its prompt did not fit in one)
    if trimmed:
        final_report["prompt_trimmed"] = trimmed          # dimension -> what an over-long single indicator lost
    final_report["indicator_count"] = len(final_report["evaluation_results"])
    final_report["evidence_verification_failures"] = [
        key
        for key, item in final_report["evaluation_results"].items()
        if item.get("evidence_verified") is False
    ]

    # How often the original-language evidence was paired at exact sentence
    # granularity rather than falling back to the whole chunk (see
    # sentence_align.py). Denominator is indicators where any original-language
    # pairing resolved at all -- i.e. evidence_original_aligned is not None.
    aligned_flags = [
        item.get("evidence_original_aligned")
        for item in final_report["evaluation_results"].values()
    ]
    resolved_flags = [flag for flag in aligned_flags if flag is not None]
    sentence_aligned = sum(1 for flag in resolved_flags if flag)
    final_report["evidence_original_aligned_rate"] = {
        "sentence_aligned": sentence_aligned,
        "resolved": len(resolved_flags),
        "rate": round(sentence_aligned / len(resolved_flags), 4) if resolved_flags else None,
    }
    return final_report, all_candidates_by_id


def _evaluate_policy_serial(
    policy_path: Path,
    criteria_folder: Path,
    template_text: str,
    deployment_name: str,
    rag_store_dir: Path,
    embedder: AzureEmbedder | None,
    retrieval_config: RetrievalConfig,
    complete_fn: Callable[[str], dict],
) -> tuple[dict, dict[str, list[RetrievedChunk]]]:
    store_chunks = (
        load_policy_store(store_path_for(rag_store_dir, policy_path.name))
        if retrieval_config.enabled
        else []
    )
    dimension_results = [
        _evaluate_dimension(
            policy_path,
            criteria_file,
            criteria_folder,
            template_text,
            store_chunks,
            embedder,
            retrieval_config,
            complete_fn,
        )
        for criteria_file in list_criteria_files(criteria_folder)
    ]
    return _merge_dimension_results(policy_path, deployment_name, criteria_folder.name, dimension_results)


class _PolicyJob:
    """One policy being evaluated by the flat scheduler: its dimension results arrive from different threads."""

    def __init__(self, policy_path: Path, n_dimensions: int) -> None:
        self.policy_path = policy_path
        self.results: list = [None] * n_dimensions
        self.remaining = n_dimensions
        self.lock = threading.Lock()
        self.store_chunks: list[dict] | None = None
        self.load_error: BaseException | None = None


def _evaluate_policies_flat(
    policy_files: list[Path],
    criteria_folder: Path,
    template_text: str,
    deployment_name: str,
    rag_store_dir: Path,
    embedder: AzureEmbedder | None,
    retrieval_config: RetrievalConfig,
    complete_fn: Callable[[str], dict],
    limiter: ConcurrencyLimiter,
    settle: Callable[[Path, object], None],
) -> None:
    """Evaluate every (policy, dimension) pair on ONE pool of `limiter.max_workers` threads.

    Evaluating policies in parallel and, inside each policy, its dimensions in parallel nests two pools of
    `max_workers`: up to `max_workers ** 2` threads at once, all retrieving, reranking, aligning and encoding
    together. That exhausted the machine's memory (PyTorch crashed at ~43 GB). Here the total never exceeds
    `max_workers`, and the tasks are queued policy by policy, so policies finish one after another instead of
    all crawling together; the tail of the queue keeps every thread busy.

    The thread that completes a policy's last dimension merges it and calls `settle(policy_path, outcome)`
    (`outcome` = `(final_report, candidates_by_id)` or the exception), so a report is saved as soon as its
    policy is done. The policy's retrieval store is loaded when its first dimension starts and dropped when
    it settles, which keeps only a couple of stores in memory.
    """
    criteria_files = list_criteria_files(criteria_folder)      # the dimensions are whatever the prompt version has
    jobs = {path: _PolicyJob(path, len(criteria_files)) for path in policy_files}

    def store_for(job: _PolicyJob) -> list[dict]:
        with job.lock:
            if job.store_chunks is None and job.load_error is None:
                try:
                    job.store_chunks = (
                        load_policy_store(store_path_for(rag_store_dir, job.policy_path.name))
                        if retrieval_config.enabled
                        else []
                    )
                except BaseException as exc:
                    job.load_error = exc
            if job.load_error is not None:
                raise job.load_error
            return job.store_chunks

    def run_one(job: _PolicyJob, index: int, criteria_file: str) -> None:
        try:
            result = _evaluate_dimension(
                job.policy_path,
                criteria_file,
                criteria_folder,
                template_text,
                store_for(job),
                embedder,
                retrieval_config,
                complete_fn,
            )
        except BaseException as exc:
            result = exc
        with job.lock:
            job.results[index] = result
            job.remaining -= 1
            last = job.remaining == 0
        if not last:
            return
        if job.load_error is not None:
            outcome: object = job.load_error
        else:
            try:
                outcome = _merge_dimension_results(
                    job.policy_path, deployment_name, criteria_folder.name, job.results
                )
            except BaseException as exc:
                outcome = exc
        job.store_chunks = None
        settle(job.policy_path, outcome)

    logger.info(
        "Evaluating %d policies x %d dimensions on %d threads", len(policy_files), len(criteria_files), limiter.max_workers
    )
    tasks = [
        lambda job=job, index=index, cf=cf: run_one(job, index, cf)
        for job in jobs.values()
        for index, cf in enumerate(criteria_files)
    ]
    limiter.run_parallel(tasks)


def _write_candidates_file(
    rag_candidates_dir: Path,
    policy_filename: str,
    candidates_by_id: dict[str, list[RetrievedChunk]],
) -> None:
    """Persist the full RAG candidate list per indicator (not just the one the
    judge cited) to `mid_product/rag_candidates/<stem>.json`, so the
    discrepancy_diagnosis step can read them later without recomputing
    retrieval.
    """
    rag_candidates_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_file": policy_filename,
        "candidates": {
            indicator_id: [
                {"chunk_id": c.chunk_id, "text": c.text, "score": c.score} for c in candidates
            ]
            for indicator_id, candidates in candidates_by_id.items()
        },
    }
    path = rag_candidates_dir / f"{Path(policy_filename).stem}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _process_policy_result(
    policy_path: Path,
    final_report: dict,
    output_dir: Path,
    deployment_name: str,
    allow_partial: bool,
    candidates_by_id: dict[str, list[RetrievedChunk]],
    rag_candidates_dir: Path,
    expected_ids: frozenset[str] | None = None,
) -> tuple[bool, str | None, bool]:
    success, output_path = finalize_and_save_report(
        final_report=final_report,
        policy_path=str(policy_path),
        output_folder=str(output_dir),
        deployment_name=deployment_name,
        allow_partial=allow_partial,
        expected_ids=expected_ids,
    )
    if output_path:
        _write_candidates_file(rag_candidates_dir, policy_path.name, candidates_by_id)
    policy_failed = not success
    return success, output_path, policy_failed


def _warm_up_aligner(method: str) -> None:
    """Load the alignment model once, here, instead of letting the first worker threads race to do it.
    If it cannot be loaded the evaluation still runs: evidence alignment falls back to the heuristic aligner."""
    try:
        warm_up(method)
    except AlignerUnavailable as exc:
        logger.warning("alignment model not available (%s); evidence alignment will use the heuristic", exc)


def run_evaluation_step(
    config: ResolvedPipelineConfig,
    wrapper: AzureLLMWrapper,
    embedder: AzureEmbedder | None,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    allow_partial: bool = False,
    force: bool = False,
) -> dict:
    run_dir = get_output_dir()
    policy_dir = resolve_results_dir(run_dir, "translation")
    rag_store_dir = resolve_mid_product_dir(run_dir, "rag_store")
    output_dir = resolve_results_dir(run_dir, "evaluation")
    rag_candidates_dir = resolve_mid_product_dir(run_dir, "rag_candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_files = filter_policy_files(
        policy_dir, small_scale, config.paths.small_scale_stems
    )
    if not policy_files:
        raise FileNotFoundError(f"No policy files to evaluate in {policy_dir}")

    ok, issues = validate_criteria_files(config.evaluation_criteria_dir)
    if not ok:
        raise ValueError(
            f"The criteria files in {config.evaluation_criteria_dir} are malformed; fix them before evaluating "
            "(each costs model calls): " + " | ".join(issues[:5])
        )
    template_text = config.evaluation_template_path.read_text(encoding="utf-8")

    total_candidates = len(policy_files)
    skipped = 0
    pending_files: list[Path] = []
    for policy_path in policy_files:
        if not force and _has_eval_report(output_dir, policy_path.name):
            skipped += 1
            logger.info("[%s] skipped (eval report exists)", policy_path.name)
        else:
            pending_files.append(policy_path)
    policy_files = pending_files

    response_model = response_model_for(wrapper.confidence)

    def complete_fn(prompt: str) -> dict:
        return wrapper.complete_structured(prompt, response_model)

    deployment_name = wrapper.profile.deployment
    use_serial = not limiter.enabled or limiter.max_workers == 1
    start = time.time()
    saved_paths: list[str] = []
    failed_policies: list[str] = []

    if not policy_files:
        elapsed = round(time.time() - start, 2)
        return {
            "counts": {
                "total": total_candidates,
                "succeeded": 0,
                "failed": 0,
                "skipped": skipped,
                "saved_reports": 0,
            },
            "failed_policies": failed_policies,
            "elapsed_s": elapsed,
            "token_usage": dict(wrapper.token_usage),
            "output_dir": str(output_dir),
        }

    bookkeeping_lock = threading.Lock()
    # An evaluation is complete when it answers every indicator of the prompt version in use.
    expected_ids = indicator_ids_in(config.evaluation_criteria_dir)

    def settle(policy_path: Path, outcome) -> None:
        """Save one policy's report the moment its evaluation is done, and record how it went.

        `outcome` is the exception the evaluation raised or its `(final_report, candidates_by_id)`.
        Runs in the worker thread that evaluated the policy, so a finished policy is on disk while the others are
        still running. The files are per policy; only the shared lists and the failure log need the lock.
        """
        if isinstance(outcome, BaseException):
            entry = _failure_entry_from_exc(policy_path.name, outcome)
            logger.error(
                "[%s] evaluation failed [%s]: %s", policy_path.name, entry["error_type"], entry["message"]
            )
            with bookkeeping_lock:
                record_failure("evaluation", entry)
                failed_policies.append(policy_path.name)
            return

        final_report, candidates_by_id = outcome
        success, output_path, policy_failed = _process_policy_result(
            policy_path,
            final_report,
            output_dir,
            deployment_name,
            allow_partial,
            candidates_by_id,
            rag_candidates_dir,
            expected_ids,
        )
        with bookkeeping_lock:
            if output_path:
                saved_paths.append(output_path)
                print(f"Saved evaluation: {Path(output_path).name}")
            if policy_failed:
                failed_policies.append(policy_path.name)
                record_failure("evaluation", _failure_entry_from_report(final_report))
            elif success:
                clear_failure("evaluation", policy_path.name)

    if use_serial:
        for index, policy_path in enumerate(policy_files, 1):
            print("\n" + "=" * 50)
            print(f"Policy [{index}/{len(policy_files)}]: {policy_path.name}")
            print("=" * 50)

            try:
                outcome = _evaluate_policy_serial(
                    policy_path,
                    config.evaluation_criteria_dir,
                    template_text,
                    deployment_name,
                    rag_store_dir,
                    embedder,
                    config.retrieval,
                    complete_fn,
                )
            except Exception as exc:
                outcome = exc
            settle(policy_path, outcome)
    else:

        # One pool over every (policy, dimension) pair; each policy is saved by the thread that finishes it.
        _evaluate_policies_flat(
            policy_files,
            config.evaluation_criteria_dir,
            template_text,
            deployment_name,
            rag_store_dir,
            embedder,
            config.retrieval,
            complete_fn,
            limiter,
            settle,
        )
        order = {p.name: i for i, p in enumerate(policy_files)}
        failed_policies.sort(key=lambda name: order.get(name, len(order)))

    elapsed = round(time.time() - start, 2)
    total = total_candidates
    succeeded = len(policy_files) - len(failed_policies)

    return {
        "counts": {
            "total": total,
            "succeeded": succeeded,
            "failed": len(failed_policies),
            "skipped": skipped,
            "saved_reports": len(saved_paths),
        },
        "failed_policies": failed_policies,
        "elapsed_s": elapsed,
        "token_usage": dict(wrapper.token_usage),
        "output_dir": str(output_dir),
    }
