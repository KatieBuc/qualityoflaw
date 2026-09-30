"""Storage stage: chunk translated policy files and embed each chunk.

Writes one JSON store file per policy under
`data/automation/<run_id>/mid_product/rag_store/<stem>.json` — this
one-file-per-policy layout is what guarantees retrieval never mixes evidence
across policies.

Two ways to get chunk text for a policy, tried in order:
1. Reuse the translation step's own chunk boundaries and per-chunk English
   translations, if
   `data/automation/<run_id>/mid_product/chunks/<stem>.chunks.json`
   exists and is valid (see `_load_translation_chunks`). These boundaries were
   decided on the source-language text (reliable BAB/Pasal/Bagian/Paragraf
   markers), so they're preferred over re-detecting structure in translated
   English.
2. Otherwise, chunk the translated (English) file directly using the English
   marker regex (Chapter/Article/Part/Paragraph).
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from automation.src.chunking import STRUCTURE_MARKER_EN_RE, chunk_policy_text, fallback_split
from automation.src.concurrency import ConcurrencyLimiter
from automation.src.config_loader import (
    ResolvedPipelineConfig,
    get_run_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.failure_log import clear_failure, record_failure
from automation.src.policy_files import filter_policy_files
from automation.src.rag.embedder import AzureEmbedder
from automation.src.common import chunks_artifact_path

logger = logging.getLogger(__name__)

# Azure's embedding models cap input at 8192 tokens. Translation chunking's
# safe_limit (chars, ~32000) is sized for the translation LLM's context, not
# this — and legal-citation-heavy text (lots of digits/punctuation) can
# tokenize less efficiently than prose, so a chunk under safe_limit can still
# exceed 8192 tokens. BPE tokenizers never split a single ASCII character
# into more than one token, so staying well under this many *characters*
# guarantees staying under the token limit regardless of tokenization
# density, without needing a tokenizer dependency to measure exactly.
EMBEDDING_SAFE_CHAR_LIMIT = 6000


def _split_oversized_for_embedding(chunk_dicts: list[dict]) -> list[dict]:
    """Sub-split any chunk whose text could exceed the embedding model's
    token limit, independent of the translation step's own chunk boundaries.

    `chunk_id` is renumbered sequentially across the result — retrieval only
    ever treats it as an index into this one policy's own rag_store chunk
    list (see retriever.py), never against chunks.json, so re-splitting here
    doesn't affect anything downstream.

    A sub-split keeps its parent's whole `source_text` rather than splitting
    it too — the split is decided purely on translated-text length, and
    slicing the original-language text to match is not sound. It's a rare
    case (chunk over ~6000 chars) so the whole-section original text is a
    fine substitute for a wholly accurate sub-excerpt.
    """
    expanded: list[dict] = []
    for chunk in chunk_dicts:
        text = chunk["text"]
        if len(text) <= EMBEDDING_SAFE_CHAR_LIMIT:
            expanded.append(chunk)
            continue
        for sub_text, _context in fallback_split(text, safe_limit=EMBEDDING_SAFE_CHAR_LIMIT):
            expanded.append({**chunk, "text": sub_text})

    return [{**c, "chunk_id": i} for i, c in enumerate(expanded)]


@dataclass
class StoreResult:
    filename: str
    status: str
    chunk_count: int = 0
    error: str | None = None
    source: str = "translated_text"  # "translation_chunks" or "translated_text"


def store_path_for(store_dir: Path, filename: str) -> Path:
    return store_dir / f"{Path(filename).stem}.json"


def _load_translation_chunks(chunks_path: Path) -> list[dict] | None:
    """Load and validate the translation step's chunks.json, if usable.

    Returns None (triggering the direct-chunking fallback) when the file is
    missing, malformed, empty, or missing a translated_text for any chunk —
    e.g. because translation.chunking was disabled for this run.
    """
    if not chunks_path.exists():
        return None
    try:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(c, dict) and c.get("translated_text") for c in data):
        return None

    return [
        {
            "chunk_id": i,
            "section_id": c["section_id"],
            "chunk_index": c["chunk_index"],
            "type": c["type"],
            "text": c["translated_text"],
            # Original-language text for the same section, so evidence can be
            # shown side-by-side with its translation. Empty when the
            # translation step couldn't align source to translated sections
            # (see markdown/to_text.py:build_retrieval_chunks).
            "source_text": c.get("text") or None,
        }
        for i, c in enumerate(data)
    ]


def _embed_policy_file(
    input_path: Path,
    store_path: Path,
    chunks_path: Path,
    embedder: AzureEmbedder,
    safe_limit: int,
) -> StoreResult:
    filename = input_path.name
    try:
        chunk_dicts = _load_translation_chunks(chunks_path)
        if chunk_dicts is not None:
            source = "translation_chunks"
            logger.info(
                "[%s] reusing %d chunk(s) from the translation step", filename, len(chunk_dicts)
            )
        else:
            source = "translated_text"
            text = input_path.read_text(encoding="utf-8")
            chunks = chunk_policy_text(text, safe_limit=safe_limit, marker_re=STRUCTURE_MARKER_EN_RE)
            if not chunks:
                raise ValueError("no chunks produced")
            chunk_dicts = [
                {
                    "chunk_id": i,
                    "section_id": c.section_id,
                    "chunk_index": c.chunk_index,
                    "type": c.type,
                    "text": c.text,
                }
                for i, c in enumerate(chunks)
            ]

        chunk_dicts = _split_oversized_for_embedding(chunk_dicts)
        vectors = embedder.embed_texts([c["text"] for c in chunk_dicts])
        payload = {
            "policy_file": filename,
            "chunks": [{**c, "embedding": v} for c, v in zip(chunk_dicts, vectors)],
        }
        store_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info("[%s] stored %d chunks (source=%s)", filename, len(chunk_dicts), source)
        return StoreResult(filename=filename, status="succeeded", chunk_count=len(chunk_dicts), source=source)
    except Exception as exc:
        logger.error("[%s] storage failed: %s", filename, exc)
        return StoreResult(filename=filename, status="failed", error=str(exc))


def run_storage_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    embedder: AzureEmbedder,
    *,
    limiter: ConcurrencyLimiter,
    small_scale: bool = False,
    force: bool = False,
) -> dict:
    run_dir = get_run_dir(run_id)
    policy_dir = resolve_results_dir(run_dir, "translation")
    chunks_dir = resolve_mid_product_dir(run_dir, "chunks")
    store_dir = resolve_mid_product_dir(run_dir, "rag_store")
    store_dir.mkdir(parents=True, exist_ok=True)

    policy_files = filter_policy_files(
        policy_dir, small_scale, config.paths.small_scale_stems
    )
    if not policy_files:
        raise FileNotFoundError(f"No policy files to embed in {policy_dir}")

    counts = {"total": len(policy_files), "succeeded": 0, "skipped": 0, "failed": 0}
    failed_files: list[dict] = []
    start = time.time()

    pending: list[tuple[Path, Path, Path]] = []
    for input_path in policy_files:
        store_path = store_path_for(store_dir, input_path.name)
        if store_path.exists() and not force:
            counts["skipped"] += 1
            logger.info("[%s] skipped (store exists)", input_path.name)
            continue
        pending.append((input_path, store_path, chunks_artifact_path(chunks_dir, input_path.name)))

    if pending:
        tasks = [
            lambda inp=inp, out=out, chunks_path=chunks_path: _embed_policy_file(
                inp, out, chunks_path, embedder, config.chunking.safe_limit
            )
            for inp, out, chunks_path in pending
        ]
        results = limiter.run_parallel(tasks)
        for result in results:
            if isinstance(result, BaseException):
                counts["failed"] += 1
                entry = {"filename": "unknown", "error_type": type(result).__name__, "message": str(result)}
                failed_files.append(entry)
                record_failure(run_id, "storage", entry)
                logger.error("Storage task failed: %s", result)
                continue
            if result.status == "succeeded":
                counts["succeeded"] += 1
                clear_failure(run_id, "storage", result.filename)
            else:
                counts["failed"] += 1
                entry = {
                    "filename": result.filename,
                    "error_type": "StorageError",
                    "message": result.error or "unknown error",
                }
                failed_files.append(entry)
                record_failure(run_id, "storage", entry)

    elapsed = round(time.time() - start, 2)
    return {
        "counts": counts,
        "failed_files": failed_files,
        "elapsed_s": elapsed,
        "output_dir": str(store_dir),
    }
