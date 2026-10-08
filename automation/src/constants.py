from pathlib import Path

AUTOMATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUTOMATION_ROOT.parent
DEFAULT_MODEL_CONFIG = AUTOMATION_ROOT / "config" / "model_config.yaml"
DEFAULT_PIPELINE_CONFIG = AUTOMATION_ROOT / "config" / "pipeline_config.yaml"

# Static data layout (see `paths.py`). The active project is a config choice
# (`project:` in pipeline_config.yaml); DEFAULT_PROJECT is only the fallback.
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_PROJECT = "indonesia"
RAW_DIRNAME = "raw"
PREPROCESSED_DIRNAME = "preprocessed"
PROCESSED_DIRNAME = "processed"
AUTOMATION_DIRNAME = "automation"
CLEANED_MARKDOWN_DIRNAME = "cleaned_markdown"
TRANSLATION_MARKDOWN_DIRNAME = "translation_markdown"
CORRECTIONS_DIRNAME = "corrections"
MANUAL_OVERWRITES_FILENAME = "manual_overwrites.yaml"
GOLDEN_CSV_FILENAME = "long_policy_encoding.csv"
INDEX_SCHEMA_RELPATH = "mapping/index_schema.yaml"
PROMPTS_ROOT = AUTOMATION_ROOT / "prompts"
CHUNKING_FALLBACK_PROMPT = PROMPTS_ROOT / "translation" / "chunking" / "fallback_prompt.txt"
MARKDOWN_FALLBACK_PROMPT = (
    PROMPTS_ROOT / "translation" / "chunking" / "fallback_prompt_md.txt"
)

# The default `--small-scale` benchmark set, on the Indonesian localpolicies
# corpus. It is only a fallback: the set actually used comes from
# `paths.small_scale_stems` in pipeline_config.yaml, so it travels with
# `markdown_input_dir` / `markdown_input_suffix` when the pipeline is pointed
# at another corpus (see `config_loader.PipelinePaths.small_scale_stems`).
# The raw-text translation path (`resolve_input_files`) stays pinned to this
# list, since it only ever reads the project's raw folder.
SMALL_SCALE_FILES = [
    "ACEH_BIREUEN.txt",
    "NUSA_TENGGARA_BARAT_LOMBOK_TENGAH_v2.txt",
    # "NUSA_TENGGARA_BARAT_DOMPU.txt",
    "RIAU_PEKANBARU_KOTA.txt",
]
DEFAULT_SMALL_SCALE_STEMS = tuple(Path(name).stem for name in SMALL_SCALE_FILES)

# The Markdown path is the default: the project's `cleaned_markdown/` corpus
# is already cleaned and structured, so translation reads and writes Markdown and `md_to_text`
# renders the plain-text artifact everything downstream reads.
DEFAULT_STEPS = (
    "translation_md",
    "md_to_text",
    "storage",
    "evaluation",
)

# translation_qa_md, comparison and discrepancy_diagnosis stay out of the
# default chain (extra QA / comparison against golden labels / diagnosis
# isn't wanted on every run) but remain valid --steps values so they can
# still be run explicitly, e.g. `--steps comparison`.
OPTIONAL_STEPS = ("translation_qa_md", "comparison", "discrepancy_diagnosis")

# The raw-OCR-text path (`translation`, `translation_qa`, `markdown`) is kept
# runnable via --steps so the two can be compared on the same corpus, but is
# no longer part of the default chain.
LEGACY_TEXT_STEPS = ("translation", "translation_qa", "markdown")

VALID_STEPS = DEFAULT_STEPS + OPTIONAL_STEPS + LEGACY_TEXT_STEPS

STRUCTURE_HINT = (
    "Preserve the original paragraph and line structure where possible."
)

# The Markdown path's equivalent. Structure is explicit in the input here, so
# the instruction can be exact rather than a best-effort hint.
MARKDOWN_STRUCTURE_HINT = (
    "Reproduce the Markdown structure exactly: every heading keeps its own "
    "level (the same number of leading # characters), and every list marker, "
    "number and clause marker is preserved. Do not add, remove, merge or "
    "reorder any heading, list item or paragraph. Do not wrap the output in a "
    "code fence."
)
