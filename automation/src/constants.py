from pathlib import Path

AUTOMATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUTOMATION_ROOT.parent
DEFAULT_MODEL_CONFIG = AUTOMATION_ROOT / "config" / "model_config.yaml"
DEFAULT_PIPELINE_CONFIG = AUTOMATION_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "automation"
PROMPTS_ROOT = AUTOMATION_ROOT / "prompts"
CHUNKING_FALLBACK_PROMPT = PROMPTS_ROOT / "translation" / "chunking" / "fallback_prompt.txt"
MARKDOWN_FALLBACK_PROMPT = (
    PROMPTS_ROOT / "translation" / "chunking" / "fallback_prompt_md.txt"
)
DEFAULT_MANUAL_OVERWRITES = PROJECT_ROOT / "data" / "corrections" / "manual_overwrites.yaml"

# The raw-OCR-text path's benchmark four -- `resolve_input_files` (.txt-only)
# and `diagnose.py` (matches these exact filenames against the golden
# dataframe) both need real files from `data/raw/localpolicies`, so this
# stays pinned to that corpus regardless of which corpus `translation_md`
# is pointed at. See `MARKDOWN_SMALL_SCALE_FILES` for the Markdown path's
# equivalent, which can point at a different corpus's own benchmark files.
SMALL_SCALE_FILES = [
    "ACEH_BIREUEN.txt",
    "NUSA_TENGGARA_BARAT_LOMBOK_TENGAH_v2.txt",
    # "NUSA_TENGGARA_BARAT_DOMPU.txt",
    "RIAU_PEKANBARU_KOTA.txt",
]

# The Markdown path's `--small-scale` benchmark set. Deliberately separate
# from `SMALL_SCALE_FILES`: the two paths can point at entirely different
# corpora (e.g. the Indonesian localpolicies corpus vs. globallaws_markdown),
# whose benchmark documents share no filenames at all. Only the stem is used
# (`markdown.policy_files.SMALL_SCALE_STEMS`), so the extension here is
# cosmetic -- match whatever `markdown_input_suffix` the corpus actually uses.
MARKDOWN_SMALL_SCALE_FILES = [
    "21. Bosnia and Herzegovina - DomViol - Bosnian (b).md",
    "94. Latvia - DomViol - Latvian (a).md",
    "121. New Zealand - DomViol - English.md",
]

# The Markdown path is the default: the corpus in
# `data/processed/localpolicies/cleaned_markdown/` is already cleaned and
# structured, so translation reads and writes Markdown and `md_to_text`
# renders the plain-text artifact everything downstream reads.
DEFAULT_STEPS = (
    "translation_md",
    "md_to_text",
    "storage",
    "evaluation",
    "comparison",
)

# translation_qa_md and discrepancy_diagnosis stay out of the default chain
# (extra QA pass / diagnosis cost isn't wanted on every run) but remain valid
# --steps values so they can still be run explicitly.
OPTIONAL_STEPS = ("translation_qa_md", "discrepancy_diagnosis")

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
