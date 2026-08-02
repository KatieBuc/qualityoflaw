from pathlib import Path

AUTOMATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUTOMATION_ROOT.parent
DEFAULT_MODEL_CONFIG = AUTOMATION_ROOT / "config" / "model_config.yaml"
DEFAULT_PIPELINE_CONFIG = AUTOMATION_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "automation"
PROMPTS_ROOT = AUTOMATION_ROOT / "prompts"
CHUNKING_FALLBACK_PROMPT = PROMPTS_ROOT / "translation" / "chunking" / "fallback_prompt.txt"
DEFAULT_MANUAL_OVERWRITES = PROJECT_ROOT / "data" / "corrections" / "manual_overwrites.yaml"

SMALL_SCALE_FILES = [
    "ACEH_BIREUEN.txt",
    "LAMPUNG_LAMPUNG_TIMUR.txt",
    "NUSA_TENGGARA_TIMUR_TIMOR_TENGAH_UTARA.txt",
    "SUMATERA_BARAT_PADANG_PARIAMAN.txt",
    "JAWA_TENGAH_SEMARANG.txt",
    # "KALIMANTAN_TENGAH_SUKAMARA.txt"
]

DEFAULT_STEPS = ("translation", "markdown", "storage", "evaluation", "comparison", "discrepancy_diagnosis")
VALID_STEPS = DEFAULT_STEPS

STRUCTURE_HINT = (
    "Preserve the original paragraph and line structure where possible."
)
