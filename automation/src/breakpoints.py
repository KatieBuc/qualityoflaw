"""Pipeline breakpoints: places where a person has to act before the run can go on.

A step raises `PipelineBreakpoint` with a short account of what happened, what to do next and
which docs explain it. `run_pipeline` prints it, marks the run ``paused`` and stops cleanly
(this is not a failure). Running the same command again resumes: every step skips finished work.
"""

from dataclasses import dataclass, field

DOCS_PIPELINE = "automation/docs/pipeline.md"
DOCS_INPUT_PROCESS = "automation/docs/input_process.md"


@dataclass
class PipelineBreakpoint(Exception):
    step: str
    title: str
    details: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=lambda: [DOCS_PIPELINE])
    #: Appended to the next steps: how to go on without doing the work.
    override: str = ""

    def __str__(self) -> str:
        return self.title

    def format(self) -> str:
        lines = ["", "=" * 72, f"PAUSED at step '{self.step}': {self.title}", "=" * 72]
        lines += [f"  {d}" for d in self.details]
        if self.next_steps:
            lines += ["", "What to do next:"]
            lines += [f"  {i}. {text}" for i, text in enumerate(self.next_steps, start=1)]
        if self.override:
            lines += ["", f"To go on without doing this: {self.override}"]
        lines += ["", "Docs: " + ", ".join(self.docs), "Then run the same command again; finished work is skipped."]
        return "\n".join(lines)

    def as_record(self) -> dict:
        return {"step": self.step, "title": self.title, "next_steps": self.next_steps, "docs": self.docs}
