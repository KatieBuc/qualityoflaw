"""Stage 1.1 - input processing.

    raw/raw_files/ (PDF, HTML, TXT)
      -> extract             -> preprocessed/cleaned_txt/<stem>.txt   (every text; the one folder later steps read)
      -> quality check + fix -> the same files, fixed in place when they pass
      -> raw/input_process.log   (PASS / NEEDS_OCR / ELIMINATED per file; WARNING = failed the quality check)

File names carry no markers: the same <stem> is used everywhere. A policy that already has its
``cleaned_txt/<stem>.txt`` is not extracted again, so OCR text or a fix saved over that file is kept and
checked again by the next quality check. OCR is done outside the pipeline.

Run with ``python -m automation.src.input_process --help``.
The earlier ``scripts/`` folder is the finished cleaning pipeline of another
project (GlobalLaws), kept for reference only; nothing here imports it.
"""
