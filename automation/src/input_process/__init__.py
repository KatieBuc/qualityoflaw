"""Stage 1.1 - input processing.

    raw/raw_files/ (PDF, HTML, TXT)
      -> extract             -> raw/extracted_txt/<stem>.txt
      -> quality check + fix -> preprocessed/cleaned_txt/<stem>.txt
      -> raw/input_process.log   (what happened to each file; WARNING = failed the quality check)

File names carry no markers: the same <stem> is used everywhere and the folder says where a policy
stands (cleaned_txt = done, extracted_txt = waiting, neither = extract it). OCR is done outside the
pipeline: save the OCRed text over ``extracted_txt/<stem>.txt`` and run the quality check again.

Run with ``python -m automation.src.input_process --help``.
The earlier ``scripts/`` folder is the finished cleaning pipeline of another
project (GlobalLaws), kept for reference only; nothing here imports it.
"""
