"""Stage 1.1 - input processing.

    raw_files/ (PDF, HTML, TXT)
      -> extract            -> extracted_txt/<stem>__pdf|html|txt.txt
      -> quality check + fix -> cleaned_txt/<stem>.txt
      -> input_report.json   (eliminated files, PDFs that need OCR)

OCR is done outside the pipeline: put the OCRed text of a PDF in
``extracted_txt/`` as ``<stem>__ocr.txt`` and run the quality check again.

Run with ``python -m automation.src.input_process --help``.
The earlier ``scripts/`` folder is the finished cleaning pipeline of another
project (GlobalLaws), kept for reference only; nothing here imports it.
"""
