from deep_translator import GoogleTranslator
from pathlib import Path


class PolicyTranslator:
    def __init__(self, source_lang="auto", target_lang="en"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translator = GoogleTranslator(
            source=source_lang,
            target=target_lang
        )

    def translate_line(self, line: str) -> str:
        """
        Translate a single line safely.
        """
        line = line.strip()

        if not line:
            return "\n"

        try:
            translated = self.translator.translate(line)

            # ensure string output
            if translated is None:
                return line + "\n"

            return str(translated) + "\n"

        except Exception as e:
            print(f"Error translating line: {line}\n{e}")
            return line + "\n"

    def translate_file(self, input_path, output_path):
        """
        Translate an entire file line-by-line.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        with input_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        translated_lines = [
            self.translate_line(line)
            for line in lines
        ]

        output_path.write_text(
            "".join(translated_lines),
            encoding="utf-8"
        )

        print(f"Done: {output_path}")

    def translate_files(self, file_pairs):
        """
        Batch translate multiple files.

        file_pairs: list of (input_path, output_path)
        """
        for inp, out in file_pairs:
            self.translate_file(inp, out)