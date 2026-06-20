import time
import os
from googletrans import Translator
import asyncio
import argparse
import sys

async def translate_file(input_path, output_path, target_lang='en'):
    if not os.path.exists(input_path):
        print(f"File doesn't exisy '{input_path}'")
        return

    translator = Translator()
    print(f"Translating... \nInput file: {input_path}\nTarget Lang: {target_lang}\n" + "-"*30)

    with open(input_path, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)
    sys.stdout.write(f"\rTotal Lines: {total_lines}\n")

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                stripped_line = line.strip()

                if not stripped_line:
                    outfile.write('\n')
                    continue
                else:
                    try:
                        translated = await translator.translate(stripped_line, dest=target_lang)
                        outfile.write(translated.text + '\n')
                        time.sleep(0.5)

                    except Exception as e:
                        print(f"Error in line {line_num} : {e}")
                        outfile.write(line)

                percentage = (line_num / total_lines) * 100
                sys.stdout.write(f"\r{line_num} / {total_lines} ({percentage:.1f}%)")
                sys.stdout.flush()

        print("-"*30 + f"\nTrnslation Finished: {output_path}")

    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translting with Google Translate for TXT file")

    parser.add_argument('-i', '--input', required=True, type=str, help="Input filepath")
    parser.add_argument('-o', '--output', required=True, type=str, help="Output filepath")
    parser.add_argument('-l', '--lang', type=str, default='en', help="Target language code")

    args = parser.get_args() if hasattr(parser, 'get_args') else parser.parse_args()

    asyncio.run(translate_file(args.input, args.output, args.lang))