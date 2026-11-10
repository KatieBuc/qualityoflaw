#!/usr/bin/env python3
"""
a5_remove_markers.py — Route A5: remove a specific residual source marker
(leftover raw HTML tag, or source-site branding text) via exact string
match, found by reading the full file.

Reads from globallaws/, writes to globallaws_cleaned/.

USAGE
-----
    python a5_remove_markers.py
"""
import os

SRC = r"C:\myProject\Foundation_model\globallaws"
OUT = r"C:\myProject\Foundation_model\globallaws_cleaned"
os.makedirs(OUT, exist_ok=True)


def read(fn):
    with open(os.path.join(SRC, fn), encoding="utf-8") as f:
        return f.read()


def write(fn, text):
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def run():
    fn = "137. Puerto Rico - DomViol - Spanish.txt"
    t = read(fn)
    t = t.replace("<english>\n\n", "", 1)
    t = t.replace("{Ir a Tabla de Contenido}\n", "", 1)
    write(fn, t)
    print(fn, "-> removed leftover <english> webpage marker and {Ir a Tabla de Contenido} nav link")

    fn = "146. Serbia - DomViol - Bosnian (b).txt"
    t = read(fn)
    marker = ("BUDITE NA PRAVNOJ STRANI\nonline@paragraf.rs\nwww.paragraf.rs\n\n"
              "Preuzeto iz elektronske pravne baze Paragraf Lex\n")
    t = t.replace(marker, "", 1)
    write(fn, t)
    print(fn, "-> removed leftover source-site branding text (Paragraf Lex)")


if __name__ == "__main__":
    run()
