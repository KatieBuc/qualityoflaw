# -*- coding: utf-8 -*-
r"""
Convert the cleaned statutory .txt files in ``globallaws_cleaned/`` into structured Markdown.

Goal
----
Turn flat, extraction-damaged statute text into Markdown whose headings mirror the statute's own hierarchy and whose
body is readable prose, without paraphrasing or inventing content.

Pipeline (per document)
-----------------------
1. preprocess      remove web-navigation chrome, running headers / footers / page numbers, and turn footnotes into
                   Markdown footnotes -- before heading detection, because all of these can look like headings.
2. analyse         decide which heading levels (Book / Part / Title / Chapter / Section / Sub-section / Article /
                   paragraph) a file really uses, in ~30 languages and several scripts, plus bare ``1. 2. 3.`` sections
                   used by common-law acts.  Levels must be anchored at the start of lines so that cross-references
                   ("pursuant to section 22") never create a level.
3. convert         emit ATX headings (``#`` .. ``######``) for the detected labels, reject citations and table-of-contents
                   entries, and format the text between headings (format_body): re-joined paragraphs, nested lists,
                   pipe tables, TOC lists.
4. index           write ``_INDEX.md`` listing every converted file and its detected hierarchy.

Heading depth
-------------
``--depth relative``  depth follows the levels the file uses (``#`` title, ``##`` outermost level, ...).
``--depth absolute``  fixed depths shared by all files: Book/Part/Title ``##``, Chapter ``###``, Section ``####``,
                      Article ``#####``.

Files with no usable structure (prose summaries, technical notes, web-portal dumps) are skipped; see
``SKIP_SUBSTRINGS`` and the "no heading structure" guard in ``main``.

Usage
-----
    python txt_to_md/txt_to_md_v7.py                      # uses the default folders
    python txt_to_md/txt_to_md_v7.py --depth absolute --out globallaws_markdown_v5
    python txt_to_md/txt_to_md_v7.py --src IN --out OUT
    python txt_to_md/txt_to_md_v7.py --list               # dry run, just print the plan
"""
from __future__ import annotations
import argparse
import collections
import os
import re
import sys

try:                                    # package import (python -m automation.src.input_process...)
    from . import heading_keywords as KW
except ImportError:                     # run as a plain script next to heading_keywords.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import heading_keywords as KW

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # GlobalLaws/
SRC_DEFAULT = os.path.join(HERE, "globallaws_cleaned")
OUT_DEFAULT = os.path.join(HERE, "globallaws_markdown_v5")

# --------------------------------------------------------------------------- #
# Heading vocabulary.  Canonical order = outer -> inner.                      #
# Values are regex alternations (case-insensitive) of the word used for that  #
# level across the languages present in the corpus.                          #
# --------------------------------------------------------------------------- #
CANON = list(KW.LEVELS)

# Every keyword lives in heading_keywords.py (grouped by language); the converter only composes them.
TOK = {level: KW.alternation(words) for level, words in KW.HEADING_WORDS.items()}

# ordinal words that can stand in for a number after the heading word
ORD = KW.alternation(KW.ORDINAL_WORDS)

# Post-posed unit words: languages that write "5 straipsnis" / "1. člen" instead
# of "Article 5".  Only these get a number-first pattern; Romance/English words
# stay strictly pre-posed so a stray margin number is never read as the ordinal.
POST_TOK = {level: KW.alternation(words) for level, words in KW.POST_WORDS.items()}

# start-of-line prefix, tolerating one leading margin/paragraph number
# ("10 Artículo 1." in consolidated-law HTML); also after a sentence/markup
# boundary in run-on dumps, or right after "(" for "(Article 1)".
_SOL = r"^[ \t>|(]{0,3}(?:\d{1,4}[ \t]{1,3})?"
_MID = r"(?:(?<=[.\)\]•·+|]) |(?<=§) |(?<=\())"
# heading numbers, including a letter/ordinal suffix ('5A', '5-A', '5a', '1er') or in parentheses ('Article (1)')
_NUM = r"(?:\d{1,4}(?:-?[a-z]{1,2})?|\(\d{1,4}\)|[IVXLC]{1,7}|" + ORD + r")"


def _mk(name: str, alt: str) -> re.Pattern:
    pre = (r"(?:" + _SOL + r"|" + _MID + r")\s{0,3}(?:\d{1,3}[.\)]\s*)?(?:" + alt + r")"
           r"\s*(?:N[o°º]\.?\s*)?[\s.\-:–—º°]{0,3}" + _NUM + r"(?![a-z0-9])")
    parts = [pre]
    if name in POST_TOK:
        parts.append(_SOL + r"\s{0,3}\d{1,3}\.?\s{0,2}(?:" + POST_TOK[name] + r")(?![A-Za-zÀ-ÿ])")
    return re.compile(r"(?im)(?:" + r"|".join(parts) + r")")


LEVEL_RX = {name: _mk(name, alt) for name, alt in TOK.items()}
# a number-first label ('57 straipsnis', '3 skyrius'): its leading number is the label's own number, not a margin number
_POST_WORD_RX = re.compile(r"(?i)^\d{1,4}\.?\s+(?:" + "|".join(POST_TOK.values()) +
                           r"|skyrius|nodaļa|kafli|poglavje|peatükk|fejezet|dalis|daļa|hluti|rész)(?![A-Za-zÀ-ÿ])(?![ \t]*\d)")
# line-anchored variant: only a heading if the word opens the line (real heading
# position) -- used to decide which levels are "active", so that cross-references
# like "pursuant to section 22" don't make SECTION look like a heading level.
def _mk_line(name: str, alt: str) -> re.Pattern:
    parts = [_SOL + r"(?:\d{1,3}[.\)]\s*)?(?:" + alt + r")(?=$|[ \t.:)\-–—0-9]|[IVXLC]{1,7}(?![A-Za-zÀ-ÿ]))"]       # a Roman numeral must be a whole word ('Partisipasi' is not 'Part I...')
    if name in POST_TOK:
        parts.append(_SOL + r"\d{1,3}\.?\s{0,2}(?:" + POST_TOK[name] + r")(?![A-Za-zÀ-ÿ])")
    return re.compile(r"(?im)(?:" + r"|".join(parts) + r")")


LINE_RX = {name: _mk_line(name, alt) for name, alt in TOK.items()}
# § behaves like an Article-rank unit marker (Germanic / Nordic / CZ / SK / HU)
PARA_SYM_RX = re.compile(
    r"(?m)(?:"
    r"(?:^|\n|(?<=[.\)]) |\|\s*)\s{0,6}\(?§+\s?\d+"   # "§ N" at a real heading position (also '(§ 3)')
    r"|"
    r"(?:^|\n)\s{0,6}\d+[A-Za-z]?\.?\s?§(?!\w)"       # "N §" (Swedish/Finnish/Hungarian style) -- line-start only (also '4A. §')
    r")"
)
BARE_RX = re.compile(r"(?m)^[ \t]{0,3}(\d{1,3})\.[ \t]+(?=\S)")

# Common-law "marginal headnote" sections: "12.<TAB>Short title.- (1) body ..." (glued) or
# "12.<TAB>Definitions" on its own line.  A TAB after the number + a capitalised short
# headnote is a much stronger signal than a bare "12. " (which is also every enumerated item).
_TAB_GLUED_RX = re.compile(r"(?m)^[ \t]{0,3}(\d{1,3}[A-Z]?)\.\t+([A-Z][^\t\n]{2,100}?)\.-(?=[ \t]|$)")
_TAB_OWN_RX = re.compile(r"(?m)^[ \t]{0,3}(\d{1,3}[A-Z]?)\.\t+([A-Z][^\t\n]{2,100}?)\.?[ \t]*$")
_TAB_SENTENCE_RX = re.compile(r"[.;:,]$|\.\s|\b(?:shall|must|may not|is entitled|means|commits|are)\b")


def _tab_headnotes(text: str) -> list:
    """[(start, end, heading_text)] for tab-headnote section headings (no overlaps)."""
    out = []
    for m in _TAB_GLUED_RX.finditer(text):
        out.append((m.start(), m.end(), f"{m.group(1)}. {m.group(2).strip()}.-"))
    for m in _TAB_OWN_RX.finditer(text):
        head = m.group(2).strip()
        if _TAB_SENTENCE_RX.search(head) or len(head.split()) > 16:
            continue                                   # an enumerated sentence, not a headnote
        out.append((m.start(), m.end(), f"{m.group(1)}. {head}"))
    out.sort()
    res, last = [], -1
    for s, e, t in out:
        if s >= last:
            res.append((s, e, t)); last = e
    return res

# Non-Latin heading markers (CJK / Thai / Arabic / Armenian / Georgian).
# Each entry maps to a canonical level; folded into LEVEL counting + collection.
NONLATIN_RX = [(level, re.compile(rx)) for level, rx in KW.NONLATIN_MARKERS]

# Additional heading words for languages the main vocabulary (TOK) covers poorly.  Line-anchored, case-SENSITIVE (so Spanish "del 5 de mayo" or
# Norwegian "del" in prose is not a heading), and the number must not be followed by a lowercase
# word/digit.  Entries are folded into level counting/collection exactly like NONLATIN_RX.
_XN = r"(?:\d{1,4}(?:-?[a-zA-Z]{1,2}(?![A-Za-z]))?|\(\d{1,4}\)|[IVXLC]{1,7})"     # numbers like '5a' / '(5)'
_XG = r"(?![ \t]*[a-zà-ÿα-ωά-ώа-яё0-9])"
_XS = r"(?m)^[ \t]{0,3}\(?"          # '(Чл. 3.)': a bracketed label is then judged by _is_bracketed


def _xr(level: str, words: str, post: bool = False):
    if post:      # number first: "VII skyrius", "7. peatükk", "VII. Fejezet", "7 kap."
        return (level, re.compile(_XS + _XN + r"\.?[ \t]+(?:" + words + r")(?![A-Za-zÀ-ÿ])"))
    return (level, re.compile(_XS + r"(?:" + words + r")\.?[ \t]*(?:N[o°º]\.?[ \t]*)?" + _XN + _XG))


def _heading_pos(text: str, s: int) -> bool:
    """The non-latin markers (第N条, มาตรา, المادة, Հոդված, მუხლი) carry no line anchor, so they would also match
    cross-references in running prose ("in accordance with Article 5").  A heading sits at the start
    of a line (after margin numbers / bullets) or right after a sentence stop in run-on dumps."""
    pre = text[text.rfind("\n", 0, s) + 1:s]
    if re.fullmatch(r"[ \t>|(]*(?:\d{1,4}[ \t]{1,3})?", pre):
        return True
    return pre.rstrip().endswith(("。", "；", ";", "։", "།"))


# A label directly followed by a "... of this Act" tail is a cross-reference / sentence, not a heading
# ("Article 31 of this Act shall apply ...").  Anchored right after the number; one entry per language family.
_XREF_TAIL_RX = re.compile(r"[ 	]*[.,;:)]{0,2}[ 	]*(?:" + KW.alternation(KW.XREF_TAILS) + r")", re.I)


def _is_xref_tail(text: str, e: int) -> bool:
    return bool(_XREF_TAIL_RX.match(text[e:e + 80]))


# Cross-reference PREFIX guard.  HTML-extracted statutes (wetten.nl) put every inline link on its own
# line, so "als bedoeld in <link>artikel 4.1.1</link>" becomes "als bedoeld in\nartikel 4.1.1\n, ..." and the
# label sits at a line start exactly like a real heading.  A label whose preceding words are a reference
# connective is a citation, not a heading.  Per language; NOT "op" (real headings follow "Sla het
# regelingonderdeel op").  Phrases are matched at the end of the preceding text, whitespace/newline-insensitive.
XREF_PRE = KW.XREF_PREFIXES
_XREF_PRE_RX_CACHE: dict = {}


def _xref_pre_rx(lang: str):
    rx = _XREF_PRE_RX_CACHE.get(lang)
    if rx is None:
        words = sorted(XREF_PRE[lang], key=len, reverse=True)
        rx = re.compile(r"(?<![\wÀ-ÿ])(?:" + "|".join(r"\s+".join(map(re.escape, w.split())) for w in words) + r")\s*$", re.I)
        _XREF_PRE_RX_CACHE[lang] = rx
    return rx


_BRACKETS = {"(": ")", "[": "]", "{": "}"}


def _is_bracketed(text: str, s: int, e: int) -> bool:
    """A label wrapped in a bracket -- "(Article 5)", "[Section 3]", "{Art. 7}", "(Article 5, para 2)" -- is a
    cross-reference.  True when an opener on the label's own line is still unclosed at the end of the label
    (the label regexes tolerate one leading "(", so the opener may be the first char of the match) and the
    matching closer follows on the same line."""
    ls = text.rfind(chr(10), 0, s) + 1
    stack = []
    for ch in text[ls:e]:
        if ch in _BRACKETS:
            stack.append(ch)
        elif stack and ch == _BRACKETS[stack[-1]]:
            stack.pop()
    if not stack:
        return False
    le = text.find(chr(10), e)
    return _BRACKETS[stack[-1]] in text[e:le if le != -1 else len(text)]


def _is_bracket_line(text: str, s: int, e: int) -> bool:
    """the label is bracketed AND its whole line is just "(label)" (optionally a trailing period)."""
    if not _is_bracketed(text, s, e):
        return False
    ls = text.rfind(chr(10), 0, s) + 1
    le = text.find(chr(10), e)
    return bool(re.fullmatch(r"[ \t]*[(\[{][^()\[\]{}]{1,60}[)\]}][ \t.:]*", text[ls:le if le != -1 else len(text)]))


def _is_paren_num_xref(text: str, s: int, e: int, level: str) -> bool:
    """A parenthesised number is a heading number only in 'Article (1)' style headings: a capitalised Book..Article
    word, not followed by another '(' ('paragraph (1)(a)').  Sub-levels ('subsection (1)') are always citations."""
    if not re.search(r"\(\s*\d{1,4}\s*\)\s*$", text[s:e]):
        return False
    if level in ("SUBSECTION", "PARAGRAPH") or text[e:e + 1] == "(":
        return True
    word = re.search(r"[^\W\d_]", text[s:e])
    return bool(word) and word.group().islower()


def _is_xref_prefix(text: str, s: int, lang: str) -> bool:
    return bool(_xref_pre_rx(lang).search(text[max(0, s - 60):s]))


EXTRA_RX = [_xr(level, words, post) for level, words, post in KW.EXTRA_WORDS]
EXTRA_RX.insert(2, ("PART", re.compile(_XS + r"[IVXLC]{1,7}\.[ 	]+(?:" + KW.EXTRA_PART_ROMAN_FIRST + r")(?![A-Za-zÀ-ÿ])")))

# --------------------------------------------------------------------------- #
# Laws.Africa "Legislation Commons" exports (currently: Malawi, Uganda) embed  #
# a Table of Contents ahead of the real body, with every ToC line ending in a  #
# literal tab + page number.  Left alone, each ToC line is promoted to a       #
# heading, and the SAME heading recurs from the real body later in the file.   #
# Gate narrowly on the literal "Laws.Africa" boilerplate string (confirmed in  #
# exactly these two files) rather than a general page-number heuristic -- 30+  #
# other files have superficially similar tab+number lines for unrelated       #
# reasons (citation tables, etc.), so a generic rule would be unsafe.          #
# --------------------------------------------------------------------------- #
_TOC_GATE = "Laws.Africa"
_TOC_MARKER_RX = re.compile(r"(?m)^.*\bContents[ \t]*$")
_TOC_LINE_RX = re.compile(r"^.*\t\d{1,4}[ \t]*$")


_PIPE_ROW_RX = re.compile(r"^\s*\|.*\|\s*$")
_PIPE_SEP_RX = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _unwrap_pipe_layout(text: str) -> str:
    """HTML layout tables extracted as one-cell pipe rows ('|  | text |' + '|---|') -- e.g. the
    Slovak file -- carry plain text, not a data table.  When >=40% of the non-blank lines are pipe
    rows, unwrap every row that has at most one non-empty cell and drop the '|---|' separators
    around them; multi-cell rows (real data tables) are left untouched."""
    lines = text.split("\n")
    nb = [l for l in lines if l.strip()]
    if len(nb) < 4 or sum(1 for l in nb if _PIPE_ROW_RX.match(l)) < 0.4 * len(nb):
        return text
    out = []
    for l in lines:
        if _PIPE_SEP_RX.match(l):
            continue
        if _PIPE_ROW_RX.match(l):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            cells = [c for c in cells if c]
            if len(cells) <= 1:
                if cells:
                    out.append(cells[0])
                continue
        out.append(l)
    return "\n".join(out)


def _strip_toc_block(text: str) -> str:
    """Remove a Laws.Africa-style page-numbered Table of Contents so its
    entries aren't duplicated as headings alongside the real body headings
    later in the file.  No-op unless the Laws.Africa boilerplate is present.

    Also the first pre-processing step for every document (called before analyse()): it unwraps pipe layout
    tables first."""
    text = _unwrap_pipe_layout(text)
    if _TOC_GATE not in text:
        return text
    marker = _TOC_MARKER_RX.search(text)
    if not marker:
        return text
    pos = marker.end() + 1
    end = pos
    matched_lines = 0
    for line in text[pos:].splitlines(keepends=True):
        if line.strip() == "":
            end += len(line)
            continue
        if _TOC_LINE_RX.match(line):
            end += len(line)
            matched_lines += 1
            continue
        break
    if matched_lines < 3:        # not really a ToC block; leave text untouched
        return text
    return text[:pos] + text[end:]


# files that are not statute text at all -> never convert
SKIP_SUBSTRINGS = [
    "82. Italy - DomViol - English (c)",
    "82. Italy - DomViol - Italian (d)",
    "115. Morocco - DomViol - Arabic",
    "146. Serbia - DomViol - Bosnian (a)",
    "85. Jordan - DomViol - English",
]

GROUPING = {"BOOK", "PART", "TITLE", "CHAPTER"}
UNITS = {"SECTION", "ARTICLE", "PARAGRAPH"}


def analyse(text: str, profile: dict | None = None) -> dict:
    """Count heading hits per level and decide which levels are 'active'.  `profile` (heading_keywords.PROFILES)
    may lower the hit gate of some levels."""
    min_hits = (profile or {}).get("min_hits", {})
    nl = max(text.count("\n"), 1)
    runon = len(text) / nl > 300          # HTML dumps with the whole act on one line

    any_ct, line_ct = {}, {}
    for name, rx in LEVEL_RX.items():
        any_ct[name] = len(rx.findall(text))
    for name, rx in LINE_RX.items():
        line_ct[name] = len(rx.findall(text))
    sym = len(PARA_SYM_RX.findall(text))
    line_ct["ARTICLE"] += sym
    any_ct["ARTICLE"] += sym
    for level, rx in NONLATIN_RX + EXTRA_RX:
        if (level, rx) in NONLATIN_RX:      # these markers have no line anchor -> count heading positions only
            n = sum(1 for m in rx.finditer(text) if _heading_pos(text, m.start()))
        else:
            n = len(rx.findall(text))
        if n:
            any_ct[level] = any_ct.get(level, 0) + n
            line_ct[level] = line_ct.get(level, 0) + n   # non-latin markers are already heading-position

    def is_active(name: str) -> bool:
        if name in GROUPING:
            return line_ct.get(name, 0) >= 2 or (runon and any_ct.get(name, 0) >= 5)
        return line_ct.get(name, 0) >= min_hits.get(name, 3) or (runon and any_ct.get(name, 0) >= 8)

    active = [n for n in CANON if is_active(n)]

    # Bare "1. 2. 3." sections are real (common-law bill style) only when they form
    # a long ascending run; scattered/resetting numbers are just list items.
    nums = [int(m.group(1)) for m in re.finditer(r"(?m)^[ \t]{0,3}(\d{1,3})\.[ \t]+\S", text)]
    asc = sum(1 for a, b in zip(nums, nums[1:]) if b >= a)
    mono = asc / max(len(nums) - 1, 1)          # ~1.0 => a real section sequence
    bare = len(nums)
    # Only treat bare "1. 2. 3." as sections when the file has no worded unit
    # heading at all -- otherwise those numbers are enumerated list items inside
    # sections/articles, not headings.
    worded_unit = any(line_ct.get(u, 0) >= 3 for u in UNITS)
    bare_enabled = bare >= 8 and mono >= 0.8 and not worded_unit and not runon
    bare_mode = "num"
    # tab-headnote sections (see _tab_headnotes): >=3 hits with mostly ascending numbers
    tabs = _tab_headnotes(text)
    tnums = [int(re.match(r"\d+", t).group()) for _, _, t in tabs]
    tasc = sum(1 for a, b in zip(tnums, tnums[1:]) if b >= a) / max(len(tnums) - 1, 1)
    if len(tabs) >= 3 and tasc >= 0.7 and not worded_unit_strict(line_ct):
        bare_enabled, bare_mode = True, "tab"
    return {"any": any_ct, "line": line_ct, "active": active, "bare": bare,
            "bare_enabled": bare_enabled, "bare_mode": bare_mode, "sym": sym, "runon": runon}


def worded_unit_strict(line_ct: dict) -> bool:
    """tab-headnote docs may still carry a few 'Section 5' cross-references; only >=6 line-anchored unit words mean a worded-unit statute."""
    return any(line_ct.get(u, 0) >= 6 for u in UNITS)


def depth_map(active: list[str], bare_enabled: bool) -> tuple[dict, int]:
    """Assign Markdown depth (>=2; 1 is the document title) to each active level."""
    dm = {lv: i + 2 for i, lv in enumerate(active)}
    if "SECTION" in dm:                       # bare numbers ARE the sections
        bare_depth = dm["SECTION"]
    elif "ARTICLE" in dm:
        bare_depth = dm["ARTICLE"]
    elif dm:
        bare_depth = min(max(dm.values()) + 1, 6)
    else:
        bare_depth = 2
    return dm, bare_depth


def collect_headings(text: str, dm: dict, bare_enabled: bool, bare_depth: int, bare_mode: str = "num"):
    hits = []  # (start, num_end, depth, level)
    for level, d in dm.items():
        for m in LEVEL_RX[level].finditer(text):
            hits.append((m.start(), m.end(), d, level))
    if "ARTICLE" in dm:
        for m in PARA_SYM_RX.finditer(text):
            hits.append((m.start(), m.end(), dm["ARTICLE"], "ARTICLE"))
    for level, rx in NONLATIN_RX + EXTRA_RX:
        if level in dm:
            nonlatin = (level, rx) in NONLATIN_RX
            for m in rx.finditer(text):
                if nonlatin and not _heading_pos(text, m.start()):
                    continue                                  # cross-reference in running text
                hits.append((m.start(), m.end(), dm[level], level))
    if bare_enabled and bare_mode == "tab":
        # tab-headnote sections, plus plain "N. " lines that continue the section sequence (N == previous + 1):
        # a section whose marginal headnote is missing is still a section boundary, a list item "1." is not.
        events = [(s, e, int(re.match(r"\d+", t).group()), "TAB:" + t) for s, e, t in _tab_headnotes(text)]
        events += [(m.start(), m.end(), int(m.group(1)), "BARE") for m in BARE_RX.finditer(text)]
        events.sort(key=lambda ev: ev[0])
        last = None
        for s, e, num, kind in events:
            if kind != "BARE":
                hits.append((s, e, bare_depth, kind)); last = num
            elif last is not None and num == last + 1:
                hits.append((s, e, bare_depth, "BARE")); last = num
        first_tab = next((k for k, ev in enumerate(events) if ev[3] != "BARE"), None)
        if first_tab:                                   # sections 1..k-1 printed before the first headnote
            last = events[first_tab][2]
            for s, e, num, kind in reversed(events[:first_tab]):
                if kind == "BARE" and num == last - 1:
                    hits.append((s, e, bare_depth, "BARE")); last = num
    elif bare_enabled:
        # 'N. ' lines that follow a lead-in ending in ':' (and the run of numbered lines continuing it) are an
        # enumeration, not bare-numbered sections
        in_enum = False
        for m in BARE_RX.finditer(text):
            prev = text[:m.start()].rstrip().rsplit("\n", 1)[-1].strip()
            if prev.endswith((":", "：")) or (in_enum and BARE_RX.match(prev)):
                in_enum = True
                continue
            in_enum = False
            hits.append((m.start(), m.end(), bare_depth, "BARE"))
    hits.sort(key=lambda h: (h[0], h[2]))
    # drop overlaps (keep the first / outer one)
    kept = []
    last_end = -1
    for s, e, d, lv in hits:
        if s < last_end:
            continue
        kept.append((s, e, d, lv))
        last_end = e
    return kept


# --------------------------------------------------------------------------- #
# Line-level Table-of-Contents detector.                                       #
# A TOC is a run of >=3 lines that each end in a page number after a leader     #
# (dots / underscores / tab / 2+ spaces), or a leader line whose page number    #
# sits alone on the next line, and whose entries start with a heading label.    #
# Heading hits inside a detected span are dropped (text is kept as body); the   #
# TOC title line, when it is a known TOC word, becomes a heading itself.        #
# --------------------------------------------------------------------------- #
_TOC_TITLES = {t.casefold() for titles in KW.TOC_TITLES.values() for t in titles}
_LEADER = r"(?:\.{3,}|(?:\.[ \t]){3,}\.?|…+|·{3,}|(?:_[ \t]){2,}_?|_{3,}|[-–—]{3,}|[ \t]{2,}|\t)"
_TOC_ENTRY_RX = re.compile(r"^[ \t]*(?P<head>\S.*?\S)[ \t]*" + _LEADER + r"[ \t]*(?P<pg>\d{1,4}|[ivxlcIVXLC]{1,6})[ \t]*$")
_TOC_LEAD_RX = re.compile(r"^[ \t]*(?P<head>\S.*?\S)[ \t]*(?:\.{4,}|(?:\.[ \t]){4,}\.?|…{2,}|·{4,}|(?:_[ \t]){3,}_?|_{4,})[ \t]*$")
_PAGE_ONLY_RX = re.compile(r"^[ \t]*\d{1,4}[ \t]*$")
_BARE_LABEL_RX = re.compile(r"^\d{1,3}[A-Z]?\.[ \t]")


def _toc_head_ok(head: str) -> bool:
    if any(rx.match(head) for rx in LINE_RX.values()) or _BARE_LABEL_RX.match(head) or _NUM_ENTRY_RX.match(head):
        return True
    return any(rx.match(head) for _, rx in NONLATIN_RX + EXTRA_RX) or bool(PARA_SYM_RX.match(head))


# 'N<TAB>caption' TOC entries (no period after the number, e.g. '1<TAB>Short title, commencement<TAB>7')
_NUM_ENTRY_RX = re.compile(r"^\d{1,3}[A-Z]?\.?[ \t]+[^\W\d_]")


def _toc_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(".:;").casefold()


def _toc_spans_nopage(text: str, lines: list, offs: list, taken: list) -> list:
    """A TOC without page numbers: a TOC title line followed by >=3 consecutive entry lines (label-led or
    'N.' numbered, blank lines allowed between groups).  The block ends at the first non-entry line or where an entry
    repeats (the body starts: real TOCs are echoed by the headings).  Accepted only if >=50% of the entries recur later
    in the document -- which separates a TOC from an ordinary list of labels."""
    spans = []
    n = len(lines)
    for k, ln in enumerate(lines):
        t = ln.strip()
        if not t or len(t) > 60 or re.sub(r"[\s:.]+$", "", t).casefold() not in _TOC_TITLES:
            continue
        if any(a <= offs[k] < b for a, b, _, _ in taken + spans):
            continue
        j, entries, seen, blanks, gap_used = k + 1, [], set(), 0, False
        while j < n:
            s = lines[j].strip()
            if not s:
                blanks += 1
                if blanks > 2:
                    break
                j += 1; continue
            blanks = 0
            m = _TOC_ENTRY_RX.match(lines[j])
            head = m.group("head") if m else s
            if len(s) > 200 or not _toc_head_ok(head.strip()):
                # one foreign line (a running header where the TOC crosses a page break) is tolerated if entries go on
                nxt = next((l for l in lines[j + 1:] if l.strip()), "")
                if not gap_used and entries and _toc_head_ok((_TOC_ENTRY_RX.match(nxt).group("head") if _TOC_ENTRY_RX.match(nxt) else nxt).strip()):
                    gap_used = True; j += 1
                    continue
                break
            key = _toc_key(head)
            if key in seen:
                break
            seen.add(key); entries.append(j)
            j += 1
        if len(entries) < 2:                              # a TOC title + >=2 entries
            continue
        later = {_toc_key(l) for l in lines[entries[-1] + 1:] if l.strip()}
        later_lab = {" ".join(x.split()[:2]) for x in later}
        echo = sum(1 for e in entries if _toc_key(_TOC_ENTRY_RX.match(lines[e]).group("head") if _TOC_ENTRY_RX.match(lines[e]) else lines[e]) in later
                   or " ".join(_toc_key(lines[e]).split()[:2]) in later_lab)
        if echo < 0.5 * len(entries):
            continue
        last = entries[-1]
        spans.append((offs[k], offs[last] + len(lines[last]), t, offs[k] + len(lines[k])))
    return spans


def _toc_spans(text: str) -> list:
    """[(start, end, title_or_None, title_start)] character spans of TOC blocks in ``text``."""
    lines = text.split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos); pos += len(ln) + 1
    spans = []
    i, n = 0, len(lines)

    def _is_entry(ln):
        m = _TOC_ENTRY_RX.match(ln)
        return bool(m and any(c.isalpha() for c in m.group("head")) and len(ln) <= 260) or bool(_TOC_LEAD_RX.match(ln) and len(ln) <= 260)

    while i < n:
        j, entries, ok, last_end, pending_lead, blanks = i, 0, 0, None, False, 0
        first, gap_used, label_pending = None, False, False
        while j < n:
            ln = lines[j]
            if not ln.strip():
                blanks += 1
                if blanks > 1 or entries == 0:
                    break
                j += 1; continue
            blanks = 0
            m = _TOC_ENTRY_RX.match(ln)
            nxt_ln = next((l for l in lines[j + 1:] if l.strip()), "")
            if not m and len(ln.strip()) <= 40 and _toc_head_ok(ln.strip()) and _TOC_ENTRY_RX.match(nxt_ln) \
                    and not _toc_head_ok(_TOC_ENTRY_RX.match(nxt_ln).group("head").strip()):
                label_pending = True                      # label on its own line, caption + page on the next line
            elif m and any(c.isalpha() for c in m.group("head")) and len(ln) <= 260:
                entries += 1; ok += _toc_head_ok(m.group("head")) or label_pending; pending_lead = False
                label_pending = False
            elif _TOC_LEAD_RX.match(ln) and len(ln) <= 260:
                pending_lead = True; head = _TOC_LEAD_RX.match(ln).group("head")
                entries += 1; ok += _toc_head_ok(head)
            elif pending_lead and _PAGE_ONLY_RX.match(ln):
                pending_lead = False
            elif entries and not gap_used and _is_entry(next((l for l in lines[j + 1:] if l.strip()), "")):
                gap_used = True                           # one foreign line (running header) inside a page-crossing TOC is tolerated
            else:
                break
            if first is None:
                first = j
            last_end = j
            j += 1
        k = (first if first is not None else i) - 1
        while k >= 0 and not lines[k].strip():
            k -= 1
        titled = k >= 0 and len(lines[k].strip()) <= 60 and re.sub(r"[\s:.]+$", "", lines[k].strip()).casefold() in _TOC_TITLES
        if (entries >= 3 or (titled and entries >= 2)) and ok >= max(2, 0.5 * entries):     # a titled TOC needs only >=2 entries
            title = tstart = None
            if k >= 0 and len(lines[k].strip()) <= 60 and re.sub(r"[\s:.]+$", "", lines[k].strip()).casefold() in _TOC_TITLES:
                title, tstart = re.sub(r"\s+", " ", lines[k]).strip(), offs[k]      # TOC title kept verbatim
                spans.append((offs[k], offs[last_end] + len(lines[last_end]), title, offs[k] + len(lines[k])))
            else:
                spans.append((offs[first], offs[last_end] + len(lines[last_end]), None, None))
            i = last_end + 1
        else:
            i = max(j, i + 1) if entries == 0 else i + 1
    spans += _toc_spans_nopage(text, lines, offs, spans)      # TOCs without page numbers
    spans.sort()
    return spans


def _apply_toc_spans(hits: list, text: str, min_depth: int) -> list:
    """drop heading hits inside TOC spans; add the TOC title as a heading at the outermost depth."""
    spans = _toc_spans(text)
    if not spans:
        return hits
    kept = [h for h in hits if not any(a <= h[0] < b for a, b, _, _ in spans)]
    for a, b, title, te in spans:
        if title:
            kept.append((a, te, min_depth, "RAW:" + title))
    kept.sort(key=lambda h: (h[0], h[2]))
    return kept


def _suppress_toc_echo(hits: list, text: str) -> list:
    """Drop a dense early run of heading hits whose (level, label) sequence
    recurs later in the document -- a Table of Contents, in whatever format
    the per-level regexes already happen to match, without needing a new
    per-format marker gate (like ``_strip_toc_block`` above, which only
    covers the Laws.Africa tab+page-number shape). Nothing is deleted from
    ``text``; a suppressed hit's span just reverts to ordinary body prose,
    which is what makes this safe even for a ToC with no clean line
    boundary before the real body (e.g. some sources run the last ToC entry
    directly into body prose on the same physical line).

    Skipped entirely for large multi-law compilations (>300 heading hits),
    where the same dense/recurring signature is legitimate hierarchical
    body text (e.g. book/title/chapter groupings with short articles), not
    a ToC -- confirmed against several 100+ KB compiled-code files where an
    unconditional version of this check produced false positives.

    Also requires the later, corroborating occurrences to be markedly LESS
    dense (larger average gap) than the candidate run itself -- a real ToC
    is always followed by much sparser real body text. Without this check,
    a statute whose real articles are short (small gaps) and which also has
    a second, denser re-listing later (e.g. an Indonesian-style "ARTICLE BY
    ARTICLE" explanatory annex that restates every article number with a
    one-line "Self-explanatory." note) gets the direction backwards: the
    genuine first occurrence looks "dense" too, and would otherwise be
    wrongly suppressed in favor of the annex. Confirmed via real gap
    measurements: genuine ToCs in this corpus have a later/earlier gap
    ratio of 7x-50x; the Indonesian false positive measured 0.5x."""
    GAP_MAX = 300        # max char gap between consecutive hits in the run
    FRONT_IDX = 3         # run must start within the first few hits
    MIN_RUN_LEN = 6       # minimum consecutive hits to call it a run
    MIN_CORROB = 0.55     # fraction of the run's labels that recur later
    MIN_SPARSITY_RATIO = 2.0  # later occurrences' avg gap / run's avg gap

    if len(hits) > 300:
        return hits

    def key(h):
        s, e, d, lv = h
        return (lv, re.sub(r"\s+", " ", text[s:e].strip()).upper())

    keys = [key(h) for h in hits]
    starts = [h[0] for h in hits]
    best = None
    for i0 in range(min(FRONT_IDX + 1, len(hits))):
        j = i0
        while j + 1 < len(hits) and starts[j + 1] - starts[j] <= GAP_MAX:
            j += 1
        run_len = j - i0 + 1
        if run_len < MIN_RUN_LEN:
            continue
        run_keys = keys[i0:j + 1]
        later = set(run_keys)
        later_idxs = [i for i in range(j + 1, len(hits)) if keys[i] in later]
        corrob = sum(1 for k in run_keys if k in set(keys[j + 1:])) / run_len
        if corrob < MIN_CORROB:
            continue
        run_gaps = [starts[k] - starts[k - 1] for k in range(i0 + 1, j + 1)]
        run_avg_gap = sum(run_gaps) / len(run_gaps) if run_gaps else 0
        later_gaps = [starts[later_idxs[k]] - starts[later_idxs[k - 1]]
                      for k in range(1, len(later_idxs))]
        later_avg_gap = sum(later_gaps) / len(later_gaps) if later_gaps else 0
        if later_avg_gap < MIN_SPARSITY_RATIO * max(run_avg_gap, 1):
            continue
        if best is None or run_len > best[2]:
            best = (i0, j, run_len)
    if best is None:
        return hits
    i0, j, _ = best
    return hits[:i0] + hits[j + 1:]


def _split_heading(text: str, s: int, e: int, allow_long: bool = True) -> tuple[str, int]:
    """Given a heading match spanning text[s:e] ('<word> <number>'), return the
    heading string and the index where body text resumes.  Absorbs a trailing
    ordinal mark, a '(rubric)', and a short same-line title; leaves anything
    longer as body."""
    m = re.match(r"[ \t]*[º°ª\.]{0,2}[ \t]*[.\-–—:)]?[ \t]*(\([^)\n]*[A-Za-zÀ-ÿ]{3}[^)\n]*\))?", text[e:e + 96])
    lab_end = e + (m.end() if m else 0)
    # swallow trailing punctuation/space (but not a newline) so the body doesn't
    # start with an orphan ". " left over after "Article 1º. (RUBRIC). ..."
    # a heading number occasionally continues past a hyphen with a stray
    # space ("§1- 101" for a compound chapter-section number, e.g.
    # Micronesia); once a hyphen has been absorbed, keep swallowing digits
    # that continue the SAME number so they don't leak into the body as a
    # stray line-start fragment.
    seen_hyphen = any(ch in "-–—" for ch in text[e:lab_end])
    while lab_end < len(text):
        ch = text[lab_end]
        if ch in " \t.:;-–—":
            seen_hyphen = seen_hyphen or ch in "-–—"
            lab_end += 1
        elif seen_hyphen and ch.isdigit():
            lab_end += 1
        else:
            break
    line_end = text.find("\n", lab_end)
    if line_end == -1:
        line_end = len(text)
    raw_rest = text[lab_end:line_end]
    rest = raw_rest.strip(" \t.:-–—")
    if 0 < len(rest) <= 72 and not re.search(r"[.:]\s", rest):          # ';' is allowed in a caption ('Community policing; ...')
        return text[s:line_end], line_end
    if allow_long and _is_long_caption(rest, raw_rest):
        return text[s:line_end], line_end
    return text[s:lab_end], lab_end


# Real captions are often longer than 72 chars or contain ':' / ';'.  A same-line
# remainder up to _CAPTION_MAX chars is still a caption (not a body sentence) when it
# reads like a title: does not start lowercase, has no sentence break inside, does not
# end like a sentence, and is short in words / commas.
_CAPTION_MAX = 160


def _is_long_caption(rest: str, raw_rest: str) -> bool:
    if not (72 < len(rest) <= _CAPTION_MAX):
        return False
    tail = raw_rest.rstrip()
    if tail.endswith((".", ";", ",")) and not tail.endswith(("..", ".-")):
        return False                                  # a finished sentence -> body text
    if re.search(r"[.;]\s+\S", rest):
        return False                                  # sentence break inside
    if rest.count(",") > 2 or len(rest.split()) > 24:
        return False
    first = rest[0]
    if first.isalpha() and first.islower() and first.isascii():
        return False                                  # continues a wrapped sentence
    return True


# --------------------------------------------------------------------------- #
# Body-text line classification: numbering markers and junk lines.            #
# --------------------------------------------------------------------------- #

# A line matching one of these IS a recognized numbering/lettering marker --
# e.g. "1", "1.", "(1)", "IV.", "(a)", "§ 5" -- and must never be treated as
# junk by the two rules below, even though it is nothing but digits/symbols.
_LISTING_LINE_RX = re.compile(
    r"^(?:"
    r"\(?\d{1,3}\)?[.:)]?"          # 1   1.   1)   (1)   1:
    r"|\(?[ivxlcdm]{1,7}\)?[.:)]?"  # iv   (iv)   IV.
    r"|\(?[a-z]\)?[.:)]?"           # a   (a)   a.
    r"|§\s*\d{1,4}[a-z]?"           # § 5   §5a
    r")$",
    re.IGNORECASE,
)

# Decorative bullet/reference glyphs that sometimes lead a line but carry no
# listing value of their own (unlike "-", digits, letters, roman numerals or
# "§", which are handled/preserved elsewhere) -- table/OCR noise.
_JUNK_LEAD_RX = re.compile(
    r"^[ \t]*[•●○◦▪▫■□♦♢‣·†‡¶→⇒➤✓✔✗✘~^`"
    r"\U0001F300-\U0001FAFF☀-➿](?:️)?[ \t]*"
)


def _is_junk_line(line: str) -> bool:
    """True for a line that is nothing but digits/punctuation/symbols -- no
    letter in any script -- and does not itself read as a numbering/lettering
    marker: table-fragment pipes ("|---|"), ellipsis/dotted-leader remnants,
    orphaned punctuation, stray dates, tab-separated cross-reference debris
    ("1\\t-\\t1"), etc. Confirmed against real globallaws_markdown output."""
    s = line.strip()
    if not s or any(ch.isalpha() for ch in s):
        return False
    return not _LISTING_LINE_RX.match(s)


# --------------------------------------------------------------------------- #
# Body formatter.  Language-agnostic: relies on punctuation, letter case   #
# (only in cased scripts), line-length geometry and enumeration-marker shape.  #
# --------------------------------------------------------------------------- #
_SENT_END_RX = re.compile(r"[.!?;:。！？；：։།༎।።؛؟۔…][\"'”’»)\]]*$")
_NOTE_LINE_RX = re.compile(r"^\s*[(\[{（][^\n]*[)\]}）]\s*\.?\s*$")     # a whole line wrapped in brackets (editorial note / reference)
_CJK_RX = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿＀-￯]")   # Han/Kana + CJK punctuation: no inter-word spaces
_BULLETS = "-–—•●▪◦·*"
_MARKER_RX = re.compile(
    r"^(?P<ind>[ \t]*)(?P<m>"
    r"\((?:[a-zA-Z]{1,2}|[ivxlcIVXLC]{1,6}|\d{1,3}[a-z]?)\)"      # (a) (ii) (1) (1a)
    r"|(?:[a-zA-Z]|\d{1,3}|[ivxlcIVXLC]{1,6})[.)](?!\d)"           # a) 1) 1. iv.
    r"|[-–—•●▪◦·*]"                                                # bullets
    r")(?=[ \t]+\S)")
_ROMAN_RX = re.compile(r"^[ivxlc]+$", re.I)


def _is_cased_lower(ch: str) -> bool:
    return ch.isalpha() and ch.islower()


def _starts_lower(s: str) -> bool:
    s = s.lstrip(" \t\"'“‘«(")
    return bool(s) and _is_cased_lower(s[0])


def _join_pair(a: str, b: str) -> str:
    """join two physical lines of one paragraph (de-hyphenation, no space between CJK/Thai characters)."""
    if re.search(r"[a-zà-ÿа-яё]-$", a) and _starts_lower(b):
        return a[:-1] + b
    if _CJK_RX.match(a[-1:]) and _CJK_RX.match(b[:1]):
        return a + b
    return a + " " + b


def _is_page_number_line(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", s.strip()))


def _norm_line(ln: str) -> str:
    ln = ln.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return ln.rstrip()


def _marker_family(m: str, stack: list) -> str:
    """classify an enumeration marker: '(a)'->'(a)', '(ii)'->'(i)', '1.'->'1.', '•'->'•'.  Roman vs. alpha is resolved from
    the families already open on the stack ('(i)' after '(h)' is alpha)."""
    if m in _BULLETS:
        return "•" + m
    paren = m.startswith("(")
    core = m.strip("()").rstrip(".)")
    tail = ")" if paren else m[-1]
    pre = "(" if paren else ""
    if core[:1].isdigit():
        return pre + "1" + tail
    up = core.isupper()
    alpha_f = pre + ("A" if up else "a") + tail
    roman_f = pre + ("I" if up else "i") + tail
    if _ROMAN_RX.match(core):
        if len(core) == 1 and alpha_f in stack and roman_f not in stack:
            return alpha_f                             # '(i)' continuing '(h)', '(v)' continuing '(u)'
        if roman_f in stack or core.lower() in ("i", "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"):
            return roman_f
        return alpha_f if len(core) == 1 else roman_f
    return alpha_f


def _render_list(items: list) -> list:
    """items: [(indent_width, marker, text)] -> markdown lines with 2-space nesting by marker family."""
    out, stack, ind_stack = [], [], []
    for ind, mk, txt in items:
        fam = _marker_family(mk, stack)
        if fam in stack:
            k = stack.index(fam)
            del stack[k + 1:]; del ind_stack[k + 1:]
        else:
            # a new family nests below the current one (also when extraction lost the indentation)
            while ind_stack and ind < ind_stack[-1] and len(stack) > 1:
                stack.pop(); ind_stack.pop()
            stack.append(fam); ind_stack.append(ind)
        out.append("  " * (len(stack) - 1) + "- " + (mk + " " if mk not in _BULLETS else "") + txt.strip())
    return out


def _canon_pipe(lines: list) -> list:
    out = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?|-+", c) for c in cells if c) and any(cells):
            out.append("|" + "|".join("---" for _ in cells) + "|")
        else:
            out.append("| " + " | ".join(cells) + " |")
    return out


def _paragraphs(lines: list) -> list:
    """lines of one blank-separated block (no list/table lines) -> paragraphs (each a single line)."""
    if not lines:
        return []
    lens = [len(l) for l in lines]
    # wrap width: longest line that is short enough to be a hard-wrapped line, trusted only when >=2 lines reach it
    # (a real wrap margin); long paragraph-per-line lines in the same block are ignored
    short = [n for n in lens if n <= 130]
    width = max(short) if short else 0
    margin_ok = width > 0 and sum(n >= 0.85 * width for n in short) >= 2
    paras, cur = [], lines[0]
    for i, (prev_raw, nxt) in enumerate(zip(lines, lines[1:])):
        prev = cur.rstrip()
        wrapped_geom = margin_ok and lens[i] <= 130
        join = False
        if _MARKER_RX.match(nxt) or _NOTE_LINE_RX.match(nxt) or _NOTE_LINE_RX.match(prev_raw) or \
                _FOOTDEF_RX.match(nxt) or _FOOTDEF_RX.match(prev_raw) or _is_caps_line(nxt) or _is_caps_line(prev_raw):
            join = False                                   # an enumerated line '(d) ...' starts its own paragraph; bracketed notes
            #                                                and short ALL-CAPS lines (captions / titles) stand alone
        elif _starts_lower(nxt) or prev.endswith((",", "-", "–", "—", "(", "/")):
            join = True
        elif wrapped_geom:
            # greedy-wrap test (script independent): the line was broken because the next word did not fit
            first = (nxt[:1] if _CJK_RX.match(nxt[:1]) else nxt.split()[0]) if nxt.split() else ""
            join = len(prev_raw) + 1 + len(first) > 0.9 * width
        if join:
            cur = _join_pair(prev, nxt.strip())
        else:
            paras.append(cur); cur = nxt.strip()
    paras.append(cur)
    return paras


def _is_caps_line(s: str) -> bool:
    """A short line whose cased letters are all upper-case (a caption / title), in a cased script."""
    s = s.strip()
    cased = [c for c in s if c.isalpha() and c.lower() != c.upper()]
    return 4 <= len(cased) and len(s) <= 80 and all(c.isupper() for c in cased)


# Plain-text tables -> pipe tables.  A table is >=3 consecutive rows that split into the same number (>=3) of
# columns on TABs, or on runs of >=2 spaces (space-aligned columns).  In TAB tables a row whose first cell is empty
# continues the previous row (a wrapped cell).
def _cells(line: str, mode: str) -> list:
    if mode == "tab":
        return [c.strip() for c in line.rstrip().split("\t")]
    return [c.strip() for c in re.split(r" {2,}", line.strip())]


def _table_at(lines: list, i: int):
    for mode in ("tab", "space"):
        first = _cells(lines[i], mode)
        ncol = len(first)
        if ncol < 3 or sum(1 for c in first if c) < 3 or (mode == "space" and "\t" in lines[i]):
            continue
        if _MARKER_RX.match(lines[i]) or _TOC_ENTRY_RX.match(lines[i]) and ncol < 3:
            continue
        rows, j = [first], i + 1
        while j < len(lines):
            c = _cells(lines[j], mode)
            if mode == "tab" and c and not c[0] and len(c) <= ncol and rows and any(c):
                c = c + [""] * (ncol - len(c))              # wrapped cell: append to the previous row
                rows[-1] = [(a + " " + b).strip() for a, b in zip(rows[-1], c)]
                j += 1; continue
            if len(c) != ncol or sum(1 for x in c if x) < 2:
                break
            rows.append(c); j += 1
        if len(rows) >= 3:
            return j, rows
    return None


def _pipe_table(rows: list) -> str:
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def format_body(raw: str) -> list:
    """format one body chunk (text between two headings) -> list of markdown blocks (paragraphs, lists, tables)."""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_norm_line(l) for l in raw.split("\n")]
    # drop page-number-only lines and symbol/number junk; strip decorative lead glyphs
    kept = []
    for ln in lines:
        s = ln.strip()
        if s and (_is_page_number_line(s) or _is_junk_line(s)):
            continue
        if s and not _LISTING_LINE_RX.match(s) and not _MARKER_RX.match(ln):
            ln = _JUNK_LEAD_RX.sub("", ln)
        kept.append(ln)
    # group into blocks separated by blank lines
    blocks, cur = [], []
    for ln in kept:
        if ln.strip():
            cur.append(ln)
        elif cur:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    # re-join blocks split by a blank line in mid-sentence (page break / extraction gap)
    merged = []
    for b in blocks:
        if merged and not _MARKER_RX.match(b[0]) and not b[0].lstrip().startswith("|"):
            last = merged[-1][-1].rstrip()
            cjk_cont = bool(_CJK_RX.match(last[-1:])) and bool(_CJK_RX.match(b[0].lstrip()[:1]))   # Han/Kana: no case signal
            if not _SENT_END_RX.search(last) and (_starts_lower(b[0]) or cjk_cont) and not last.lstrip().startswith("|") \
                    and not _FOOTDEF_RX.match(last.lstrip()):
                merged[-1] = merged[-1] + b
                continue
        merged.append(b)
    out = []
    lead_in = False                        # previous emitted text ended with ':' -> enumeration may follow
    list_items = None
    for b in merged:
        i = 0
        while i < len(b):
            ln = b[i]
            if ln.lstrip().startswith("|") and ln.rstrip().endswith("|"):
                j = i
                while j < len(b) and b[j].lstrip().startswith("|"):
                    j += 1
                if list_items:
                    out.append("\n".join(_render_list(list_items))); list_items = None
                out.append("\n".join(_canon_pipe(b[i:j]))); i = j; lead_in = False
                continue
            t = _table_at(b, i)                          # TAB / space-aligned plain-text table
            if t:
                if list_items:
                    out.append("\n".join(_render_list(list_items))); list_items = None
                i, rows = t
                out.append(_pipe_table(rows)); lead_in = False
                continue
            m = _MARKER_RX.match(ln)
            if m and (lead_in or list_items is not None):
                if list_items is None:
                    list_items = []
                ind = len(m.group("ind").expandtabs(4))
                txt = re.sub(r"\s+", " ", ln[m.end():]).strip()
                list_items.append([ind, m.group("m"), txt])
                i += 1
                # continuation lines of this item (hard-wrapped items)
                while i < len(b) and not _MARKER_RX.match(b[i]) and \
                        (not _SENT_END_RX.search(list_items[-1][2]) or _starts_lower(b[i])):
                    list_items[-1][2] = _join_pair(list_items[-1][2], re.sub(r"\s+", " ", b[i]).strip())
                    i += 1
                continue
            # plain text run up to the next marker line that would start a list
            j = i
            run = []
            while j < len(b):
                mm = _MARKER_RX.match(b[j])
                if run and mm and _SENT_END_RX.search(run[-1]) and run[-1].rstrip().endswith((":", "：")):
                    break
                if list_items is not None and mm:
                    break
                if b[j].lstrip().startswith("|") and b[j].rstrip().endswith("|"):
                    break
                if run and _table_at(b, j):                  # a plain-text table starts here
                    break
                run.append(re.sub(r"[ \t ]+", " ", b[j]).strip())
                j += 1
            if list_items:
                out.append("\n".join(_render_list(list_items))); list_items = None
            paras = _paragraphs(run)
            out.extend(paras)
            lead_in = bool(paras) and paras[-1].rstrip().endswith((":", "："))
            i = j
        # a blank line ends a list unless the next block starts with a marker of an open family
    if list_items:
        out.append("\n".join(_render_list(list_items)))
    return [o for o in out if o.strip()]


def _emit_body(buf: list[str], raw: str) -> None:
    blocks = format_body(raw)
    if blocks:
        buf.append("\n\n".join(blocks) + "\n")


def _doc_title(text: str, fallback: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.match(r"(?i)^(current version|version|last amended|updated|about this|www\.|http|page \d|\d+\s*$)", s):
            continue
        return re.sub(r"\s+", " ", s)[:160]
    return fallback


# Fixed heading depths (used by --depth absolute): Book/Part/Title ##, Chapter ###, Section ####, Article #####
ABS_DEPTH = {"BOOK": 2, "PART": 2, "TITLE": 2, "CHAPTER": 3, "SECTION": 4, "SUBSECTION": 5, "ARTICLE": 5, "PARAGRAPH": 6}


def convert(text: str, fallback_title: str, info: dict, depth: str = "relative", xref_pre: str | None = None, bracket_xref: bool = True,
            profile: dict | None = None) -> str:
    """profile (heading_keywords.PROFILES): fixed depths per level and no auto title; overrides `depth`.
    depth='relative' (default): depth follows the levels this file uses;
    depth='absolute': fixed depths per level (ABS_DEPTH); bare/tab-numbered sections are Section (####).
    xref_pre='nl' (key of XREF_PRE): labels preceded by a reference connective ("bedoeld in", "ingevolge", ...) are body text."""
    dm, bare_depth = depth_map(info["active"], info["bare_enabled"])
    if profile:
        dm = {lv: profile["depth"][lv] for lv in info["active"]}
        bare_depth = profile["depth"]["ARTICLE"]
    elif depth == "absolute":
        dm = {lv: ABS_DEPTH[lv] for lv in info["active"]}
        bare_depth = 4
    hits = collect_headings(text, dm, info["bare_enabled"], bare_depth, info.get("bare_mode", "num"))
    # with fixed depths a TOC title is always '##' (outermost), not the outermost level this file happens to use
    hits = _apply_toc_spans(hits, text, 2 if depth == "absolute" and not profile else (min(dm.values()) if dm else 2))
    hits = _suppress_toc_echo(hits, text)
    # every detected TOC span is rendered as a nested list (after its title heading, if any)
    for a, b, title, te in _toc_spans(text):
        hits.append((te if title else a, b, 0, "TOCL:"))
    hits.sort(key=lambda h: (h[0], h[2]))
    # Safeguard: when most of the file's labelled hits are a bracketed label alone on its line ("(Article 1)" as the
    # heading of every article, e.g. the Saudi regulation) that IS the file's heading convention, not a citation.
    labelled = [h for h in hits if h[3] != "BARE" and h[3][:4] not in ("TAB:", "RAW:")]
    bracket_own_line = bool(bracket_xref and labelled and
                            sum(_is_bracket_line(text, h[0], h[1]) for h in labelled) >= 0.5 * len(labelled))
    buf = ["# " + _doc_title(text, fallback_title) + "\n"] if fallback_title is not None else []
    cur = 0
    for s, e, d, lv in hits:
        if s < cur:
            continue
        if lv == "TOCL:":
            _emit_body(buf, text[cur:s])
            toc = _render_toc(text[s:e])
            if toc:
                buf.append("\n" + toc + "\n")
            cur = e
            continue
        if lv[:4] not in ("TAB:", "RAW:"):
            # skip BEFORE emitting body: a skipped hit must not flush text[cur:s] (it would be emitted again by the next hit)
            if lv != "BARE" and _is_xref_tail(text, e):       # "Article 31 of this Act shall apply ..." is body text
                continue
            if (bracket_xref and lv != "BARE" and _is_bracketed(text, s, e)     # "(Article 5)" / "[Section 3]" is a citation
                    and not (bracket_own_line and _is_bracket_line(text, s, e))):
                continue
            if xref_pre and lv != "BARE" and _is_xref_prefix(text, s, xref_pre):   # "bedoeld in <newline> artikel 4.1.1" is a citation
                continue
            if lv != "BARE" and _is_paren_num_xref(text, s, e, lv):   # 'subsection (1)', 'Subsection (1)(a)' are citations
                continue
        _emit_body(buf, text[cur:s])
        if lv[:4] in ("TAB:", "RAW:"):                 # heading text already exact (tab-headnote / TOC title)
            cur = e
            buf.append("\n" + "#" * min(d, 6) + " " + lv[4:] + "\n")
            continue
        htext, cur = _split_heading(text, s, e, allow_long=(lv != "BARE"))
        # heading text is verbatim (a caption may end in ':' or '.', a label may be 'Članak 5.'); only when the body
        # continues on the same line is the dangling separator ('Article 5 -', 'Art. 5.-') dropped, keeping a final '.'
        le = text.find("\n", cur)
        full_line = not text[cur:le if le != -1 else len(text)].strip()
        htext = re.sub(r"\s+", " ", htext).strip()
        if not full_line:
            htext = re.sub(r"[\s\-–—:]+$", "", htext)
        # drop a leading margin/paragraph number that sits before the heading word
        # -- but not when that number IS the heading's own number for a
        # number-first "N §" heading (Swedish/Finnish style, from PARA_SYM_RX
        # branch B), where stripping it would delete the section's own number.
        if not _POST_WORD_RX.match(htext):
            htext = re.sub(r"^\d{1,4}\s+(?=[^\d\s§])", "", htext)
        buf.append("\n" + "#" * min(d, 6) + " " + htext + "\n")
    _emit_body(buf, text[cur:])
    md = "\n".join(buf)
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def _toc_entry_level(head: str):
    """canonical level name of a TOC entry's label (None if it carries no label)."""
    for name in CANON:
        if LINE_RX[name].match(head):
            return name
    for level, rx in NONLATIN_RX + EXTRA_RX:
        if rx.match(head):
            return level
    if re.match(r"^\d{1,4}[A-Z]?\.?[ \t]", head):
        return "SECTION"                                 # common-law 'N. headnote' / 'N<TAB>headnote'
    return None


def _render_toc(block: str) -> str:
    """TOC entry lines -> nested '- head — page' list; indentation = dense rank of the entries' label kinds."""
    entries = []
    label = None                                        # a label line waiting for its caption + page line
    for ln in block.split("\n"):
        if not ln.strip():
            continue
        m = _TOC_ENTRY_RX.match(ln)
        if not m and len(ln.strip()) <= 40 and _toc_head_ok(ln.strip()) and label is None:
            label = ln.strip()
            continue
        if m and any(c.isalpha() for c in m.group("head")):
            head = m.group("head")
            if label is not None and not _toc_head_ok(head.strip()):
                head = label + " " + head.strip()
            elif label is not None:
                entries.append([label, None])
            label = None
            entries.append([head, m.group("pg")]); continue
        if label is not None:
            entries.append([label, None]); label = None
        m = _TOC_LEAD_RX.match(ln)
        if m:
            entries.append([m.group("head"), None]); continue
        if _PAGE_ONLY_RX.match(ln) and entries and entries[-1][1] is None:
            entries[-1][1] = ln.strip(); continue
        if not _toc_head_ok(ln.strip()) and ("\t" in ln or re.search(r"\S {3,}\S", ln)):
            continue                                       # a running header caught inside a page-crossing TOC
        entries.append([ln.strip(), None])
    if label is not None:
        entries.append([label, None])
    if not entries:
        return ""
    names, last = [], None
    for head, _ in entries:
        n = _toc_entry_level(head.strip())
        last = n or last
        names.append(last)
    order = sorted({CANON.index(n) for n in names if n in CANON})
    rank = {c: i for i, c in enumerate(order)}
    out = []
    for (head, pg), n in zip(entries, names):
        lvl = rank.get(CANON.index(n), 0) if n in CANON else 0
        h = re.sub(r"\s+", " ", head).strip()
        out.append("  " * lvl + "- " + h + (f" — {pg}" if pg else ""))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Page furniture (running headers / footers / page numbers).  Language-    #
# agnostic: a furniture line repeats across the document once its numbers are   #
# normalised, is short, is not a sentence and carries page evidence.            #
# --------------------------------------------------------------------------- #
_SEP_LINE_RX = re.compile(r"^[ \t]*(?:_{5,}|-{5,}|―{3,}|—{3,}|={5,})[ \t]*$")
_FURN_MAX = 140
# the word "page" in the corpus languages (same kind of lexicon as TOK); used only for page-footer lines
PAGE_WORDS = KW.alternation(KW.PAGE_WORDS)
_PAGE_FOOTER_RX = re.compile(r"(?i)(?:" + PAGE_WORDS + r")[ \t]*\d{1,4}(?:[ \t]*(?:/|of|de|di|von|z|od|из|з|από)[ \t]*\d{1,4})?"
                             r"|\d{1,4}[ \t]*(?:" + PAGE_WORDS + r")|第\d{1,4}[页頁]")


def _furn_key(line: str) -> tuple:
    """(core, digits): core = line with number runs -> '#', leading/trailing page-number fields dropped."""
    s = re.sub(r"\s+", " ", line.strip()).casefold()
    toks = re.sub(r"\d+", "#", s).split(" ")
    while toks and re.fullmatch(r"[#./\-–—|]+", toks[0]):
        toks.pop(0)
    while toks and re.fullmatch(r"[#./\-–—|]+", toks[-1]):
        toks.pop()
    return " ".join(toks), tuple(re.findall(r"\d+", s))


def _page_counter(digit_tuples: list) -> bool:
    """True if one number field increases in small steps across the occurrences (a page number: +1 per page, +2 for
    odd/even book headers).  Occurrences that print the page only on alternate pages are compared among themselves."""
    by_len = collections.defaultdict(list)
    for t in digit_tuples:
        by_len[len(t)].append([int(x) for x in t])
    for rows in by_len.values():
        if len(rows) < 2 or not rows[0]:
            continue
        for j in range(len(rows[0])):
            col = [r[j] for r in rows]
            steps = [b - a for a, b in zip(col, col[1:])]
            if all(1 <= s <= 4 for s in steps) and max(col) <= 1500:     # years / law numbers are not page numbers
                return True
    return False


def _label_led(line: str) -> bool:
    s = line.strip()
    return any(rx.match(s) for rx in LINE_RX.values()) or bool(PARA_SYM_RX.match(s)) or \
        any(rx.match(s) for _, rx in EXTRA_RX) or any(rx.match(s) and _heading_pos(s, 0) for _, rx in NONLATIN_RX)


def find_furniture(text: str) -> set:
    """indices (0-based line numbers) of running-header / footer / page-number lines."""
    lines = text.split("\n")
    protect = set()
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos); pos += len(ln) + 1
    for a, b, _, _ in _toc_spans(text):                   # TOC entries repeat as headings; never furniture
        protect.update(i for i, o in enumerate(offs) if a <= o < b)
    groups = collections.defaultdict(list)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or len(s) > _FURN_MAX or i in protect or _NOTE_LINE_RX.match(s) or _MARKER_RX.match(ln):
            continue
        if _SENT_END_RX.search(s) or s.endswith((",", "—", "–")):
            continue                                      # a sentence / lead-in ('In this Part—'), not furniture
        core, digits = _furn_key(s)
        if len(re.sub(r"[^\w]", "", core)) < 3:
            continue
        groups[core].append((i, digits, "\t" in ln or bool(re.search(r"\S {3,}\S", s))))
    furn = set()
    for core, occ in groups.items():
        if len(occ) < 2:
            continue
        idx = [o[0] for o in occ]
        gaps = [b - a for a, b in zip(idx, idx[1:])]
        if min(gaps) < 3:
            continue                                      # adjacent repeats: a list / form, not page furniture
        digit_var = len({o[1] for o in occ}) > 1
        counter = digit_var and _page_counter([o[1] for o in occ])
        tabbed = sum(o[2] for o in occ) >= 2
        led = _label_led(lines[idx[0]])
        words = [w for w in core.split() if re.search(r"[^\W\d_]", w)]
        if led:
            # 'Chapter 1 - Caption<TAB>12' repeated per page keeps its label number and carries a page counter / extra
            # tab field; repeated real headings ('CAPÍTULO II' in every Title of a code) do not
            same_label = len({o[1][:1] for o in occ}) == 1
            ok = same_label and tabbed and (counter or (not digit_var and len(occ) >= 3))   # laid-out header, not 'Neni 143/a/1..4'
        elif digit_var:
            # varying numbers must be a page counter ('Act 2007<TAB>11', '... 12'): repeated amendment notes
            # ('Fracción adicionada DOF 14-01-1985') and bare labels ('Artigu 5') have arbitrary numbers
            ok = counter and len(words) >= 1 and (len(words) >= 2 or tabbed)
        else:
            # a constant line: needs a tab-laid-out header, or >=3 regularly spaced repeats of a title-like line
            mean = sum(gaps) / len(gaps)
            regular = mean >= 8 and (max(gaps) - min(gaps)) <= 1.2 * mean
            long_w = [w for w in words if len(w) >= 4]
            cased = [w for w in long_w if w[:1].isalpha() and w[:1].lower() != w[:1].upper()]
            # headers are UPPERCASE / Title Case; a repeated sentence-case body fragment is content ('Lorsque les faits ont ...')
            title_case = not cased or sum(w[:1].isupper() for w in cased) >= 0.6 * len(cased)
            titleish = len(words) >= 3 and title_case and not (words[-1].islower() and len(words[-1]) <= 3)
            ok = (tabbed and len(words) >= 2) or (len(occ) >= 3 and titleish and regular)
        if ok:
            furn.update(idx)
    # page-number footers: '<page word> 3', 'Página 3/10', '- 3 -', '3/10'.  A bare 'word N' line is NOT enough
    # ('Artigu 5', 'Artigu 6' are headings and also count up), so the word must be a page word (PAGE_WORDS)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if i in protect or not s:
            continue
        if _PAGE_FOOTER_RX.fullmatch(s) or re.fullmatch(r"[-–—]\s*\d{1,4}\s*[-–—]|\d{1,4}\s*/\s*\d{1,4}", s):
            furn.add(i)
    # Label-leading running headers change with the current chapter, so they rarely repeat; but each one reprints
    # an EARLIER heading line plus a page number / title ('Poglavlje II Područje primjene<TAB>25'). TOC entries precede
    # their headings and are protected anyway.
    earlier = set()
    for i, ln in enumerate(lines):
        s = re.sub(r"[ \t]+", " ", ln.strip())
        if not s or i in protect or not _label_led(s):
            continue
        for k in range(len(s) - 1, 3, -1):
            if s[k] == " " and s[:k] in earlier:
                rest = s[k + 1:]
                alpha_words = [w for w in rest.split() if re.search(r"[^\W\d_]{2,}", w)]
                page_like = bool(re.fullmatch(r"\d{1,4}", rest)) or ("\t" in ln.strip() and ln.strip().split("\t")[0].strip() == s[:k]) \
                    or (bool(re.search(r"\d", rest)) and len(alpha_words) >= 2 and len(rest) <= 40 and "," not in rest
                        and not _starts_lower(rest))       # 'Document Generated: 2026-03-24', not a wrapped sentence
                if len(rest) <= 80 and page_like and not _SENT_END_RX.search(rest) and not _label_led(rest):
                    furn.add(i)
                break
        # only a heading WITH a caption can be the stem of a running header ('CAPÍTULO II' alone is re-used by the
        # next Title of a code: 'CAPÍTULO II APLICACIÓN DE LAS PENAS' is a real heading)
        if len([w for w in s.split() if re.search(r"[^\W\d_]{2,}", w)]) >= 3:
            earlier.add(s)
    # second pass: a constant line repeated >=2x that sits next to removed furniture on every occurrence
    for core, occ in groups.items():
        idx = [o[0] for o in occ]
        if len(idx) >= 2 and not set(idx) <= furn and not _label_led(lines[idx[0]]):
            def near(i):
                j, k = i - 1, i + 1
                while j >= 0 and not lines[j].strip():
                    j -= 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                return j in furn or k in furn
            if all(near(i) for i in idx):
                furn.update(idx)
    return furn


# --------------------------------------------------------------------------- #
# Footnotes -> Markdown footnotes ('[^N]' markers, '[^N]: text' blocks).     #
# --------------------------------------------------------------------------- #
_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUP_TR = str.maketrans(_SUP, "0123456789")
_SUP_INLINE_RX = re.compile(r"(?<=[^\s\d" + _SUP + r"])([" + _SUP + r"]{1,3})\)?")
_BRK_INLINE_RX = re.compile(r"(?<=[^\s\[(])\[(\d{1,3})\](?!\()")
_SUP_DEF_RX = re.compile(r"^[ \t]*([" + _SUP + r"]{1,3})\)?[ \t]+(\S.*)$")
_BRK_DEF_RX = re.compile(r"^[ \t]*\[(\d{1,3})\][ \t]+(\S.*)$")
_NUM_DEF_RX = re.compile(r"^[ \t]*(\d{1,3})[ \t]+(\S.*)$")
_FOOTDEF_RX = re.compile(r"^\[\^\d{1,3}\]:")


def _split_sup_defs(s: str) -> list:
    """'¹) L. 88/2008. ²) L. 153/2020.' -> [(1, 'L. 88/2008.'), (2, 'L. 153/2020.')]"""
    parts = re.split(r"(?:^|[ \t]+)([" + _SUP + r"]{1,3})\)?[ \t]+", " " + s.strip())
    out = []
    for k in range(1, len(parts) - 1, 2):
        out.append((int(parts[k].translate(_SUP_TR)), parts[k + 1].strip()))
    return out


def _footnote_defs(lines: list) -> tuple:
    """pass 1: {line index: [(n, text)]} of footnote-definition lines + indices of separator rules to drop.
    A definition is a line opening with a superscript number, a '[N]' whose marker occurs inline, or a plain
    number right after a separator rule ('____') or whose superscript marker occurs inline."""
    cand = set()
    for ln in lines:
        if not (_SUP_DEF_RX.match(ln) or _BRK_DEF_RX.match(ln)):
            cand.update(int(m.group(1).translate(_SUP_TR)) for m in _SUP_INLINE_RX.finditer(ln))
            cand.update(int(m.group(1)) for m in _BRK_INLINE_RX.finditer(ln))
    defs, seps, after_sep = {}, set(), False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if _SEP_LINE_RX.match(ln):
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if _SUP_DEF_RX.match(nxt) or _BRK_DEF_RX.match(nxt) or _NUM_DEF_RX.match(nxt):
                seps.add(i); after_sep = True
            continue
        if not s:
            continue
        m = _SUP_DEF_RX.match(ln)
        if m:
            defs[i] = _split_sup_defs(s); continue
        m = _BRK_DEF_RX.match(ln)
        if m and int(m.group(1)) in cand:
            defs[i] = [(int(m.group(1)), m.group(2).strip())]; continue
        m = _NUM_DEF_RX.match(ln)
        if m and (after_sep or int(m.group(1)) in cand) and not _MARKER_RX.match(ln):
            defs[i] = [(int(m.group(1)), m.group(2).strip())]; continue
        after_sep = False
    return defs, seps


def convert_footnotes(text: str) -> str:
    lines = text.split("\n")
    defs, seps = _footnote_defs(lines)
    if not defs:
        return text
    known = {n for d in defs.values() for n, _ in d}      # pass 2: only markers with a definition ('m²' stays)
    out, defs_at = [], []                                 # defs_at: (output index of the first def line, n lines)
    for i, ln in enumerate(lines):
        if i in seps:
            continue
        if i in defs:
            new = [f"[^{n}]: {t}" for n, t in defs[i]]
            if defs_at and out and not out[-1].strip() and len(out) == defs_at[-1][0] + defs_at[-1][1] + 1:
                out.pop()                                  # continues the previous footnote block
                out.extend(new); out.append("")
                defs_at[-1] = (defs_at[-1][0], defs_at[-1][1] + len(new))
                continue
            if out and out[-1].strip():
                out.append("")
            defs_at.append((len(out), len(new)))
            out.extend(new); out.append("")
            continue
        if not ln.strip() and defs_at and out and not out[-1].strip() and len(out) == defs_at[-1][0] + defs_at[-1][1] + 1:
            continue                                       # blank lines inside / right after a footnote block
        ln = _SUP_INLINE_RX.sub(lambda mm: f"[^{int(mm.group(1).translate(_SUP_TR))}]"
                                if int(mm.group(1).translate(_SUP_TR)) in known else mm.group(0), ln)
        ln = _BRK_INLINE_RX.sub(lambda mm: f"[^{mm.group(1)}]" if int(mm.group(1)) in known else mm.group(0), ln)
        out.append(ln)
    # a footnote block that interrupted a paragraph moves to after the end of that paragraph
    res = list(out)
    for start, n in reversed(defs_at):
        block = res[start:start + n]
        prev = next((j for j in range(start - 1, -1, -1) if res[j].strip()), None)
        nxt = next((j for j in range(start + n, len(res)) if res[j].strip()), None)
        if prev is None or nxt is None or _FOOTDEF_RX.match(res[prev]) or _FOOTDEF_RX.match(res[nxt]):
            continue
        cont = not _SENT_END_RX.search(res[prev].rstrip()) and (_starts_lower(res[nxt]) or
                                                              (_CJK_RX.match(res[prev].rstrip()[-1:]) and _CJK_RX.match(res[nxt].lstrip()[:1])))
        if not cont:
            continue
        end = nxt
        while end + 1 < len(res) and res[end + 1].strip() and not _FOOTDEF_RX.match(res[end + 1]):
            end += 1
        del res[start:start + n]
        end -= n
        res[end + 1:end + 1] = [""] + block + [""]
    return "\n".join(res)


def strip_leading_chrome(text: str) -> str:
    """Web-page navigation chrome before the statute (HTML extractions): a run of >=8 short lines, none of them a
    sentence (no final punctuation) and most of them menu-short, before the first heading-like or sentence line."""
    lines = text.split("\n")
    run = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if _label_led(s) or len(s) > 70 or (_SENT_END_RX.search(s) and len(s) >= 30) or _toc_head_ok(s) or \
                re.sub(r"[\s:.]+$", "", s).casefold() in _TOC_TITLES:
            break                                        # the statute (or its TOC) starts here
        run.append(i)
    if len(run) < 10:
        return text
    body = [lines[i].strip() for i in run]
    # navigation menus: many 1-3 word items, few numbers, few ALL-CAPS lines; a title / assent block ('I ASSENT',
    # 'KING OF ESWATINI', 'Act No of 2018', 'DECREE of the PRESIDENT') is upper-case and full of numbers
    menu = sum(len(s.split()) <= 3 for s in body)
    digits = sum(bool(re.search(r"\d", s)) for s in body)
    caps = sum(_is_caps_line(s) for s in body)
    if any(_SENT_END_RX.search(s) for s in body) or menu < 0.7 * len(body) or digits > 0.2 * len(body) or caps > 0.3 * len(body):
        return text
    drop = set(run)
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)


def preprocess(text: str) -> str:
    """Clean the raw text before heading detection: drop web navigation chrome at the top, then page furniture, then
    convert footnotes (all of these can look like headings)."""
    text = strip_leading_chrome(text)
    lines = text.split("\n")
    furn = find_furniture(text)
    if furn:                                              # deleted, not blanked: a split paragraph can then re-join
        text = "\n".join(ln for i, ln in enumerate(lines) if i not in furn)
    return convert_footnotes(text)


def convert_document(text: str, title: str | None = None, *, snippet: bool = False, depth: str = "absolute",
                     xref_pre: str | None = None, profile: dict | None = None) -> str:
    """Single entry point for the corpus CLI and the golden-dataset runner.  title=None -> no '# title' line.  snippet=True lowers the file-level heading gates to
    '>=1 line-anchored hit per level' (the golden-runner 'relax' mode: a few-line excerpt cannot reach the >=2/>=3 gates).
    Documents with no detected heading level are still body-formatted (the corpus CLI decides whether to skip them)."""
    text = preprocess(_strip_toc_block(text))            # page furniture + footnotes first
    info = analyse(text, profile)
    if snippet:
        info["active"] = [n for n in CANON if info["line"].get(n, 0) >= 1]
    return convert(text, title, info, depth=depth, xref_pre=xref_pre, profile=profile)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT, help="folder of cleaned .txt files")
    ap.add_argument("--out", default=OUT_DEFAULT, help="destination folder for .md files")
    ap.add_argument("--list", action="store_true", help="dry run: print the plan, write nothing")
    ap.add_argument("--depth", choices=["relative", "absolute"], default="relative", help="heading depth: per-file relative (default) or fixed per level")
    ap.add_argument("--no-xref-pre", action="store_true", help="disable the language-specific cross-reference prefix guard (XREF_PRE)")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"source folder not found: {args.src}", file=sys.stderr)
        return 2
    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith(".txt"))
    if not args.list:
        os.makedirs(args.out, exist_ok=True)

    converted, skipped = [], []
    for fn in files:
        path = os.path.join(args.src, fn)
        if any(sub in fn for sub in SKIP_SUBSTRINGS):
            skipped.append((fn, "listed as non-statutory"))
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        text = preprocess(_strip_toc_block(text))
        info = analyse(text)
        if not info["active"] and not info["bare_enabled"]:
            skipped.append((fn, "no heading structure detected"))
            continue
        stem = re.sub(r"\.txt$", "", fn, flags=re.I)
        chain = " > ".join(info["active"]) or ("BARE-numbered" if info["bare_enabled"] else "?")
        converted.append((fn, chain))
        if args.list:
            continue
        xp = None if args.no_xref_pre else next((k for tag, k in KW.XREF_PREFIX_FILENAME_TAGS.items() if tag in fn), None)
        md = convert(text, stem, info, depth=args.depth, xref_pre=xp)
        with open(os.path.join(args.out, stem + ".md"), "w", encoding="utf-8") as fh:
            fh.write(md)

    if not args.list:
        idx = ["# Markdown conversion index",
               "",
               f"Converted **{len(converted)}** files from `{os.path.basename(args.src)}`; "
               f"skipped **{len(skipped)}**.",
               "",
               "| File | Detected hierarchy (outer → inner) |",
               "|---|---|"]
        for fn, chain in converted:
            idx.append(f"| {re.sub(r'.txt$', '.md', fn)} | {chain} |")
        if skipped:
            idx += ["", "## Skipped", "", "| File | Reason |", "|---|---|"]
            idx += [f"| {fn} | {why} |" for fn, why in skipped]
        with open(os.path.join(args.out, "_INDEX.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(idx) + "\n")

    print(f"converted: {len(converted)}   skipped: {len(skipped)}")
    for fn, why in skipped:
        print(f"  skip  {fn}  ({why})")
    if not args.list:
        print(f"\noutput -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
