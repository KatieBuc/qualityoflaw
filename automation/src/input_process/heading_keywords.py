# -*- coding: utf-8 -*-
r"""Every language-specific keyword and marker used by ``txt_to_md_v7``.

Nothing in the converter's logic names a language; to support a new one, add its words here.
Values are regex fragments (matched case-insensitively unless a section says otherwise), grouped
by language code, so a language can be read, added or removed in one place.

Sections
--------
HEADING_WORDS    level -> language -> words that open a heading ("Article", "Pasal", "Chapitre" ...)
ORDINAL_WORDS    language -> ordinals that may stand in for a number ("Premier", "Kesatu" ...)
POST_WORDS       level -> language -> words written AFTER the number ("5 straipsnis")
EXTRA_WORDS      case-sensitive, line-anchored words for languages HEADING_WORDS covers poorly
NONLATIN_MARKERS CJK / Thai / Arabic / Armenian / Georgian markers (regexes)
XREF_TAILS       language -> phrases that, right after a label, make it a citation ("of this Act")
XREF_PREFIXES    language -> phrases that, right before a label, make it a citation
TOC_TITLES       language -> words that title a table of contents
PAGE_WORDS       language -> the word "page" (page-footer detection)

Level names (outer -> inner, see ``LEVELS``): BOOK, PART, TITLE, CHAPTER, SECTION, SUBSECTION,
ARTICLE, PARAGRAPH.  Indonesian statutes run BAB > BAGIAN > PARAGRAF > PASAL, so
BAB = CHAPTER, BAGIAN = SECTION, PARAGRAF = SUBSECTION, PASAL = ARTICLE.
"""
from __future__ import annotations

LEVELS = ["BOOK", "PART", "TITLE", "CHAPTER", "SECTION", "SUBSECTION", "ARTICLE", "PARAGRAPH"]

# --------------------------------------------------------------------------- #
# Heading words: level -> language -> regex alternation                       #
# --------------------------------------------------------------------------- #
HEADING_WORDS: dict[str, dict[str, str]] = {
    "BOOK": {
        "en": r"BOOK",
        "fr": r"LIVRE",
        "es": r"LIBRO",
        "pt": r"LIVRO",
        "hr": r"KNJIGA",
        "cs": r"KNIHA",
        "ru": r"КНИГА",
        "ro": r"CARTEA",
        "id": r"BUKU",
    },
    "PART": {
        "en": r"PART",
        "fr": r"PARTIE",
        "es-it-pt": r"PARTE",
        "ro": r"PARTEA",
        "pl": r"DZIA[LŁ]",
        "lt": r"DALIS",
        "lv": r"DA[LĻ]A",
        "hu": r"R[EÉ]SZ",
        "cs-sk": r"[CČ][AÁ]S[TŤ]|[CČ]AST",
        "ru-uk": r"ЧАСТ|ЧАСТИНА",
        "el": r"ΜΕΡΟΣ",
        "tr": r"B[OÖ]L[UÜ]M|KISIM",
    },
    "TITLE": {
        "en": r"TITLE",
        "fr": r"TITRE",
        "es": r"T[IÍ]TULO",
        "it": r"TITOLO",
        "cs": r"TITUL",
        "ro": r"TITLUL",
        "sk": r"HLAVA",
        "ru": r"ТИТУЛ",
    },
    "CHAPTER": {
        "en": r"CHAPTER",
        "fr": r"CHAPITRE",
        "es": r"CAP[IÍ]TULO|CAPITULO",
        "it": r"CAPITOLO|CAPO",
        "ro": r"CAPITOL|CAPITOLUL",
        "de": r"KAPITEL",
        "no": r"KAPITTEL",
        "cs": r"KAPITOLA",
        "sq": r"KAPITULLI|KAPITLI|KREU",
        "pl": r"ROZDZIA[LŁ]",
        "hr": r"POGLAVLJE",
        "nl": r"HOOFDSTUK",
        "hu": r"FEJEZET",
        "is": r"KAFLI",
        "lv": r"NODA[LĻ]A",
        "lt": r"SKYRIUS",
        "id": r"BAB",
        "tl": r"KABANATA",
        "ru": r"ГЛАВА",
        "uk-mk": r"РОЗД[IІ]Л",
        "el": r"ΚΕΦΑΛΑΙΟ",
    },
    "SECTION": {
        "en": r"SECTION",
        "es": r"SECCI[OÓ]N",
        "it": r"SEZIONE",
        "pt": r"SE[CÇ][CÇ]?[AÃ]O",
        "ro": r"SEC[TŢ]IUNE|SEC[TŢ]IUNEA",
        "de": r"ABSCHNITT",
        "cs": r"ODD[IÍ]L",
        "hr-bs": r"ODJELJAK|ODSJEK",
        "sk": r"ODSEK",
        "et": r"JAOS",
        "ru": r"РАЗДЕЛ",
        "el": r"ΤΜΗΜΑ",
        "id": r"BAGIAN",
    },
    "SUBSECTION": {
        "en": r"SUB-?SECTION",
        "fr": r"SOUS-?SECTION",
        "es": r"SUBSECCI[OÓ]N",
        "hr": r"POT?-?ODJELJAK",
        "id": r"PARAGRAF",            # Indonesian "Paragraf" sits between Bagian and Pasal
    },
    "ARTICLE": {
        "en-fr": r"ARTICLE",
        "es": r"ART[IÍ]CULO",
        "pt": r"ARTIGO",
        "it": r"ARTICOLO",
        "ro": r"ARTICOLUL",
        "de": r"ARTIKEL",
        "no": r"ARTIKKEL",
        "lv": r"ARTIKLA",
        "pl": r"ARTYKU[LŁ]",
        "sq": r"ARTIKULLI|NENI",
        "cs-sk": r"[CČ]L[AÁ]NEK",
        "hr-sl": r"[CČ]LANAK|[CČ]LAN|[CČ]LEN",
        "lv-panta": r"PANTS",
        "lt": r"STRAIPSNIS",
        "hu": r"SZAKASZ",
        "is": r"GREIN",
        "tr": r"MADDE",
        "abbrev": r"ART\.|GR\.",
        "ru": r"СТАТ[ЬТ]Я",
        "uk": r"СТАТТЯ",
        "bg-mk": r"ЧЛЕН|ЧЛАН",
        "el": r"[ΆΑ]ΡΘΡΟ",
        "id": r"PASAL",
    },
    "PARAGRAPH": {
        "en": r"PARAGRAPH",
        "fr": r"PARAGRAPHE",
        "es": r"P[AÁ]RRAFO",
        "pt": r"PAR[AÁ]GRAFO",
        "ru": r"ПАРАГРАФ",
        # "PARAGRAF" is under SUBSECTION (Indonesian); the same word was never needed here for the global corpus
    },
}

# Ordinal words that can stand in for a number after the heading word (Roman numerals included).
ORDINAL_WORDS: dict[str, str] = {
    "fr": r"PREMIER|PREMI[EÈ]RE|UNIQUE",
    "es": r"PRIMER[OA]|[ÚU]NIC[OA]",
    "pt": r"PRIMEIR[OA]",
    "sk-cs": r"PRV[ÁYÉ]|PRVN[ÍI]",
    "pl": r"PIERWSZ[YA]",
    "sv": r"F[OÖ]RSTA",
    # "Bagian Kesatu", "Bagian Kedua" ... (Indonesian spells section/paragraph ordinals out)
    "id": (r"KESATU|KEDUA|KETIGA|KEEMPAT|KELIMA|KEENAM|KETUJUH|KEDELAPAN|KESEMBILAN|KESEPULUH|"
           r"KESEBELAS|KEDUA\s+BELAS|KETIGA\s+BELAS|KEEMPAT\s+BELAS|KELIMA\s+BELAS"),
    "roman": r"I|II|III|IV|V|VI|VII|VIII|IX|X",
}

# Number-first labels ("5 straipsnis", "1. člen"): only these languages write the unit word after the number.
POST_WORDS: dict[str, dict[str, str]] = {
    "ARTICLE": {
        "lt": r"STRAIPSNIS",
        "lv": r"PANTS",
        "cs-sk": r"[CČ]L[AÁ]NEK",
        "hr-sl": r"[CČ]LANAK|[CČ]LAN|[CČ]LEN",
        "is": r"GREIN|GR\.",
        "hu": r"SZAKASZ",
        "ru-uk": r"СТАТ[ЬТ]Я|СТАТТЯ",
    },
}

# Case-SENSITIVE, line-anchored words (so Spanish "del 5 de mayo" is not a heading).
# (level, words, number_first)
EXTRA_WORDS: list[tuple[str, str, bool]] = [
    ("PART", r"Dio|DIO|Afsnit|AFSNIT|Teil|TEIL|Del|DEL|Pjesa|PJESA|Avdelning|AVDELNING|Titel|TITEL|Μέρος|ΜΈΡΟΣ|ΜΕΡΟΣ|Բաժին|კარი|ส่วนที่|Igice cya|Ibice bya", False),
    ("PART", r"dalis|daļa|hluti|Rész|Dalis|Daļa|Hluti|DALIS|DAĻA|HLUTI|RÉSZ", True),
    ("CHAPTER", r"Glava|GLAVA|Κεφάλαιο|ΚΕΦΆΛΑΙΟ|Κεφαλαιο|Գլուխ|თავი|Umutwe wa|UMUTWE WA", False),
    ("CHAPTER", r"skyrius|nodaļa|kafli|poglavje|peatükk|Fejezet|kap\.|Skyrius|Nodaļa|Kafli|Poglavje|Peatükk|fejezet|"
                r"SKYRIUS|NODAĻA|KAFLI|POGLAVJE|PEATÜKK|FEJEZET|KAP\.", True),
    ("SECTION", r"Sec\.|SEC\.|Afdeling|AFDELING|Seksioni|SEKSIONI|Sekcija|SEKCIJA|Seção|Secção|SEÇÃO|SECÇÃO|Sezione|SEZIONE", False),
    ("ARTICLE", r"Čl\.|ČL\.|Чл\.|ЧЛ\.|Ingingo ya|INGINGO YA|Ingingo|INGINGO", False),
]
# Roman-numeral-first Nordic/Estonian part marker ("III del", "II osa")
EXTRA_PART_ROMAN_FIRST = r"del|osa|Del|Osa"

# Non-Latin scripts: (level, regex).  Roman-free, so they live here as patterns.
CJK_NUMBER = r"[0-9〇零一二三四五六七八九十百千]{1,7}"
NONLATIN_MARKERS: list[tuple[str, str]] = [
    ("PART", r"第\s*" + CJK_NUMBER + r"\s*[編编篇]"),
    ("CHAPTER", r"第\s*" + CJK_NUMBER + r"\s*章"),
    ("SECTION", r"第\s*" + CJK_NUMBER + r"\s*[節节]"),
    ("ARTICLE", r"第\s*" + CJK_NUMBER + r"\s*[条條]"),
    ("CHAPTER", r"(?m)^\s*หมวด\s*[\d๐-๙]+"),
    ("ARTICLE", r"มาตรา\s*[\d๐-๙]+"),
    ("PART", r"الباب\s*[ء-ي\d]+"),
    ("CHAPTER", r"الفصل\s*\(?[\d٠-٩]+\)?"),
    ("ARTICLE", r"(?:ال)?مادة\s*\(?[\d٠-٩]+\)?"),
    ("ARTICLE", r"Հոդված\s*\d+"),
    ("ARTICLE", r"მუხლი\s*\d+"),
]

# --------------------------------------------------------------------------- #
# Cross-reference vocabulary                                                  #
# --------------------------------------------------------------------------- #
# Right after a label: "Article 31 of this Act ..." is a citation, not a heading.
XREF_TAILS: dict[str, str] = {
    "en": r"of\s+this\b|hereof",
    "es": r"de\s+esta\s+ley|de\s+la\s+presente",
    "fr": r"du\s+pr[ée]sent|de\s+la\s+pr[ée]sente|de\s+cette\s+loi",
    "pt-it": r"da\s+presente|desta\s+lei|do\s+presente|della\s+presente|di\s+questa",
    "de": r"dieses\s+Gesetzes|des\s+vorliegenden",
    "nl": r"van\s+deze\s+wet",
    "no-sv": r"i\s+denne\s+lov|i\s+denna\s+lag|av\s+denne\s+lov",
    "pl": r"niniejszej\s+ustawy|tej\s+ustawy",
    "cs-sk": r"tohoto\s+z[aá]kona|tohto\s+z[aá]kona",
    "hr-sl": r"ovog(?:a)?\s+z[a-z]+|tega\s+zakona",
    "bg-ru-uk": r"от\s+(?:този|настоящия)\s+закон|настоящего\s+Закона|цього\s+Закону",
    "mk": r"од\s+овој\s+закон",
    "el": r"του\s+παρόντος",
    "ro": r"din\s+prezenta",
    "hu": r"(?:e|ezen)\s+t[öo]rv[ée]ny",
    "lt-lv": r"šio\s+įstatymo|šā\s+likuma",
    "et": r"käesoleva",
    "tr": r"bu\s+Kanun",
    "sq": r"i\s+k[eë]tij\s+ligji",
    "rw": r"y[’']iri\s+tegeko",
    "is": r"laga\s+þessara",
    "ar": r"من\s+هذا\s+القانون",
    "hy": r"սույն\s+օրենքի",
    "ka": r"ამ\s+კანონის",
    "th": r"แห่ง",
    "ja-zh": r"[のにをはがでとへや]|から|又は|及び|若しくは|的",
    "id": (r"(?:Undang-Undang|Peraturan|Qanun|Perda|Peraturan\s+Daerah|Perbup|Perwali|Pergub)\s+(?:ini|tersebut)|"
           r"ayat\b|huruf\b|angka\b|dan\s+Pasal\b|sampai\s+dengan\b|s\.?d\.?\s+Pasal\b|atau\s+Pasal\b|jo\.|junto\b"),
    # "Pasal 5 Undang-Undang ini", "Pasal 12 ayat (1)", "Pasal 3 huruf a", "Pasal 7 dan Pasal 8":
    # a Pasal followed by these is a citation inside a sentence, not a heading line
}

# Right before a label (HTML dumps put every inline link on its own line, so a citation can sit at a line start).
# Used per document language; only enabled for the languages listed here.
XREF_PREFIXES: dict[str, tuple[str, ...]] = {
    "nl": ("bedoeld in", "bedoeld", "ingevolge", "krachtens", "overeenkomstig", "onverminderd", "behoudens",
           "in afwijking van", "op grond van", "ter uitvoering van", "met toepassing van",
           "met overeenkomstige toepassing van", "in strijd met", "voor inwerkingtreding van", "waarop",
           "omschreven in", "van", "en", "of", "met", "in", "aan", "voor", "tot", "bij", "onder", "volgens"),
}
# filename marker -> XREF_PREFIXES key (the converter reads the language off the file name)
XREF_PREFIX_FILENAME_TAGS: dict[str, str] = {"Dutch": "nl"}

# --------------------------------------------------------------------------- #
# Table of contents and page furniture                                        #
# --------------------------------------------------------------------------- #
TOC_TITLES: dict[str, tuple[str, ...]] = {
    "en": ("Table of Contents", "Contents", "Arrangement of Sections", "Arrangement of sections"),
    "fr": ("Sommaire", "Table des matières"),
    "es-pt-it": ("Índice", "Indice", "Contenido", "Sumário"),
    "de": ("Inhaltsverzeichnis", "Inhalt"),
    "cs-sk-hr": ("Obsah", "Sadržaj"),
    "pl": ("Spis treści",),
    "ro": ("Cuprins",),
    "ru-uk-bg-mk": ("Оглавление", "Содержание", "Зміст", "Съдържание", "Содржина"),
    "no-da-sv": ("Innhold", "Indhold", "Innehåll"),
    "nl": ("Inhoudsopgave",),
    "et-hu-is": ("Sisukord", "Tartalomjegyzék", "Efnisyfirlit"),
    "lv-lt-sl": ("Saturs", "Turinys", "Kazalo"),
    "tr": ("İçindekiler",),
    "el": ("Πίνακας περιεχομένων",),
    "hy-ka-ar-th": ("Բովանդակություն", "სარჩევი", "الفهرس", "สารบัญ"),
    "zh-ja": ("目录", "目次"),
    "rw-sq": ("Ibikubiye muri iri tegeko", "Ibikubiyemo", "Përmbajtja"),
    "id": ("Daftar Isi", "Isi", "Daftar Pasal"),
}

# the word "page" - only used to recognise page-footer lines ("Halaman 3", "Page 3 of 10")
PAGE_WORDS: dict[str, str] = {
    "en": r"page|pages|pg\.?|p\.",
    "es-pt": r"página|pagina|pág\.?|pag\.?",
    "de": r"seite|s\.",
    "cs-pl-sl-hr": r"strana|str\.|strona|stran|stranica",
    "nordic": r"side|sida|sivu",
    "hu": r"oldal",
    "tr": r"sayfa",
    "et-lv-lt-is": r"lk|lpp\.?|lappuse|puslapis|psl\.?|bls\.?",
    "sq": r"faqe",
    "ru-uk": r"страница|стр\.?|сторінка|стор\.?|страна",
    "el": r"σελίδα|σελ\.?",
    "hy-ka-ar-th": r"էջ|გვერდი|صفحة|หน้า",
    "zh-ja": r"頁|页|ページ",
    "rw": r"urupapuro",
    "id": r"halaman|hal\.",
}


# --------------------------------------------------------------------------- #
# Per-language output conventions                                             #
# --------------------------------------------------------------------------- #
# Languages listed here are converted with these settings; every other language uses the generic
# "relative" depths (levels the file really uses, starting at ``##`` under a ``#`` title).
#   depth     level -> Markdown heading depth (1 = ``#``)
#   min_hits  line-anchored occurrences needed before a level counts as used. The generic gate is 3 for
#             SECTION/SUBSECTION/ARTICLE; a spelled-out ordinal such as "Bagian Kedua" is never a citation, so
#             a short regulation with only two Bagian still keeps them.
#   title     False: no ``# <title>`` line is added (``#`` is the top statute level, as in the existing corpus)
PROFILES: dict[str, dict] = {
    "indonesian": {
        "depth": {"BOOK": 1, "PART": 1, "TITLE": 1, "CHAPTER": 1, "SECTION": 2, "SUBSECTION": 3,
                  "ARTICLE": 4, "PARAGRAPH": 5},
        "min_hits": {"SECTION": 2, "SUBSECTION": 2},
        "title": False,
    },
}


def profile_for(language: str | None) -> dict | None:
    return PROFILES.get((language or "").strip().casefold())


def alternation(by_language: dict[str, str]) -> str:
    """Join every language's words into one regex alternation."""
    return "|".join(by_language.values())
