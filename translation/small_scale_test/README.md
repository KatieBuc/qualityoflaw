# Translation Small-Scale Test

This project evaluates and compares the quality of regional Indonesian-language translations produced by different translation methods, including Google Translate and LLM-based translation (via web chat interfaces).

## Selected Files

The following source files were used for this test (randomly selected):

- `ACEH_BIREUEN.txt`
- `LAMPUNG_LAMPUNG_TIMUR.txt`
- `NUSA_TENGGARA_TIMUR_TIMOR_TENGAH_UTARA.txt`
- `SUMATERA_BARAT_PADANG_PARIAMAN.txt`
- `JAWA_TENGAH_SEMARANG.txt`

## Folder Structure

```
./translation/small_scale_test
├─ eval                  (evaluation prompt)
└─ test_version          (translation results & evaluation reports)
   ├─ claude
   │  └─ version1        (versions for further testing)
   │  └─ ...
   │  └─ report.json     (evlauation report)
   ├─ gemini
   │  └─ version1
   │  └─ ...
   └─ google_translate
      └─ translated
```

## Test Versions

Translations were generated using the following methods:

- **Google Translate**
- **LLM Translation** (tested via web chat interfaces)
  - 1 prompt version used: `translation/small_scale_test/test_version/prompt1.txt`
  - 2 models tested:
    - Gemini 3.1 Flash Lite
    - Claude Sonnet 4.6

## Evaluation

Translation outputs were assessed using an **LLM-as-a-Judge** approach (via web chat interfaces):

- **Judge Model:** Gemini 3.5 Flash (selected for its larger context window)
- **Evaluation Prompt:** `translation/small_scale_test/eval/prompt.txt`

## Evaluation Criteria & Metrics

The translation quality is assessed across 5 core domains, tailored specifically for government policy and public administration standards. Each domain is scored on a **1 to 5 scale** (5 being Excellent, 1 being Unacceptable).

| Domain | Abbreviation | Focus Area |
| :--- | :--- | :--- |
| **Accuracy & Fidelity** | ACC | Semantic alignment, absence of omissions/distortions, and preservation of policy intent. |
| **Terminology & Domain Appropriateness**| TERM | Correct use of official government jargon, statutory terms, and institutional names. |
| **Tone & Style** | TONE | Formality, objectivity, neutrality, and authoritative presentation. |
| **Fluency & Grammar** | FLUE | Target language syntax, grammatical correctness, and natural readability. |
| **Consistency & Alignment** | CONS | Internal stability of recurring terms and parallel grammatical structures. |

### Scoring Rubric Overview

* **5 (Excellent):** Flawless; meets professional, publication-ready standards for official governance documents.
* **4 (Good):** Highly accurate and formal, with only 1–2 minor, non-critical nuances or stylistic deviations.
* **3 (Satisfactory):** Generally accurate and understandable, but relies on colloquial terms or exhibits a noticeable "machine translation" feel.
* **2 (Poor):** Contains critical mistranslations, informal language, or severe grammatical errors that distort policy mandates.
* **1 (Unacceptable):** Completely unreliable, unreadable, or unrelated to the source context.

## Results

All the report is in JSON format in the corresponding folder

ex. **translation\small_scale_test\test_version\claude\report.json**

### Summary of Evaluation Scores

The table below presents the average scores for each translation method across the five evaluation domains, along with their final overall averages:

| Translation Method | Accuracy (ACC) | Terminology (TERM) | Tone & Style (TONE) | Fluency (FLUE) | Consistency (CONS) | Overall Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Google Translate** | 3.2 | 3.2 | 4.2 | 3.2 | 2.6 | **3.28** |
| **Gemini 3.1 Flash Lite** | 4.2 | 5.0 | 5.0 | 4.6 | 4.0 | **4.56** |
| **Claude Sonnet 4.6** | 4.8 | 5.0 | 5.0 | 5.0 | 4.0 | **4.76** |

**Note: LLM-as-a-Judge result is for reference only.**