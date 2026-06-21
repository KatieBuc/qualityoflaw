## Selected Files

The following source files were used for this test (randomly selected):

- `ACEH_BIREUEN.txt`
- `LAMPUNG_LAMPUNG_TIMUR.txt`
- `NUSA_TENGGARA_TIMUR_TIMOR_TENGAH_UTARA.txt`
- `SUMATERA_BARAT_PADANG_PARIAMAN.txt`
- `JAWA_TENGAH_SEMARANG.txt`


## Folder Structure

```
├─v1
│  ├─prompts    (prompts file for LLM evaluation)
│  └─result     (final reports of LLM evaluation)
```

## How to Run

1. **Add the `.env` file** to the project root folder.
2. **Install the dependencies**:
```bash
pip install python-dotenv pydantic openai

```


3. **Run the script** to automatically generate the report:
```bash
python -m quality_eval.run_eval_azure -p data\raw\localpolicies\ACEH_BIREUEN.txt -o quality_eval\v1\result\ACEH_BIREUEN.json -f quality_eval\v1\prompts

```



To get more information, run:

```bash
python -m quality_eval.run_eval_azure --help

```
