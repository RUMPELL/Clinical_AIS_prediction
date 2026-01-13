# Data Schema

This project uses JSONL files where each line is a JSON object.

## ChatML JSONL (intermediate)

Produced by `scripts/ingest_split_excel.py`.

Required fields:
- `prompt`: ChatML-formatted prompt string
- `completion`: ground-truth AIS codes as a comma-separated string

Optional fields:
- `patient_id`: internal identifier (recommended to keep for debugging only)

Example:
```json
{"prompt":"<|im_start|>system\n...","completion":"1234567, 2345678\n<|im_end|>","patient_id":"cnu_..."}
```

## Model-ready JSONL (training/evaluation)

Produced by `scripts/clean_jsonl.py`.

Required fields:
- `text`: cleaned report text (extracted from the ChatML user block)
- `completion`: label string (same content as above, optionally padded)

Example:
```json
{"text":"Trauma CT report ...","completion":"1234567, 2345678\n<|im_end|>"}
```
