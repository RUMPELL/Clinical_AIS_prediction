# AIS Code Prediction with sLLM (Qwen3-8B + LoRA)

Multi-label AIS code prediction from trauma CT report text using a sequence classification sLLM (Qwen3-8B) fine-tuned with LoRA and an MLP classification head.

## Repository Structure

- `scripts/`: runnable CLI scripts (data prep / cleaning / evaluation)
- `src/aiscode_sllm/`: reusable library code
- `docs/`: documentation (data schema, label rules, etc.)

## Data Availability

The clinical text data used in this project were collected from multiple hospitals and contain sensitive patient-related information.
Due to privacy, ethical, and institutional restrictions, the original training and validation datasets are not publicly available.

However, all data processing, stratified splitting, training, and evaluation pipelines are fully reproducible using the provided codebase.
The repository includes documentation of the expected data schema and preprocessing steps.

## Quickstart (with your own data)

### 1) Build stratified ChatML JSONL from Excel
```bash
python scripts/ingest_split_excel.py \
  --excel ./chu_ct_area.xlsx ./cnu_ct_area.xlsx ./gil_ct_area.xlsx \
  --exclude-gold-xlsx ./GOLD_LABEL.xlsx \
  --exclude-ids-json ./combined_ids.json \
  --out-train train_full_strat.jsonl \
  --out-valid ais_valid_strat.jsonl
```

### 2) Convert ChatML JSONL to model-ready JSONL
```bash
python scripts/clean_jsonl.py --src train_full_strat.jsonl --dst train_full_preproc.jsonl --pad-codes
python scripts/clean_jsonl.py --src ais_valid_strat.jsonl --dst ais_valid_preproc.jsonl --pad-codes
```

### 3) Evaluate a trained checkpoint
```bash
python scripts/evaluate.py \
  --data ais_valid_preproc.jsonl \
  --code2idx code2idx_3lvl.json \
  --model-dir /path/to/training_output \
  --adapter-dir /path/to/training_output/cls_lora_adapter \
  --device cuda:0 \
  --threshold 0.3 \
  --out-metrics metrics.json \
  --out-pred predictions.jsonl
```

### 4) Train (reproduce the original run)

The training script mirrors the author's original settings (MLP head + LoRA + weighted BCE).

Using a YAML config is recommended:

```bash
python scripts/train.py --config configs/train.yaml
```

Or via explicit CLI:

```bash
python scripts/train.py \
  --train-file train_full_preproc.jsonl \
  --val-file ais_valid_preproc.jsonl \
  --code2idx code2idx_3lvl.json \
  --output-dir qwen3_8b_ais_lora_onehot_4digit_mlphead_a128_epoch10
```

### 5) External validation (CBNU-style Excel)

The CBNU export used in this project stores GT AIS codes as a Python-list string column:
`de_list_외` (e.g., `['6504182','6504182']`).

If your Excel contains a text column, specify it via `--text-col` (or in the YAML).

```bash
python scripts/external_cbnu_infer.py --config configs/external_cbnu.yaml \
  --excel /path/to/cbnu.xlsx \
  --text-col <YOUR_TEXT_COLUMN_NAME>
```

If the Excel does **not** include report text (GT-only export / sanity check):

```bash
python scripts/external_cbnu_infer.py --excel /path/to/cbnu.xlsx --gt-only
```

## Notes

- This repo assumes `code2idx_3lvl.json` defines the target label space.
- The evaluation script uses a fixed decision threshold and a top-k fallback when no label passes the threshold.
