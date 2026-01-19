# Ancient Greek Variant SBERT

An embedding model for **Ancient Greek biblical verse similarity** and **variant clustering**.  
It produces 768‑dimensional sentence embeddings (SentenceTransformers) that work well for:

- semantic similarity / duplicate & near-duplicate detection
- clustering verse variants
- semantic search over verse corpora

Base model: [`pranaydeeps/Ancient-Greek-BERT`](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)

Model on Hugging Face: `https://huggingface.co/Paulanerus/AncientGreekVariantSBERT`

---

## Repository

This repo contains:

- a **data preparation pipeline** (`src/prepare.py`) that builds train/dev/eval splits and training pairs
- a **training script** (`src/train.py`) using `MultipleNegativesRankingLoss`
- a small **inference script** (`src/infer.py`) that prints cosine similarities for sentence pairs
- a convenience entrypoint script `run.sh`

---

## Setup

Make the helper script executable, create a virtual environment, and install dependencies:

```bash
# Make the main helper script executable
chmod +x run.sh

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages (example CUDA build; adjust to your system)
pip install torch==2.9.1+cu126 \
  transformers==4.57.1 \
  sentence-transformers==5.1.2 \
  numpy==2.3.5 \
  pandas==2.3.3 \
  scipy==1.16.3 \
  datasets==4.4.1 \
  accelerate==1.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

You can then use `run.sh`:

- `./run.sh prepare` – data prep (required before training)
- `./run.sh train` – fine-tune the model
- `./run.sh infer` – run inference using the trained model in `model/`

---

## Data preparation (`src/prepare.py`)

`prepare.py` expects a CSV at the path configured in `src/config.py` (`DATA_INPUT`, default `data/verses.csv`) with at least:

- `text`: the verse text
- `nkv_group`: a group id indicating which verses are variants of the same underlying verse

What it does:

1. Drops rows with missing `text` / `nkv_group`
2. Normalizes `text` (accent-stripping + lowercasing)
3. Removes groups with fewer than 2 samples
4. Splits by **group id** into train/dev/eval (defaults: 95% / 2.5% / 2.5%)
5. Writes split CSVs and also generates training pairs (`train_pairs.csv`, etc.)

Important: **running `prepare` is required** before training, because `train.py` reads the prepared pairs/splits.

> Data note: the dataset is not provided in this repo at the moment. The preparation pipeline is included so you can reproduce the splits/pairs once the data (or a public equivalent) is available.

---

## Training (`src/train.py`)

Training uses SentenceTransformers with:

- **Loss**: `MultipleNegativesRankingLoss`
- **Evaluator**: `InformationRetrievalEvaluator` built from the dev/eval splits
- **Normalization**: performed in `prepare.py` (accent stripping + lowercase)
- **Saving**: `save_best_model=True`

Hardware note: training for the released model was run on **a single NVIDIA A100 80GB PCIe**.

Run:

```bash
./run.sh train
```

Outputs:

- model artifacts written to `model/` (see `MODEL_DIR` in `src/config.py`)

---

## Inference (`src/infer.py`)

`infer.py` loads the model and prints cosine similarities between paired lists of verses, after normalizing the text.

Run:

```bash
./run.sh infer
```

This is useful for quickly sanity-checking that:

- near-identical verses score very high
- verses with additions/omissions still score relatively high (depending on overlap)
- unrelated verses score lower

---

## Citation

If you use this model, please cite it as a model release:

```bibtex
@misc{ancient-greek-variant-sbert,
  author = {Fröhlich, Paul},
  title = {Ancient Greek Variant SBERT: Fine-tuned Embeddings for Biblical text verses in Ancient Greek},
  year = {2026},
  howpublished = {\\url{https://huggingface.co/Paulanerus/AncientGreekVariantSBERT}},
  note = {Model release}
}
```

---

## Acknowledgments

This work builds on:

- [`pranaydeeps/Ancient-Greek-BERT`](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT) (Singh, Rutten, and Lefever, 2021)
- the SentenceTransformers ecosystem: https://www.sbert.net
