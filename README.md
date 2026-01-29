# Ancient Greek Variant SBERT

An embedding model for Ancient Greek biblical verse similarity and variant clustering.  
It produces 768‑dimensional sentence embeddings (SentenceTransformers) that work well for:

- semantic similarity / duplicate & near-duplicate detection
- clustering verse variants
- semantic search over verse corpora

Base model: [`pranaydeeps/Ancient-Greek-BERT`](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)

Model on Hugging Face: [Paulanerus/AncientGreekVariantSBERT](https://huggingface.co/Paulanerus/AncientGreekVariantSBERT)

---

## Repository

This repo contains:

- a data preparation pipeline (`src/prepare.py`) that builds train/dev/eval splits and training pairs
- a training script (`src/train.py`) using `MultipleNegativesRankingLoss`
- a small inference script (`src/infer.py`) that prints cosine similarities for sentence pairs
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

### Dataset

Download `verses.csv` from Version 4 (released July 2, 2025) of this [Zenodo](https://zenodo.org/records/15789063) dataset.

1. Download `verses.csv`
2. Place it in `data/` (path: `data/verses.csv`)
3. Run `./run.sh prepare`

This creates an intermediate (cleaned/normalized) dataset and then generates the train/dev/eval splits and training pairs used for training.

`prepare.py` reads the raw verses CSV configured in `src/config.py` (`RAW_VERSES`, default `data/verses.csv`) with these columns:

- `verse_id`: verse identifier (used for stable ordering + de-duplication)
- `text`: the verse text
- `nkv`: verse identifier by NKV scheme (used as the grouping key indicating which verses belong to the same underlying verse)

It first generates `data/temp_verses.csv` (see `TEMP_VERSES` in `src/config.py`) where `nkv` is converted to an integer `nkv_group`.

What it does:

1. Reads `data/verses.csv`, de-duplicates rows, and writes `data/temp_verses.csv`
2. Drops rows with missing `text` / `nkv_group`
3. Normalizes `text` (accent-stripping + lowercasing)
4. Removes groups with fewer than 2 samples
5. Splits by group id into train/dev/eval (defaults: 95% / 2.5% / 2.5%)
6. Writes split CSVs (`data/processed/{train,dev,eval}.csv`) and also generates training pairs (`data/processed/{train,dev,eval}_pairs.csv`)

Important: **running `prepare` is required** before training, because `train.py` reads the prepared pairs/splits.

### Pair generation (positives only)

The generated `*_pairs.csv` files contain only positive pairs: both sentences come from the same `nkv_group` (i.e., they are variants of the same underlying verse). For each verse text in a group, `prepare.py` samples one other text from that same group to form a pair.

No explicit negative pairs are written. During training, `MultipleNegativesRankingLoss` uses in-batch negatives: for a given anchor sentence, the other sentences in the batch are treated as negatives.

For dev/eval, the `InformationRetrievalEvaluator` is built from the split CSVs: it treats items with the same `nkv_group` as relevant documents for a query.

---

## Training (`src/train.py`)

Training uses SentenceTransformers with:

- **Loss**: `MultipleNegativesRankingLoss`
- **Evaluator**: `InformationRetrievalEvaluator` built from the dev/eval splits
- **Normalization**: performed in `prepare.py` (accent stripping + lowercase)
- **Saving**: `save_best_model=True`

Training/data configuration (paths, split ratios, batch size, epochs, LR, etc.) lives in `src/config.py`.

Hardware note: training for the released model was run on a single NVIDIA A100 80GB PCIe.

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
  howpublished = {\url{https://huggingface.co/Paulanerus/AncientGreekVariantSBERT}},
  note = {Model release}
}
```

---

## Acknowledgments

This work builds on:

- [`pranaydeeps/Ancient-Greek-BERT`](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT) (Singh, Rutten, and Lefever, 2021)
- the SentenceTransformers ecosystem: https://www.sbert.net

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) 513300936.
