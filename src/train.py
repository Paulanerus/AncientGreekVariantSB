import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DEV_DATA,
    EPOCHS,
    EVAL_DATA,
    EVAL_STEPS,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_NAME,
    SEED,
    TRAIN_PAIRS,
)
from utils.check_cuda import check_cuda_and_gpus


def build_retrieval_evaluator(df):
    df = df.reset_index(drop=True)
    corpus = {str(i): row.text for i, row in df.iterrows()}
    grouped = df.groupby("nkv_group").apply(
        lambda x: list(x.index), include_groups=False
    )

    queries = {}
    relevant_docs = {}
    for indices in grouped:
        if len(indices) < 2:
            continue
        query_id = str(indices[0])
        queries[query_id] = corpus[query_id]
        relevant_docs[query_id] = {str(idx) for idx in indices[1:]}

    return InformationRetrievalEvaluator(queries, corpus, relevant_docs, name="dev")


torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Loading train pairs from {TRAIN_PAIRS}")
train_pairs = pd.read_csv(TRAIN_PAIRS)
print(f"Loading dev data from {DEV_DATA}")
dev_df = pd.read_csv(DEV_DATA)
print(f"Loading eval data from {EVAL_DATA}")
eval_df = pd.read_csv(EVAL_DATA)

print(f"Train pairs: {len(train_pairs)}")
print(f"Dev rows: {len(dev_df)}")
print(f"Eval rows: {len(eval_df)}")

train_examples = [
    InputExample(texts=[row.sentence1, row.sentence2])
    for row in train_pairs.itertuples(index=False)
]
train_loader = DataLoader(train_examples, batch_size=BATCH_SIZE, shuffle=True)
print(f"Steps per epoch: {len(train_loader)}")

device = check_cuda_and_gpus()
print(f"Device: {device}")

model = SentenceTransformer(MODEL_NAME, device=device)

loss = losses.MultipleNegativesRankingLoss(model)
evaluator = build_retrieval_evaluator(dev_df)
eval_evaluator = build_retrieval_evaluator(eval_df)

warmup_steps = int(len(train_loader) * EPOCHS * 0.1)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Training for epochs: {EPOCHS}")
print(f"Eval steps: {EVAL_STEPS}")
print(f"Warmup steps: {warmup_steps}")

model.fit(
    train_objectives=[(train_loader, loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    evaluation_steps=EVAL_STEPS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LEARNING_RATE},
    output_path=MODEL_DIR,
    save_best_model=True,
    show_progress_bar=True,
)

model = SentenceTransformer(MODEL_DIR, device=device)

total_steps = len(train_loader) * EPOCHS
score = eval_evaluator(model, output_path=None, epoch=EPOCHS, steps=total_steps)
print(f"Eval score: {score}")
