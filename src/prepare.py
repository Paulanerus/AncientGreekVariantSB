import csv
import os

import numpy as np
import pandas as pd

from config import (
    DATA_INPUT,
    DEV_PAIRS,
    DEV_RATIO,
    EVAL_PAIRS,
    PROCESSED_DIR,
    SEED,
    TRAIN_PAIRS,
    TRAIN_RATIO,
)
from utils.string_norm import strip_accents_and_lowercase


def split_groups(groups, train_ratio, dev_ratio, seed):
    rng = np.random.default_rng(seed)
    groups = np.array(groups, dtype=object)
    rng.shuffle(groups)
    total = len(groups)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)
    train_groups = set(groups[:train_size])
    dev_groups = set(groups[train_size : train_size + dev_size])
    eval_groups = set(groups[train_size + dev_size :])
    return train_groups, dev_groups, eval_groups


def prepare_split(df, groups):
    return df[df["nkv_group"].isin(groups)][["nkv_group", "text"]]


def write_pairs(split_df, output_path, seed):
    rng = np.random.default_rng(seed)
    pair_count = 0
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence1", "sentence2"])
        for _, group in split_df.groupby("nkv_group"):
            texts = group["text"].tolist()
            if len(texts) < 2:
                continue
            for i, text in enumerate(texts):
                other_indices = [j for j in range(len(texts)) if j != i]
                idx = rng.choice(other_indices)
                writer.writerow([text, texts[idx]])
                pair_count += 1
    print(f"Wrote {pair_count} pairs to {output_path}")


print(f"Loading data from {DATA_INPUT}")
df = pd.read_csv(DATA_INPUT)
df = df.dropna(subset=["text", "nkv_group"])
print(f"Rows after dropna: {len(df)}")

df["text"] = df["text"].astype(str).map(strip_accents_and_lowercase)

counts = df["nkv_group"].value_counts()
valid_groups = counts[counts >= 2].index
print(f"Groups with >=2 samples: {len(valid_groups)}")
df = df[df["nkv_group"].isin(valid_groups)]
print(f"Rows after singleton drop: {len(df)}")

train_groups, dev_groups, eval_groups = split_groups(
    valid_groups, TRAIN_RATIO, DEV_RATIO, SEED
)
print(
    "Split groups -> "
    f"train: {len(train_groups)}, dev: {len(dev_groups)}, eval: {len(eval_groups)}"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
train_df = prepare_split(df, train_groups)
dev_df = prepare_split(df, dev_groups)
eval_df = prepare_split(df, eval_groups)

train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
dev_df.to_csv(os.path.join(PROCESSED_DIR, "dev.csv"), index=False)
eval_df.to_csv(os.path.join(PROCESSED_DIR, "eval.csv"), index=False)
print(f"Wrote splits to {PROCESSED_DIR}")

write_pairs(train_df, TRAIN_PAIRS, SEED)
write_pairs(dev_df, DEV_PAIRS, SEED)
write_pairs(eval_df, EVAL_PAIRS, SEED)
