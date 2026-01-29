DATA_DIR = "data"
RAW_VERSES = f"{DATA_DIR}/verses.csv"
TEMP_VERSES = f"{DATA_DIR}/temp_verses.csv"
PROCESSED_DIR = "data/processed"

TRAIN_DATA = f"{PROCESSED_DIR}/train.csv"
DEV_DATA = f"{PROCESSED_DIR}/dev.csv"
EVAL_DATA = f"{PROCESSED_DIR}/eval.csv"

TRAIN_PAIRS = f"{PROCESSED_DIR}/train_pairs.csv"
DEV_PAIRS = f"{PROCESSED_DIR}/dev_pairs.csv"
EVAL_PAIRS = f"{PROCESSED_DIR}/eval_pairs.csv"

MODEL_NAME = "pranaydeeps/Ancient-Greek-BERT"
MODEL_DIR = "model/"

SEED = 42

TRAIN_RATIO = 0.95
DEV_RATIO = 0.025
BATCH_SIZE = 256
EPOCHS = 8
LEARNING_RATE = 2e-5
EVAL_STEPS = 1000
