DATA_INPUT = "data/verses.csv"
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
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512
EVAL_STEPS = 2000
