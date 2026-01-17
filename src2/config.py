# Define Tier 1 Hyperparameters (learning rate, batch size, epochs). 
# Define Ensemble Configuration (Model IDs for A, B, C). 
# Define File Paths (data_root, taxonomy.json, output directories).

import os

DATA_ROOT = "./data"

TRAIN_DATA_PARENT = os.path.join(DATA_ROOT, "target_4_December_release")
VAL_DATA_PARENT = os.path.join(DATA_ROOT, "cleaned_dev_10_january_2025")
TEST_DATA_PARENT = VAL_DATA_PARENT 

TRAIN_DATA_RATIONALE = "./paragraph/augmented_train.csv"
VAL_DATA_RATIONALE = "./paragraph/augmented_val.csv"
TEST_DATA_RATIONALE = "./paragraph/augmented_test.csv"
TAXONOMY_FILE = os.path.join(DATA_ROOT, "taxonomy.json")

TEST_DATA_RESULTS = os.path.join(DATA_ROOT, "results")
ENSEMBLE_PREDICTION_FILE = "ensemble_predictions_standardized.tsv"