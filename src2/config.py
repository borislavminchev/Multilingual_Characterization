# Define Tier 1 Hyperparameters (learning rate, batch size, epochs). 
# Define Ensemble Configuration (Model IDs for A, B, C). 
# Define File Paths (data_root, taxonomy.json, output directories).

import os

# =============================================================================
# DATA PATHS
# =============================================================================
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

# =============================================================================
# CHECKPOINT PATHS
# =============================================================================
CHECKPOINT_ROOT = "./checkpoints"
COARSE_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "coarse_classifier")
FINE_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "fine_classifier")

# =============================================================================
# PREDICTION PATHS
# =============================================================================
PREDICTIONS_ROOT = "./predictions"
COARSE_PREDICTIONS_TRAIN = os.path.join(PREDICTIONS_ROOT, "coarse_predictions_train.csv")
COARSE_PREDICTIONS_VAL = os.path.join(PREDICTIONS_ROOT, "coarse_predictions_val.csv")
COARSE_PREDICTIONS_TEST = os.path.join(PREDICTIONS_ROOT, "coarse_predictions_test.csv")
FINAL_PREDICTIONS_PATH = os.path.join(PREDICTIONS_ROOT, "final_predictions.csv")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512

# =============================================================================
# COARSE CLASSIFIER HYPERPARAMETERS
# =============================================================================
COARSE_NUM_EPOCHS = 8
COARSE_BATCH_SIZE = 4
COARSE_LEARNING_RATE = 5e-5
COARSE_WARMUP_STEPS = 100
COARSE_NUM_UNFROZEN_LAYERS = 3
COARSE_FOCAL_GAMMA = 2.0
COARSE_CLASS_BALANCE_BETA = 0.9999

# =============================================================================
# FINE CLASSIFIER HYPERPARAMETERS
# =============================================================================
FINE_NUM_EPOCHS = 10
FINE_BATCH_SIZE = 4
FINE_LEARNING_RATE = 3e-5
FINE_WARMUP_STEPS = 50
FINE_NUM_UNFROZEN_LAYERS = 3
FINE_FOCAL_GAMMA = 2.0
FINE_CLASS_BALANCE_BETA = 0.9999
FINE_THRESHOLD = 0.5  # Threshold for multi-label classification

# =============================================================================
# TAXONOMY CONSTANTS
# =============================================================================
NUM_COARSE_LABELS = 3  # Protagonist, Antagonist, Innocent
NUM_FINE_LABELS = 22   # Total subtypes across all coarse categories
