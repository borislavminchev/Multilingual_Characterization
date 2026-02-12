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
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 512

# =============================================================================
# COARSE CLASSIFIER HYPERPARAMETERS
# =============================================================================
COARSE_NUM_EPOCHS = 10              # Increased from 8 for better convergence
COARSE_BATCH_SIZE = 8               # Increased from 6 for more stable gradients
COARSE_LEARNING_RATE = 3e-5         # Reduced from 1e-4 for more stable fine-tuning
COARSE_WARMUP_RATIO = 0.1           # Warmup ratio (10% of total steps)
COARSE_NUM_UNFROZEN_LAYERS = 5      # Keep at 5 (per user request)
COARSE_FOCAL_GAMMA = 1.5            # Slightly reduced from 2.0 for less aggressive focusing
COARSE_CLASS_BALANCE_BETA = 0.9999
COARSE_LABEL_SMOOTHING = 0.1        # New: label smoothing to prevent overconfidence
COARSE_USE_CLS_HEAD = True          # New: use standard classification head (simpler, often better)

# =============================================================================
# FINE CLASSIFIER HYPERPARAMETERS
# =============================================================================
FINE_NUM_EPOCHS = 10
FINE_BATCH_SIZE = 6
FINE_LEARNING_RATE = 8e-5
FINE_WARMUP_STEPS = 50
FINE_NUM_UNFROZEN_LAYERS = 6
FINE_FOCAL_GAMMA = 2.0
FINE_CLASS_BALANCE_BETA = 0.9999

# Fine prediction parameters (smart threshold strategy)
FINE_THRESHOLD = 0.25           # Primary threshold for multi-label classification
FINE_GAP_RATIO = 0.7           # Adaptive threshold: ratio of top probability
FINE_MIN_LABELS = 1            # Guarantee at least this many predictions
FINE_MAX_LABELS = 3            # Cap maximum predictions per sample

# =============================================================================
# ASYMMETRIC LOSS (ASL) PARAMETERS
# =============================================================================
# Loss type: 'focal', 'asl', or 'asl_optimized' (recommended)
FINE_LOSS_TYPE = 'asl_optimized'

# ASL hyperparameters (Ben-Baruch et al., 2021)
# gamma_neg: Focusing parameter for negative samples (higher = more focus on hard negatives)
#            Reduced from 4.0 to 2.0 to prevent model collapse to all-zeros
ASL_GAMMA_NEG = 2.0

# gamma_pos: Focusing parameter for positive samples (lower = preserve easy positives)
#            Set to 0.0 to NEVER down-weight positive samples (critical for sparse labels)
ASL_GAMMA_POS = 0.0

# clip: Probability margin for hard thresholding negatives
#       Shifts negative probabilities to ignore very easy negatives
#       Recommended: 0.05 (5% margin)
ASL_CLIP = 0.05

# entropy_weight: Regularization weight to encourage sharper predictions
#                 Higher values = stronger push toward confident (peaky) predictions
#                 Recommended: 0.1-0.3 for sparse multi-label classification
ASL_ENTROPY_WEIGHT = 0.15

# =============================================================================
# SOFT HIERARCHY CONDITIONING PARAMETERS
# =============================================================================
# Enable soft conditioning (pass coarse probabilities to fine classifier)
USE_SOFT_CONDITIONING = True

# Cardinality regularization weight
# Penalizes deviation from expected number of labels (1-3)
# Higher values = stronger constraint on number of predictions
CARDINALITY_WEIGHT = 0.3

# Target cardinality (expected average number of fine labels)
# Based on data distribution: most samples have 1-3 fine labels
TARGET_CARDINALITY = 1.5

# =============================================================================
# TAXONOMY CONSTANTS
# =============================================================================
NUM_COARSE_LABELS = 3  # Protagonist, Antagonist, Innocent
NUM_FINE_LABELS = 22   # Total subtypes across all coarse categories
