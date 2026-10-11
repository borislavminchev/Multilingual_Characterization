# Central place for paths, hyperparameters and settings of both classifiers.

import os

# Data paths
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

# Checkpoint paths
CHECKPOINT_ROOT = "./checkpoints"
COARSE_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "coarse_classifier")
FINE_CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "fine_classifier")

# Prediction paths
PREDICTIONS_ROOT = "./predictions"
COARSE_PREDICTIONS_TRAIN = os.path.join(PREDICTIONS_ROOT, "coarse_predictions_train.csv")
COARSE_PREDICTIONS_VAL = os.path.join(PREDICTIONS_ROOT, "coarse_predictions_val.csv")
COARSE_PREDICTIONS_TEST = os.path.join(PREDICTIONS_ROOT, "coarse_predictions_test.csv")
FINAL_PREDICTIONS_PATH = os.path.join(PREDICTIONS_ROOT, "final_predictions.csv")

# Model configuration
MODEL_NAME = "xlm-roberta-base"  # multilingual backbone covering all dataset languages
MAX_LENGTH = 512

# Coarse classifier hyperparameters
COARSE_NUM_EPOCHS = 10
COARSE_BATCH_SIZE = 9
COARSE_LEARNING_RATE = 6e-5
COARSE_WARMUP_RATIO = 0.1
COARSE_WEIGHT_DECAY = 0.0
COARSE_NUM_UNFROZEN_LAYERS = 6
COARSE_DROPOUT = 0.0
COARSE_FOCAL_GAMMA = 2.0
COARSE_CLASS_BALANCE_BETA = 0.9999

# Head type: 'semantic' uses label-name embeddings as anchors; 'mlp' is a plain linear head
COARSE_HEAD_TYPE = 'semantic'

# Loss type: 'focal' | 'ce' | 'weighted_ce' | 'cb_ce' (class-balanced, Cui et al. 2019)
COARSE_LOSS_TYPE = 'cb_ce'

# Fine classifier hyperparameters
FINE_NUM_EPOCHS = 10
FINE_BATCH_SIZE = 6
FINE_LEARNING_RATE = 8e-5
FINE_WARMUP_STEPS = 50
FINE_NUM_UNFROZEN_LAYERS = 6
FINE_FOCAL_GAMMA = 2.0
FINE_CLASS_BALANCE_BETA = 0.9999

# Head type: 'semantic' or 'mlp' (see coarse head above)
FINE_HEAD_TYPE = 'semantic'

# Multi-label decision rule
FINE_THRESHOLD = 0.25          # primary probability threshold
FINE_GAP_RATIO = 0.7           # adaptive threshold as a ratio of the top probability
FINE_MIN_LABELS = 1            # always keep at least this many labels
FINE_MAX_LABELS = 3            # cap labels per sample

# Asymmetric Loss (Ben-Baruch et al., 2021)
FINE_LOSS_TYPE = 'asl'          # 'focal' | 'asl' | 'asl_optimized'
ASL_GAMMA_NEG = 2.0             # focusing on hard negatives
ASL_GAMMA_POS = 0.0             # 0 = never down-weight positives (sparse labels)
ASL_CLIP = 0.05                 # probability margin that ignores very easy negatives
ASL_ENTROPY_WEIGHT = 0.15       # pushes predictions to be sharper

# Soft hierarchy conditioning
USE_SOFT_CONDITIONING = True    # pass coarse probabilities into the fine classifier
CARDINALITY_WEIGHT = 0.3        # penalty on deviating from the expected label count
TARGET_CARDINALITY = 1.5        # expected average number of fine labels per entity

# Taxonomy sizes
NUM_COARSE_LABELS = 3  # Protagonist, Antagonist, Innocent
NUM_FINE_LABELS = 22
