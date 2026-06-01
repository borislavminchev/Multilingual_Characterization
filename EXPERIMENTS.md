# Experiment Tracking: Hierarchical Entity Role Classification

## Project Overview

This document tracks all experiments conducted for the Master's thesis on **Multilingual Hierarchical Entity Role Classification**. The system classifies named entities into:
- **Coarse roles**: Protagonist, Antagonist, Innocent (3 classes, single-label)
- **Fine roles**: 22 subtypes across the three coarse categories (multi-label)

## System Architecture

```
Input Text with Entity Markers
        ↓
┌───────────────────────────────┐
│   Base Model (XLM-RoBERTa)    │
│   + Entity Span Pooling       │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│   Coarse Classifier           │
│   (3-class single-label)      │
│   Head: Semantic/MLP          │
│   Loss: FocalLoss             │
└───────────────────────────────┘
        ↓ (soft probs or hard labels)
┌───────────────────────────────┐
│   Fine Classifier             │
│   (22-class multi-label)      │
│   Head: Semantic/MLP          │
│   Conditioning: Soft/Hard/None│
│   Loss: ASL/Focal/BCE         │
└───────────────────────────────┘
        ↓
Final Predictions (Coarse + Fine labels)
```

---

## Current Best Configuration (Baseline E01)

```python
# Model
MODEL_NAME = "xlm-roberta-base"

# Coarse Classifier
COARSE_HEAD_TYPE = 'semantic'
COARSE_LEARNING_RATE = 6e-5
COARSE_NUM_EPOCHS = 10
COARSE_NUM_UNFROZEN_LAYERS = 6
COARSE_FOCAL_GAMMA = 2.0

# Fine Classifier  
FINE_HEAD_TYPE = 'semantic'
FINE_LOSS_TYPE = 'asl_optimized'
ASL_GAMMA_NEG = 2.0
ASL_GAMMA_POS = 0.0
ASL_CLIP = 0.05
USE_SOFT_CONDITIONING = True
CARDINALITY_WEIGHT = 0.3
```

---

## Master Experiment Table

### Key Metrics
| Metric | Description |
|--------|-------------|
| **Coarse Val F1** | Weighted F1 on validation set (3-class) |
| **Coarse Test F1** | Weighted F1 on test set (3-class) |
| **Fine Val μF1** | Micro-F1 for fine labels on validation (oracle coarse) |
| **Fine Test μF1** | Micro-F1 for fine labels on test (oracle coarse) |
| **E2E Fine F1** | End-to-end sample-averaged F1 (errors propagate) |
| **Exact Match** | % samples with ALL labels correct |

---

### Full Experiment Results

| ID | Base Model | Coarse Head | Coarse Loss | Fine Head | Conditioning | Fine Loss | γ_neg | γ_pos | Clip | Card. | Coarse Val F1 | Coarse Test F1 | Fine Val μF1 | Fine Test μF1 | E2E Fine F1 | Exact Match | Status |
|----|------------|-------------|-------------|-----------|--------------|-----------|-------|-------|------|-------|---------------|----------------|--------------|---------------|-------------|-------------|--------|
| **E01** | xlm-roberta-base (6+6 unfrozen) | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | **81.16%** | **79.61%** | **93.03%** | **92.50%** | **50.66%** | **37.09%** | ✅ Baseline |
| E02 | xlm-roberta-large | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | | | | | | | ⬜ |
| E03 | microsoft/deberta-v3-base (3+3)| Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 79.66% | 75.51% | 92.20% | 92.29% | 19.18% | 14.57% | ⬜ |
| E04 | xlm-roberta-base | **MLP** | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 82.83% | 80.74% | 94.03 | 93.91 | 32.34 | 29.30 | ⬜ |
| E05 | xlm-roberta-base | Semantic | Focal | **MLP** | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 82.75 | 79.12 | 92.72 | 92.35 | 49.90 | 38.74 | ⬜ |
| E06 | xlm-roberta-base | **MLP** | Focal | **MLP** | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 82.83% | 80.74% | 94.46% | 94.72% | 35.29% | 31.29% | ⬜ |
| E07 | xlm-roberta-base | Semantic | Focal | Semantic | **Hard** | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.0 | | | | | | | ⬜ |
| E08 | xlm-roberta-base | Semantic | Focal | Semantic | **None** | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.0 | | | | | | | ⬜ |
| E09 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | **Focal** | - | - | - | 0.3 | 82.75 | 79.12 | 95.01 | 95.12 | 28.56 | 22.19| ⬜ |
| E10 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | **ASL** | 2.0 | 1.0 | 0.05 | 0.3 | 82.75 | 79.12 | 94.66 | 93.78 | 51.63 | 42.38 | ⬜ |
| E11 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | **4.0** | **1.0** | 0.05 | 0.3 | 81.24 | 80.61 | 92.62 | 92.46 | 32.36 | 25.99 | ⬜ |
| E12 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | **0.0** | 0.3 | 81.23 | 81.04 | 93.49 | 93.21 | 51.43 | 38.91 | ⬜ |
| E13 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | **0.1** | 0.3 | 81.23 | 81.04 | 94.78 | 94.81 | 20.17 |16.56 | ⬜ |
| E14 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | **0.0** | 81.23 | 81.04 | 94.63 | 94.92 | 25.97 | 22.35 | ⬜ |
| E15 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | **0.5** | 81.23 | 81.04 | 92.95 | 92.69 | 53.33 | 40.73 | ⬜ |
| E16 | xlm-roberta-base + reg | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | | | | | | | ⬜ |
| E17 | xlm-roberta-base | Semantic | **CE** | Semantic | Soft | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 82.13 | 78.05 | 94.73 | 93.94 | 50.29 | 41.06 | ⬜ |
| E18 | xlm-roberta-base | Semantic | **W-CE** | Semantic | Soft | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 81.76 | 81.19 | 94.76 | 94.08 | 50.44 | 41.06 | ⬜ |
| E19 | xlm-roberta-base | Semantic | **CB-CE** | Semantic | Soft | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 80.96 | 81.68 | 94.62 | 93.87 | 50.67 | 44.54 | ⬜ |
| E20 | xlm-roberta-base | Semantic | **CB-CE** | Semantic | HARD | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 81.58 | 82.70 | 67.13 | 67.86 | 31.13 | 0.00 | ⬜ |

---

## Experiment Categories

### A. Base Model Ablation (E01-E03)

**Purpose**: Compare different pre-trained multilingual transformers.

| ID | Model | Parameters | Hypothesis |
|----|-------|------------|------------|
| E01 | xlm-roberta-base | 270M | Baseline - strong multilingual capabilities |
| E02 | xlm-roberta-large | 550M | More capacity may improve performance |
| E03 | bert-base-multilingual-cased | 110M | Lighter model, faster training |

**Config changes for E02**:
```python
MODEL_NAME = "xlm-roberta-large"
```

**Config changes for E03**:
```python
MODEL_NAME = "bert-base-multilingual-cased"
```

---

### B. Classification Head Ablation (E04-E06)

**Purpose**: Compare semantic similarity head vs simple MLP baseline.

| ID | Coarse Head | Fine Head | Hypothesis |
|----|-------------|-----------|------------|
| E01 | Semantic | Semantic | Uses label embeddings as anchors |
| E04 | MLP | Semantic | Simpler coarse head may reduce overfitting |
| E05 | Semantic | MLP | Simpler fine head for multi-label |
| E06 | MLP | MLP | Full baseline without semantic similarity |

**Config changes for E04**:
```python
COARSE_HEAD_TYPE = 'mlp'
FINE_HEAD_TYPE = 'semantic'
```

**Config changes for E05**:
```python
COARSE_HEAD_TYPE = 'semantic'
FINE_HEAD_TYPE = 'mlp'
```

**Config changes for E06**:
```python
COARSE_HEAD_TYPE = 'mlp'
FINE_HEAD_TYPE = 'mlp'
```

---

### C. Hierarchy Conditioning Ablation (E07-E08)

**Purpose**: Compare soft conditioning vs hard masking vs no conditioning.

| ID | Conditioning | Description |
|----|--------------|-------------|
| E01 | Soft | Coarse probs → learned affinity matrix → fine prior |
| E07 | Hard | Binary mask based on predicted coarse label |
| E08 | None | Flat multi-label (no hierarchy) |

**Config changes for E07**:
```python
USE_SOFT_CONDITIONING = False  # Uses FineRoleClassifier with hard mask
```

**Config changes for E08**:
```python
USE_SOFT_CONDITIONING = False
# Modify FineRoleClassifier to not use coarse_labels at all
```

---

### D. Loss Function Ablation (E09-E13)

**Purpose**: Compare different loss functions for multi-label classification.

| ID | Fine Loss | Key Parameters | Hypothesis |
|----|-----------|----------------|------------|
| E01 | ASL-Optimized | γ_neg=2, γ_pos=0, clip=0.05 | Current best |
| E09 | Focal | γ=2.0 | Standard focal loss baseline |
| E10 | ASL (basic) | γ_neg=4, γ_pos=1 | Original ASL without entropy reg |
| E11 | ASL-Optimized | γ_neg=4, γ_pos=1 | Paper default parameters |
| E12 | ASL-Optimized | clip=0.0 | No probability clipping |
| E13 | ASL-Optimized | clip=0.1 | Stronger clipping |

**Config changes for E09**:
```python
FINE_LOSS_TYPE = 'focal'
```

**Config changes for E10**:
```python
FINE_LOSS_TYPE = 'asl'
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
```

**Config changes for E11**:
```python
FINE_LOSS_TYPE = 'asl_optimized'
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
```

---

### E. Cardinality Regularization Ablation (E14-E15)

**Purpose**: Test impact of cardinality constraint on predictions.

| ID | Weight | Target | Hypothesis |
|----|--------|--------|------------|
| E01 | 0.3 | 1.5 | Balanced constraint |
| E14 | 0.0 | - | No cardinality constraint |
| E15 | 0.5 | 1.5 | Stronger constraint |

**Config changes for E14**:
```python
CARDINALITY_WEIGHT = 0.0
```

**Config changes for E15**:
```python
CARDINALITY_WEIGHT = 0.5
```

---

### F. Regularization Ablation (E16)

**Purpose**: Test stronger regularization to reduce overfitting.

| ID | LR | Weight Decay | Dropout | Unfrozen Layers |
|----|-----|--------------|---------|-----------------|
| E01 | 6e-5 | 0.0 | 0.0 | 6 |
| E16 | 2e-5 | 0.01 | 0.25 | 2 |

**Config changes for E16**:
```python
COARSE_LEARNING_RATE = 2e-5
COARSE_WEIGHT_DECAY = 0.01
COARSE_DROPOUT = 0.25
COARSE_NUM_UNFROZEN_LAYERS = 2
```

---

### G. Coarse Loss Function Ablation (E17-E19)

**Purpose**: Compare different loss functions for the coarse classifier (single-label).

| ID | Coarse Loss | Description | Hypothesis |
|----|-------------|-------------|------------|
| E01 | Focal | FocalLoss with γ=2.0 | Handles class imbalance well |
| E17 | CE | Standard CrossEntropyLoss | Simple baseline |
| E18 | Weighted CE | CrossEntropy with inverse frequency weights | Basic class balancing |
| E19 | CB-CE | Class-Balanced CrossEntropy (Cui et al.) | Effective number of samples |

**Config changes for E17**:
```python
COARSE_LOSS_TYPE = 'ce'
```

**Config changes for E18**:
```python
COARSE_LOSS_TYPE = 'weighted_ce'
```

**Config changes for E19**:
```python
COARSE_LOSS_TYPE = 'cb_ce'
```

---

### H. Fine Loss Function Ablation (E20)

**Purpose**: Compare base loss functions for multi-label classification.

| ID | Fine Loss | Description | Hypothesis |
|----|-----------|-------------|------------|
| E01 | ASL-Opt | Asymmetric Loss with entropy regularization | Current best |
| E20 | BCE | Binary CrossEntropy (no rebalancing) | Simple baseline |

**Config changes for E20**:
```python
# Note: Need to add BCE support to FineRoleClassifier
FINE_LOSS_TYPE = 'bce'
```

---

## How to Run Experiments

### 1. Modify Configuration
Edit `src2/config.py` with the appropriate settings for each experiment.

### 2. Train Coarse Classifier
```bash
python src2/train_coarse.py
```

### 3. Train Fine Classifier
```bash
python src2/train_fine.py
```

### 4. Run Inference
```bash
# For soft conditioning
python src2/inference.py --soft

# For hard conditioning
python src2/inference.py
```

### 5. Record Results
Update this document with the results from each experiment.

---

## Key Observations

### Current Performance Analysis (E01)

1. **Coarse Classifier Overfitting**: 
   - Train F1: 94.60% vs Test F1: 78.41% → 16% gap
   - Indicates moderate overfitting despite regularization

2. **Fine Classifier Performs Well in Isolation**:
   - 92.69% Micro-F1 when given correct coarse labels
   - Model learns fine-grained distinctions effectively

3. **Error Propagation Issue**:
   - End-to-end Fine F1 drops to 49.97%
   - ~20% incorrect coarse predictions cascade to fine classifier

4. **Soft Conditioning Benefits**:
   - Using coarse probabilities instead of hard labels
   - Allows uncertainty to propagate through hierarchy

5. **ASL Loss Effectiveness**:
   - γ_neg=2.0, γ_pos=0.0 prevents all-zero predictions
   - Critical for handling sparse multi-label setting

---

## References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. **Class-Balanced Loss**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
3. **Asymmetric Loss**: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
4. **XLM-RoBERTa**: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", ACL 2020

---

## Changelog

| Date | Experiment | Changes | Results |
|------|------------|---------|---------|
| 2026-02-22 | E01 | Initial baseline with soft conditioning | Coarse 78.41%, Fine 92.69%, E2E 49.97% |
| | | | |