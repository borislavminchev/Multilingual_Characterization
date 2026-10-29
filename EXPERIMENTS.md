# Experiment Tracking: Hierarchical Entity Role Classification

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

### Full Experiment Results

| ID | Base Model | Coarse Head | Coarse Loss | Fine Head | Conditioning | Fine Loss | γ_neg | γ_pos | Clip | Card. | Coarse Val F1 | Coarse Test F1 | Fine Val μF1 | Fine Test μF1 | E2E Fine F1 | Exact Match | Status |
|----|------------|-------------|-------------|-----------|--------------|-----------|-------|-------|------|-------|---------------|----------------|--------------|---------------|-------------|-------------|--------|
| E01 | xlm-roberta-base (6+6 unfrozen) | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 81.16% | 79.61% | 93.03% | 92.50% | 50.66% | 37.09% | ⬜ |
| E02 | microsoft/deberta-v3-base (3+3)| Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 79.66% | 75.51% | 92.20% | 92.29% | 19.18% | 14.57% | ⬜ |
| E03 | xlm-roberta-base | **MLP** | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 82.83% | 80.74% | 94.03 | 93.91 | 32.34 | 29.30 | ⬜ |
| E04 | xlm-roberta-base | Semantic | Focal | **MLP** | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 82.75 | 79.12 | 92.72 | 92.35 | 49.90 | 38.74 | ⬜ |
| E05 | xlm-roberta-base | **MLP** | Focal | **MLP** | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | 0.3 | 82.83% | 80.74% | 94.46% | 94.72% | 35.29% | 31.29% | ⬜ |
| E06 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | **Focal** | - | - | - | 0.3 | 82.75 | 79.12 | 95.01 | 95.12 | 28.56 | 22.19| ⬜ |
| E07 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | **ASL** | 2.0 | 1.0 | 0.05 | 0.3 | 82.75 | 79.12 | 94.66 | 93.78 | 51.63 | 42.38 | ⬜ |
| E08 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | **4.0** | **1.0** | 0.05 | 0.3 | 81.24 | 80.61 | 92.62 | 92.46 | 32.36 | 25.99 | ⬜ |
| E09 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | **0.0** | 0.3 | 81.23 | 81.04 | 93.49 | 93.21 | 51.43 | 38.91 | ⬜ |
| E10 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | **0.1** | 0.3 | 81.23 | 81.04 | 94.78 | 94.81 | 20.17 |16.56 | ⬜ |
| E11 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | **0.0** | 81.23 | 81.04 | 94.63 | 94.92 | 25.97 | 22.35 | ⬜ |
| E12 | xlm-roberta-base | Semantic | Focal | Semantic | Soft | ASL-Opt | 2.0 | 0.0 | 0.05 | **0.5** | 81.23 | 81.04 | 92.95 | 92.69 | 53.33 | 40.73 | ⬜ |
| E13 | xlm-roberta-base | Semantic | **CE** | Semantic | Soft | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 82.13 | 78.05 | 94.73 | 93.94 | 50.29 | 41.06 | ⬜ |
| E14 | xlm-roberta-base | Semantic | **W-CE** | Semantic | Soft | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 81.76 | 81.19 | 94.76 | 94.08 | 50.44 | 41.06 | ⬜ |
| E15 | xlm-roberta-base | Semantic | **CB-CE** | Semantic | Soft | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 80.96 | 81.68 | 94.62 | 93.87 | 50.67 | 44.54 | ⬜ |
| E16 | xlm-roberta-base | Semantic | **CB-CE** | Semantic | HARD | ASL | 2.0 | 0.0 | 0.05 | 0.3 | 81.58 | 82.70 | 67.13 | 67.86 | 31.13 | 0.00 | ⬜ |

---

## Configuration

Every experiment is defined entirely by `src2/config.py`. There are no command-line
hyperparameters for training - to reproduce a row from the table above, edit the relevant
values in that file and run the pipeline. 

---

## Running the Pipeline

The system is trained and evaluated in three sequential stages. Each stage reads its settings
from `config.py` and writes checkpoints/predictions that the next stage consumes, so they must
be run in order. Run the commands from inside the `src2/` directory.

### Stage 1 - Train the coarse classifier
Trains the 3-class coarse classifier and writes its predicted coarse probabilities for the
train/val/test splits (these feed the fine classifier).

```bash
cd src2
python train_coarse.py
```

### Stage 2 - Train the fine classifier
Trains the 22-label multi-label fine classifier on top of the coarse outputs from Stage 1.

```bash
python train_fine.py
```

### Stage 3 - End-to-end inference and evaluation
Combines both classifiers in a cascade and produces the final predictions, printing the
end-to-end metrics (Coarse Accuracy, Fine Sample-F1, Exact Match).

```bash
# Uses the mode from config.py (USE_SOFT_CONDITIONING)
python inference.py

# Or force a mode explicitly:
python inference.py --soft    # soft hierarchy conditioning
python inference.py --hard    # hard masking
```

---

## Demo Application

An interactive Streamlit app demonstrates the trained system on arbitrary text in any of the
five languages and includes the word-attribution (saliency) module. It requires the checkpoints
produced by Stages 1 and 2.

```bash
cd src2
streamlit run demo.py
```

This launches a local web server and opens the app in the browser.
