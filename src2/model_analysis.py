"""
Model Training Analysis Diagrams for SemEval-2025 Task 10 ST1 (Entity Framing)

Reads trainer_state.json from HuggingFace checkpoints and generates:
  B1. Training loss curves (coarse + fine)
  B2. Evaluation metrics over training
  B3. Learning rate schedule
  B4. Gradient norm over training
  B5. Fine classifier precision/recall/F1 + prediction count

Usage:
    cd src2 && python model_analysis.py

Checkpoints expected at project root:
    checkpoints/coarse_classifier/checkpoint-{step}/trainer_state.json
    checkpoints/fine_classifier_soft/checkpoint-{step}/trainer_state.json
"""

import os
import json
import re
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COARSE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "coarse_classifier")
FINE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "fine_classifier_soft")
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, "predictions", "final_predictions.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "diagrams", "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Taxonomy labels ordered by coarse parent
COARSE_LABELS = ["Protagonist", "Antagonist", "Innocent"]
FINE_LABELS_BY_COARSE = {
    "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
    "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
    "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"],
}
FINE_LABELS_ORDERED = []
for _c in COARSE_LABELS:
    FINE_LABELS_ORDERED.extend(FINE_LABELS_BY_COARSE[_c])
COARSE_FOR_FINE = {}
for _c, _fines in FINE_LABELS_BY_COARSE.items():
    for _f in _fines:
        COARSE_FOR_FINE[_f] = _c


# =============================================================================
# DATA LOADING
# =============================================================================

def find_last_checkpoint(checkpoint_dir):
    """Find the checkpoint subdirectory with the highest step number."""
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        m = re.match(r'checkpoint-(\d+)$', name)
        if m and os.path.isdir(os.path.join(checkpoint_dir, name)):
            checkpoints.append((int(m.group(1)), name))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return os.path.join(checkpoint_dir, checkpoints[-1][1])


def load_training_history(checkpoint_dir):
    """
    Load the full training history from the last checkpoint's trainer_state.json.

    Returns:
        train_df: DataFrame with training logs (loss, grad_norm, learning_rate, step, epoch)
        eval_df:  DataFrame with evaluation logs (eval_*, step, epoch)
        state:    Full trainer_state dict (for best_model_checkpoint, etc.)
    Returns (None, None, None) if checkpoints not found.
    """
    last_ckpt = find_last_checkpoint(checkpoint_dir)
    if last_ckpt is None:
        return None, None, None

    state_path = os.path.join(last_ckpt, "trainer_state.json")
    if not os.path.isfile(state_path):
        print(f"  Warning: {state_path} not found")
        return None, None, None

    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        return None, None, None

    # Separate train logs (have 'loss') from eval logs (have 'eval_loss')
    train_logs = [e for e in log_history if 'loss' in e and 'eval_loss' not in e]
    eval_logs = [e for e in log_history if 'eval_loss' in e]

    train_df = pd.DataFrame(train_logs) if train_logs else None
    eval_df = pd.DataFrame(eval_logs) if eval_logs else None

    return train_df, eval_df, state


def ema_smooth(values, alpha=0.2):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * v + (1 - alpha) * last
        smoothed.append(last)
    return smoothed


# =============================================================================
# B1. Training Loss Curves
# =============================================================================

def plot_b1_training_loss(coarse_train, fine_train):
    """Training loss over steps for both classifiers."""
    has_coarse = coarse_train is not None and 'loss' in coarse_train.columns
    has_fine = fine_train is not None and 'loss' in fine_train.columns

    if not has_coarse and not has_fine:
        print("  Skipping B1: no training loss data")
        return

    ncols = has_coarse + has_fine
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_coarse:
        ax = axes[idx]
        steps = coarse_train['step'].values
        loss = coarse_train['loss'].values
        ax.scatter(steps, loss, alpha=0.3, s=10, color='#90CAF9', label='Raw')
        ax.plot(steps, ema_smooth(loss), color='#1565C0', linewidth=2, label='EMA smoothed')
        if 'epoch' in coarse_train.columns:
            epoch_boundaries = coarse_train.groupby(coarse_train['epoch'].astype(int))['step'].first()
            for ep, st in epoch_boundaries.items():
                if ep > 0:
                    ax.axvline(x=st, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Coarse Classifier", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        idx += 1

    if has_fine:
        ax = axes[idx]
        steps = fine_train['step'].values
        loss = fine_train['loss'].values
        ax.scatter(steps, loss, alpha=0.3, s=10, color='#EF9A9A', label='Raw')
        ax.plot(steps, ema_smooth(loss), color='#C62828', linewidth=2, label='EMA smoothed')
        if 'epoch' in fine_train.columns:
            epoch_boundaries = fine_train.groupby(fine_train['epoch'].astype(int))['step'].first()
            for ep, st in epoch_boundaries.items():
                if ep > 0:
                    ax.axvline(x=st, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        # Clip y-axis to focus on actual training dynamics (exclude initial spike)
        p99 = np.percentile(loss, 99)
        ax.set_ylim(0, min(p99 * 1.3, loss.max()))
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Fine Classifier (Soft Conditioning)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Training Loss Curves", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "b1_training_loss.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# B2. Evaluation Metrics Over Training
# =============================================================================

def plot_b2_eval_metrics(coarse_eval, fine_eval, coarse_state, fine_state):
    """Evaluation metrics vs epoch for both classifiers."""
    has_coarse = coarse_eval is not None and 'eval_f1' in coarse_eval.columns
    has_fine = fine_eval is not None and 'eval_sample_f1' in fine_eval.columns

    if not has_coarse and not has_fine:
        print("  Skipping B2: no eval data")
        return

    ncols = has_coarse + has_fine
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_coarse:
        ax = axes[idx]
        epochs = coarse_eval['epoch'].values

        if 'eval_accuracy' in coarse_eval.columns:
            ax.plot(epochs, coarse_eval['eval_accuracy'], 'o-', color='#2196F3',
                    label='Accuracy', linewidth=2, markersize=5)
        if 'eval_f1' in coarse_eval.columns:
            f1_vals = coarse_eval['eval_f1'].values
            ax.plot(epochs, f1_vals, 's-', color='#F44336',
                    label='F1', linewidth=2, markersize=5)
            # Mark best
            if coarse_state:
                best_step = coarse_state.get('best_global_step')
                if best_step and 'step' in coarse_eval.columns:
                    best_row = coarse_eval[coarse_eval['step'] == best_step]
                    if not best_row.empty:
                        ax.plot(best_row['epoch'].values[0], best_row['eval_f1'].values[0],
                                '*', color='gold', markersize=18, markeredgecolor='black',
                                markeredgewidth=1, zorder=5, label='Best checkpoint')

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Coarse Classifier", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        idx += 1

    if has_fine:
        ax = axes[idx]
        epochs = fine_eval['epoch'].values

        metrics = [
            ('eval_sample_f1', 'Sample F1', '#F44336'),
            ('eval_micro_f1', 'Micro F1', '#2196F3'),
            ('eval_macro_f1', 'Macro F1', '#4CAF50'),
        ]
        for col, label, color in metrics:
            if col in fine_eval.columns:
                ax.plot(epochs, fine_eval[col], 'o-', color=color,
                        label=label, linewidth=2, markersize=5)

        # Mark best
        if fine_state and 'eval_sample_f1' in fine_eval.columns:
            best_step = fine_state.get('best_global_step')
            if best_step and 'step' in fine_eval.columns:
                best_row = fine_eval[fine_eval['step'] == best_step]
                if not best_row.empty:
                    ax.plot(best_row['epoch'].values[0], best_row['eval_sample_f1'].values[0],
                            '*', color='gold', markersize=18, markeredgecolor='black',
                            markeredgewidth=1, zorder=5, label='Best checkpoint')

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Fine Classifier (Soft Conditioning)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle("Evaluation Metrics Over Training", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "b2_eval_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# B3. Learning Rate Schedule
# =============================================================================

def plot_b3_learning_rate(coarse_train, fine_train):
    """Learning rate vs step."""
    has_coarse = coarse_train is not None and 'learning_rate' in coarse_train.columns
    has_fine = fine_train is not None and 'learning_rate' in fine_train.columns

    if not has_coarse and not has_fine:
        print("  Skipping B3: no learning rate data")
        return

    ncols = has_coarse + has_fine
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_coarse:
        ax = axes[idx]
        ax.plot(coarse_train['step'], coarse_train['learning_rate'],
                color='#1565C0', linewidth=2)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Learning Rate", fontsize=11)
        ax.set_title("Coarse Classifier", fontsize=12, fontweight='bold')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
        ax.grid(alpha=0.3)
        idx += 1

    if has_fine:
        ax = axes[idx]
        ax.plot(fine_train['step'], fine_train['learning_rate'],
                color='#C62828', linewidth=2)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Learning Rate", fontsize=11)
        ax.set_title("Fine Classifier (Soft Conditioning)", fontsize=12, fontweight='bold')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
        ax.grid(alpha=0.3)

    plt.suptitle("Learning Rate Schedule", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "b3_learning_rate.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# B4. Gradient Norm Over Training
# =============================================================================

def plot_b4_gradient_norm(coarse_train, fine_train):
    """Gradient norm vs step — shows training stability."""
    has_coarse = coarse_train is not None and 'grad_norm' in coarse_train.columns
    has_fine = fine_train is not None and 'grad_norm' in fine_train.columns

    if not has_coarse and not has_fine:
        print("  Skipping B4: no gradient norm data")
        return

    ncols = has_coarse + has_fine
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_coarse:
        ax = axes[idx]
        steps = coarse_train['step'].values
        gnorm = coarse_train['grad_norm'].values
        ax.scatter(steps, gnorm, alpha=0.3, s=10, color='#90CAF9')
        ax.plot(steps, ema_smooth(gnorm, alpha=0.15), color='#1565C0', linewidth=2, label='EMA smoothed')
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Gradient Norm", fontsize=11)
        ax.set_title("Coarse Classifier", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        idx += 1

    if has_fine:
        ax = axes[idx]
        steps = fine_train['step'].values
        gnorm = fine_train['grad_norm'].values
        ax.scatter(steps, gnorm, alpha=0.3, s=10, color='#EF9A9A')
        ax.plot(steps, ema_smooth(gnorm, alpha=0.15), color='#C62828', linewidth=2, label='EMA smoothed')
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Gradient Norm", fontsize=11)
        ax.set_title("Fine Classifier (Soft Conditioning)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Gradient Norm Over Training", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "b4_gradient_norm.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# B5. Fine Classifier: Precision/Recall/F1 + Prediction Count
# =============================================================================

def plot_b5_fine_precision_recall(fine_eval):
    """Precision, recall, sample F1, and avg prediction count vs epoch."""
    if fine_eval is None:
        print("  Skipping B5: no fine eval data")
        return

    has_pr = 'eval_precision' in fine_eval.columns and 'eval_recall' in fine_eval.columns
    has_pred_count = 'eval_avg_pred_count' in fine_eval.columns

    if not has_pr:
        print("  Skipping B5: no precision/recall data")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = fine_eval['epoch'].values

    # Primary y-axis: precision, recall, F1
    ax1.plot(epochs, fine_eval['eval_precision'], 'o-', color='#2196F3',
             label='Precision', linewidth=2, markersize=5)
    ax1.plot(epochs, fine_eval['eval_recall'], 's-', color='#4CAF50',
             label='Recall', linewidth=2, markersize=5)
    if 'eval_sample_f1' in fine_eval.columns:
        ax1.plot(epochs, fine_eval['eval_sample_f1'], '^-', color='#F44336',
                 label='Sample F1', linewidth=2, markersize=5)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12, color='black')
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Secondary y-axis: avg prediction count
    if has_pred_count:
        ax2 = ax1.twinx()
        ax2.plot(epochs, fine_eval['eval_avg_pred_count'], 'D--', color='#FF9800',
                 label='Avg Pred Count', linewidth=2, markersize=5, alpha=0.8)
        ax2.set_ylabel("Avg Predictions per Sample", fontsize=12, color='#FF9800')
        ax2.tick_params(axis='y', labelcolor='#FF9800')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='lower right')
    else:
        ax1.legend(fontsize=10)

    ax1.set_title("Fine Classifier: Precision, Recall, F1 & Prediction Calibration",
                   fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "b5_fine_precision_recall_cardinality.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# B6. Coarse Classification Confusion Matrix
# =============================================================================

def load_predictions():
    """Load final_predictions.csv and parse label columns."""
    if not os.path.isfile(PREDICTIONS_FILE):
        return None
    df = pd.read_csv(PREDICTIONS_FILE)
    if 'labels' not in df.columns or 'predicted_labels' not in df.columns:
        return None

    def safe_parse(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val

    df['labels_parsed'] = df['labels'].apply(safe_parse)
    df['predicted_labels_parsed'] = df['predicted_labels'].apply(safe_parse)
    df['gt_coarse'] = df['labels_parsed'].apply(lambda x: x[0])
    df['gt_fine'] = df['labels_parsed'].apply(lambda x: set(x[1:]))
    df['pred_coarse'] = df['predicted_labels_parsed'].apply(lambda x: x[0])
    df['pred_fine'] = df['predicted_labels_parsed'].apply(lambda x: set(x[1:]))
    return df


def plot_b6_coarse_confusion(pred_df):
    """3x3 confusion matrix for coarse classification."""
    if pred_df is None:
        print("  Skipping B6: no predictions file")
        return

    n = len(COARSE_LABELS)
    label_to_idx = {l: i for i, l in enumerate(COARSE_LABELS)}
    cm = np.zeros((n, n), dtype=int)

    for _, row in pred_df.iterrows():
        gt = row['gt_coarse']
        pred = row['pred_coarse']
        if gt in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[gt], label_to_idx[pred]] += 1

    # Normalize per row (recall-based)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Plot with counts
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

    # Annotate with count + percentage
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = cm_pct[i, j]
            color = 'white' if pct > 60 else 'black'
            ax.text(j, i, f'{count}\n({pct:.1f}%)', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(COARSE_LABELS, fontsize=11)
    ax.set_yticklabels(COARSE_LABELS, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)

    # Per-class accuracy on the side
    for i in range(n):
        acc = cm[i, i] / row_sums[i, 0] * 100 if row_sums[i, 0] > 0 else 0
        ax.text(n + 0.15, i, f'{acc:.1f}%', ha='left', va='center',
                fontsize=11, fontweight='bold', color='#1565C0')
    ax.text(n + 0.15, -0.5, 'Recall', ha='left', va='center',
            fontsize=10, fontstyle='italic', color='gray')

    total_acc = np.trace(cm) / cm.sum() * 100
    ax.set_title(f"Coarse Classification Confusion Matrix (Accuracy: {total_acc:.1f}%)",
                 fontsize=14, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Row %', fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "b6_coarse_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# B7. Per Fine Label F1 Bar Chart
# =============================================================================

def plot_b7_per_label_f1(pred_df):
    """Horizontal bar chart of F1 per fine label, grouped by coarse parent."""
    if pred_df is None:
        print("  Skipping B7: no predictions file")
        return

    # Compute TP, FP, FN per fine label
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for _, row in pred_df.iterrows():
        gt = row['gt_fine']
        pred = row['pred_fine']
        for label in gt & pred:
            tp[label] += 1
        for label in pred - gt:
            fp[label] += 1
        for label in gt - pred:
            fn[label] += 1

    # Compute per-label P, R, F1
    results = []
    for label in FINE_LABELS_ORDERED:
        t = tp[label]
        f_p = fp[label]
        f_n = fn[label]
        precision = t / (t + f_p) if (t + f_p) > 0 else 0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = t + f_n  # ground truth count
        results.append({
            'label': label, 'coarse': COARSE_FOR_FINE[label],
            'precision': precision, 'recall': recall, 'f1': f1, 'support': support
        })

    res_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors_map = {'Protagonist': '#2196F3', 'Antagonist': '#F44336', 'Innocent': '#4CAF50'}
    y_pos = np.arange(len(FINE_LABELS_ORDERED))
    bar_colors = [colors_map[COARSE_FOR_FINE[l]] for l in FINE_LABELS_ORDERED]

    bars = ax.barh(y_pos, res_df['f1'], color=bar_colors, edgecolor='white', alpha=0.85)

    # Annotate with F1 value + support count
    for i, row in res_df.iterrows():
        f1_val = row['f1']
        support = row['support']
        x_text = f1_val + 0.01 if f1_val < 0.85 else f1_val - 0.01
        ha = 'left' if f1_val < 0.85 else 'right'
        color = 'black' if f1_val < 0.85 else 'white'
        ax.text(x_text, i, f'{f1_val:.2f} (n={support})', ha=ha, va='center',
                fontsize=9, fontweight='bold', color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(FINE_LABELS_ORDERED, fontsize=10)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add coarse group separators
    protagonist_end = len(FINE_LABELS_BY_COARSE["Protagonist"]) - 0.5
    antagonist_end = protagonist_end + len(FINE_LABELS_BY_COARSE["Antagonist"])
    for pos in [protagonist_end, antagonist_end]:
        ax.axhline(y=pos, color='black', linewidth=1.5, linestyle='-')

    # Coarse group labels on the right
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_map[c], label=c) for c in COARSE_LABELS]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

    ax.set_title("Per Fine-Grained Label F1 Score (Validation Set)",
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "b7_per_label_f1.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" Model Training Analysis — SemEval-2025 Task 10 ST1")
    print("=" * 60)

    # Load coarse classifier history
    print(f"\n[1/9] Loading coarse classifier history from {COARSE_CHECKPOINT_DIR}...")
    coarse_train, coarse_eval, coarse_state = load_training_history(COARSE_CHECKPOINT_DIR)
    if coarse_train is not None:
        print(f"  Found {len(coarse_train)} training logs, "
              f"{len(coarse_eval) if coarse_eval is not None else 0} eval logs")
    else:
        print("  No coarse classifier checkpoints found")

    # Load fine classifier history
    print(f"\n[2/9] Loading fine classifier history from {FINE_CHECKPOINT_DIR}...")
    fine_train, fine_eval, fine_state = load_training_history(FINE_CHECKPOINT_DIR)
    if fine_train is not None:
        print(f"  Found {len(fine_train)} training logs, "
              f"{len(fine_eval) if fine_eval is not None else 0} eval logs")
    else:
        print("  No fine classifier checkpoints found")

    if coarse_train is None and fine_train is None:
        print("\nNo checkpoint data found. Skipping B1-B5.")
        print(f"  Expected coarse: {COARSE_CHECKPOINT_DIR}")
        print(f"  Expected fine:   {FINE_CHECKPOINT_DIR}")
    else:
        print(f"\n[3/9] B1: Training loss curves...")
        plot_b1_training_loss(coarse_train, fine_train)

        print(f"\n[4/9] B2: Evaluation metrics...")
        plot_b2_eval_metrics(coarse_eval, fine_eval, coarse_state, fine_state)

        print(f"\n[5/9] B3: Learning rate schedule...")
        plot_b3_learning_rate(coarse_train, fine_train)

        print(f"\n[6/9] B4: Gradient norm...")
        plot_b4_gradient_norm(coarse_train, fine_train)

        print(f"\n[7/9] B5: Fine precision/recall/cardinality...")
        plot_b5_fine_precision_recall(fine_eval)

    # Prediction-based diagrams (from final_predictions.csv)
    print(f"\n[8/9] Loading predictions from {PREDICTIONS_FILE}...")
    pred_df = load_predictions()
    if pred_df is not None:
        print(f"  Found {len(pred_df)} predictions with ground truth")
    else:
        print("  No predictions file found — skipping B6, B7")

    print(f"\n[8/9] B6: Coarse confusion matrix...")
    plot_b6_coarse_confusion(pred_df)

    print(f"\n[9/9] B7: Per fine-label F1...")
    plot_b7_per_label_f1(pred_df)

    print("\n" + "=" * 60)
    print(f" All diagrams saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
