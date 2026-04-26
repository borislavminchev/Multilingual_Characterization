"""
Advanced Analysis Diagrams for the Diploma Thesis

Generates:
  D1. System architecture diagram (programmatic)
  D5. Per-language F1 comparison: baseline vs advanced coarse classifier

Usage:
    cd src2 && python advanced_analysis.py
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COARSE_PREDICTIONS_TEST = os.path.join(PROJECT_ROOT, "predictions", "coarse_predictions_test.csv")
BASELINE_PREDICTIONS = os.path.join(PROJECT_ROOT, "predictions", "baseline", "baseline_predictions.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "diagrams", "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COARSE_LABELS = ["Protagonist", "Antagonist", "Innocent"]
LANG_NAMES = {"BG": "Bulgarian", "EN": "English", "HI": "Hindi", "PT": "Portuguese", "RU": "Russian"}
LANG_ORDER = ["BG", "EN", "HI", "PT", "RU"]


def extract_language(doc_id: str) -> str:
    name = doc_id.replace('.txt', '')
    if name.startswith('A9_'):
        name = name[3:]
    lang = name.split('_')[0].split('-')[0]
    return lang if lang in ('BG', 'EN', 'HI', 'PT', 'RU') else "UNK"


def get_true_coarse(labels_str: str) -> str:
    labels = ast.literal_eval(labels_str)
    return labels[0]


def d5_per_language_f1():
    """D5: Per-language weighted F1 for the advanced coarse classifier."""
    df = pd.read_csv(COARSE_PREDICTIONS_TEST)
    df['language'] = df['doc_id'].apply(extract_language)
    df['true_coarse'] = df['labels'].apply(get_true_coarse)

    has_baseline = os.path.exists(BASELINE_PREDICTIONS)
    if has_baseline:
        df_base = pd.read_csv(BASELINE_PREDICTIONS)
        if 'language' not in df_base.columns:
            df_base['language'] = df_base['doc_id'].apply(extract_language)
        if 'label' in df_base.columns:
            df_base['true_coarse'] = df_base['label']
            df_base['predicted_coarse'] = df_base['predicted_label']
        elif 'labels' in df_base.columns:
            df_base['true_coarse'] = df_base['labels'].apply(get_true_coarse)
            if 'predicted_label' in df_base.columns:
                df_base['predicted_coarse'] = df_base['predicted_label']
        else:
            has_baseline = False

    languages = [lang for lang in LANG_ORDER if lang in df['language'].unique()]
    adv_f1s = []
    base_f1s = []
    counts = []

    for lang in languages:
        mask = df['language'] == lang
        sub = df[mask]
        y_true = sub['true_coarse']
        y_pred = sub['predicted_coarse']
        wf1 = f1_score(y_true, y_pred, labels=COARSE_LABELS, average='weighted', zero_division=0)
        adv_f1s.append(wf1 * 100)
        counts.append(len(sub))

        if has_baseline:
            mask_b = df_base['language'] == lang
            sub_b = df_base[mask_b]
            if len(sub_b) > 0 and 'predicted_coarse' in sub_b.columns:
                y_true_b = sub_b['true_coarse']
                y_pred_b = sub_b['predicted_coarse']
                wf1_b = f1_score(y_true_b, y_pred_b, labels=COARSE_LABELS,
                                 average='weighted', zero_division=0)
                base_f1s.append(wf1_b * 100)
            else:
                base_f1s.append(0)

    x = np.arange(len(languages))
    width = 0.35 if has_baseline else 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    if has_baseline and len(base_f1s) == len(languages):
        bars1 = ax.bar(x - width / 2, base_f1s, width, label='Baseline (Ch.4)',
                        color='#BDBDBD', edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + width / 2, adv_f1s, width, label='Advanced (E19)',
                        color='#1976D2', edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars1, base_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='#616161')
        for bar, val in zip(bars2, adv_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#0D47A1')
    else:
        bars = ax.bar(x, adv_f1s, width, color='#1976D2', edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, adv_f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    lang_labels = [f"{LANG_NAMES.get(l, l)}\n(n={c})" for l, c in zip(languages, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=10)
    ax.set_ylabel('Weighted F1 (%)', fontsize=12)
    ax.set_title('Coarse Classification: Per-Language Weighted F1', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    overall_f1 = f1_score(df['true_coarse'], df['predicted_coarse'],
                          labels=COARSE_LABELS, average='weighted', zero_division=0) * 100
    ax.axhline(y=overall_f1, color='#D32F2F', linestyle='--', alpha=0.7, linewidth=1.2)
    ax.text(len(languages) - 0.5, overall_f1 + 1.5, f'Overall: {overall_f1:.1f}%',
            fontsize=9, color='#D32F2F', ha='right')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "d5_advanced_per_language_f1.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def d1_system_architecture():
    """D1: System architecture diagram generated programmatically."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_props = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    box_coarse = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5)
    box_fine = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5)
    box_input = dict(boxstyle='round,pad=0.5', facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=1.5)
    box_output = dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828', linewidth=1.5)

    arrow_props = dict(arrowstyle='->', color='#37474F', lw=2,
                       connectionstyle='arc3,rad=0')

    # Input
    ax.text(7, 9.3, '... [ENTITY] Putin [/ENTITY] said that ...',
            ha='center', va='center', fontsize=11, family='monospace',
            bbox=box_input)
    ax.annotate('', xy=(7, 8.55), xytext=(7, 8.9),
                arrowprops=arrow_props)

    # XLM-RoBERTa
    ax.text(7, 8.2, 'XLM-RoBERTa-base\n(12 Transformer layers, 768 dim, 6 unfrozen)',
            ha='center', va='center', fontsize=10, bbox=box_props)
    ax.annotate('', xy=(7, 7.45), xytext=(7, 7.8),
                arrowprops=arrow_props)

    # ESP
    ax.text(7, 7.1, 'Entity Span Pooling\ne = mean(h_i ... h_j)',
            ha='center', va='center', fontsize=10, bbox=box_props)

    # Fork arrows
    ax.annotate('', xy=(4, 6.2), xytext=(6, 6.7),
                arrowprops=arrow_props)
    ax.annotate('', xy=(10, 6.2), xytext=(8, 6.7),
                arrowprops=arrow_props)

    # Stage 1: Coarse
    ax.text(4, 6.0, 'STAGE 1: COARSE', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(4, 5.3, 'Semantic Similarity Head\ncos(z, anchor) × T=15.0\n3 anchor embeddings',
            ha='center', va='center', fontsize=9, bbox=box_coarse)
    ax.text(4, 4.2, 'CB-CE Loss\n(β = 0.9999)',
            ha='center', va='center', fontsize=9, bbox=box_coarse)
    ax.text(4, 3.3, 'Output: 3 classes\nProt / Antag / Inn',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=box_output)

    ax.annotate('', xy=(4, 4.7), xytext=(4, 4.85),
                arrowprops=arrow_props)
    ax.annotate('', xy=(4, 3.7), xytext=(4, 3.85),
                arrowprops=arrow_props)

    # Arrow from coarse probs to fine
    ax.annotate('', xy=(7.5, 5.3), xytext=(5.5, 5.3),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2,
                                connectionstyle='arc3,rad=-0.2', linestyle='dashed'))
    ax.text(6.5, 5.7, 'p_coarse', ha='center', va='center',
            fontsize=9, color='#D32F2F', fontstyle='italic')

    # Stage 2: Fine
    ax.text(10, 6.0, 'STAGE 2: FINE', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#E65100')
    ax.text(10, 5.3, 'Entity Projection MLP\n768 → 768 → 22 logits\n+ Hierarchy Affinity (p · A)',
            ha='center', va='center', fontsize=9, bbox=box_fine)
    ax.text(10, 4.2, 'ASL-Opt (γ⁺=0, γ⁻=2, m=0.05)\n+ Entropy reg (λ=0.15)\n+ Cardinality reg (λ=0.3, k=1.5)',
            ha='center', va='center', fontsize=8.5, bbox=box_fine)
    ax.text(10, 3.15, 'Adaptive Threshold\nτ=0.25, gap=0.7\nmin=1, max=3',
            ha='center', va='center', fontsize=9, bbox=box_fine)
    ax.text(10, 2.2, 'Output: ≤3 labels\nfrom 22 fine roles',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=box_output)

    ax.annotate('', xy=(10, 4.7), xytext=(10, 4.85),
                arrowprops=arrow_props)
    ax.annotate('', xy=(10, 3.6), xytext=(10, 3.75),
                arrowprops=arrow_props)
    ax.annotate('', xy=(10, 2.6), xytext=(10, 2.75),
                arrowprops=arrow_props)

    # Title
    ax.text(7, 0.5, 'Hierarchical Entity Role Classification — System Architecture',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#263238')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "d1_system_architecture.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED ANALYSIS DIAGRAMS")
    print("=" * 60)

    print("\nGenerating D1: System Architecture...")
    d1_system_architecture()

    print("\nGenerating D5: Per-Language F1...")
    d5_per_language_f1()

    print("\nDone!")
