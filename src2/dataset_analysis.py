"""
Dataset Analysis Diagrams for SemEval-2025 Task 10 ST1 (Entity Framing)

Generates 6 complementary diagrams that go beyond the official paper's analysis:
  A1. Fine label co-occurrence heatmap
  A2. Coarse label distribution per language
  A3. Labels-per-entity cardinality distribution
  A4. Entity role consistency / diversity
  A5. Text length distribution per language
  A6. Domain breakdown per language

Usage:
    cd src2 && python dataset_analysis.py
"""

import os
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from data_utils import load_annotations, available_languages
from config import TRAIN_DATA_PARENT, VAL_DATA_PARENT

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "diagrams", "dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ordered label lists matching taxonomy hierarchy
COARSE_LABELS = ["Protagonist", "Antagonist", "Innocent"]

FINE_LABELS_BY_COARSE = {
    "Protagonist": ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"],
    "Antagonist": ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
                    "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver", "Bigot"],
    "Innocent": ["Forgotten", "Exploited", "Victim", "Scapegoat"],
}

# Flat ordered list grouped by coarse parent
FINE_LABELS_ORDERED = []
for coarse in COARSE_LABELS:
    FINE_LABELS_ORDERED.extend(FINE_LABELS_BY_COARSE[coarse])

LANG_NAMES = {"BG": "Bulgarian", "EN": "English", "HI": "Hindi", "PT": "Portuguese", "RU": "Russian"}


# =============================================================================
# DATA LOADING
# =============================================================================

def extract_language(doc_id):
    """Extract language code from doc_id."""
    doc_id_upper = doc_id.upper()
    for lang in ["EN", "BG", "HI", "PT", "RU"]:
        if lang in doc_id_upper:
            return lang
    return "Unknown"


def extract_domain(doc_id):
    """Extract domain from doc_id: CC (Climate Change) or URW (Ukraine-Russia War)."""
    doc_id_upper = doc_id.upper()
    if "_CC_" in doc_id_upper or doc_id_upper.startswith("A6_CC"):
        return "Climate Change"
    if "_UA_" in doc_id_upper or "_URW_" in doc_id_upper or "-URW-" in doc_id_upper or doc_id_upper.startswith("A7_URW"):
        return "Ukraine-Russia War"
    return "Unknown"


def load_all_data():
    """Load train + val data for all languages, combine into single DataFrame."""
    all_dfs = []

    for lang in available_languages:
        # Train
        train_root = os.path.join(TRAIN_DATA_PARENT, lang, "raw-documents")
        train_ann = os.path.join(TRAIN_DATA_PARENT, lang, "subtask-1-annotations.txt")
        if os.path.exists(train_ann):
            df = load_annotations(train_ann, train_root, labeled=True)
            if not df.empty:
                df['split'] = 'train'
                df['lang_folder'] = lang
                all_dfs.append(df)

        # Val
        val_root = os.path.join(VAL_DATA_PARENT, lang, "subtask-1-documents")
        val_ann = os.path.join(VAL_DATA_PARENT, lang, "subtask-1-annotations.txt")
        if os.path.exists(val_ann):
            df = load_annotations(val_ann, val_root, labeled=True)
            if not df.empty:
                df['split'] = 'val'
                df['lang_folder'] = lang
                all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Extract language and domain
    combined['language'] = combined['lang_folder']
    combined['domain'] = combined['doc_id'].apply(extract_domain)

    # Parse labels: extract coarse (first) and fine (rest)
    combined['coarse_label'] = combined['labels'].apply(lambda x: x[0])
    combined['fine_labels'] = combined['labels'].apply(lambda x: x[1:])
    combined['num_fine_labels'] = combined['fine_labels'].apply(len)

    print(f"Loaded {len(combined)} entity mentions across {combined['language'].nunique()} languages")
    print(f"Splits: {combined['split'].value_counts().to_dict()}")
    print(f"Languages: {combined['language'].value_counts().to_dict()}")
    print(f"Domains: {combined['domain'].value_counts().to_dict()}")

    return combined


# =============================================================================
# A1. Fine Label Co-occurrence Heatmap
# =============================================================================

def plot_a1_cooccurrence(df):
    """Three separate co-occurrence heatmaps, one PNG per coarse category."""
    n = len(FINE_LABELS_ORDERED)
    label_to_idx = {label: i for i, label in enumerate(FINE_LABELS_ORDERED)}
    cooccurrence = np.zeros((n, n), dtype=int)

    for fine_set in df['fine_labels']:
        for label in fine_set:
            if label in label_to_idx:
                idx = label_to_idx[label]
                cooccurrence[idx, idx] += 1

        for l1, l2 in itertools.combinations(fine_set, 2):
            if l1 in label_to_idx and l2 in label_to_idx:
                i, j = label_to_idx[l1], label_to_idx[l2]
                cooccurrence[i, j] += 1
                cooccurrence[j, i] += 1

    coarse_colors = {
        "Protagonist": "YlGn",
        "Antagonist": "YlOrRd",
        "Innocent": "PuBu",
    }

    offset = 0
    for coarse in COARSE_LABELS:
        labels = FINE_LABELS_BY_COARSE[coarse]
        size = len(labels)
        sub = cooccurrence[offset:offset + size, offset:offset + size]
        mask = sub == 0

        scale = max(6, size * 0.9)
        fig, ax = plt.subplots(figsize=(scale, scale))

        sns.heatmap(sub, xticklabels=labels, yticklabels=labels,
                    annot=True, fmt='d', cmap=coarse_colors[coarse], mask=mask,
                    linewidths=0.5, linecolor='white', ax=ax,
                    cbar=True, square=True,
                    annot_kws={"fontsize": 11})

        ax.set_title(f"{coarse} — Fine Label Co-occurrence",
                     fontsize=14, fontweight='bold', pad=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

        plt.tight_layout()
        fname = f"a1_{coarse.lower()}_cooccurrence.png"
        path = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")
        offset += size


# =============================================================================
# A2. Coarse Label Distribution per Language
# =============================================================================

def plot_a2_coarse_by_language(df):
    """Grouped bar chart: coarse label proportions per language."""
    lang_order = ["EN", "BG", "HI", "PT", "RU"]
    langs_present = [l for l in lang_order if l in df['language'].unique()]

    # Compute counts and percentages
    cross = pd.crosstab(df['language'], df['coarse_label'])
    cross = cross.reindex(index=langs_present, columns=COARSE_LABELS, fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    x = np.arange(len(langs_present))
    width = 0.25
    colors = {'Protagonist': '#2196F3', 'Antagonist': '#F44336', 'Innocent': '#4CAF50'}

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, coarse in enumerate(COARSE_LABELS):
        bars = ax.bar(x + i * width, cross_pct[coarse], width, label=coarse,
                      color=colors[coarse], edgecolor='white')
        # Add count annotations
        for j, bar in enumerate(bars):
            count = cross.iloc[j][coarse]
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Coarse Role Distribution by Language", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([LANG_NAMES.get(l, l) for l in langs_present], fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, max(cross_pct.max()) + 10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "a2_coarse_distribution_by_language.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# A3. Labels-per-Entity Cardinality Distribution
# =============================================================================

def plot_a3_cardinality(df):
    """Histogram of fine label counts per entity, overall and by coarse category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Overall
    cardinality = df['num_fine_labels']
    max_card = cardinality.max()
    bins = np.arange(0.5, max_card + 1.5, 1)

    axes[0].hist(cardinality, bins=bins, color='#5C6BC0', edgecolor='white', alpha=0.85)
    axes[0].axvline(cardinality.mean(), color='red', linestyle='--', linewidth=1.5,
                    label=f'Mean = {cardinality.mean():.2f}')
    axes[0].axvline(cardinality.median(), color='orange', linestyle='-.', linewidth=1.5,
                    label=f'Median = {cardinality.median():.1f}')
    # Mark TARGET_CARDINALITY
    axes[0].axvline(1.5, color='green', linestyle=':', linewidth=2,
                    label='Target = 1.5 (config)')
    axes[0].set_xlabel("Number of Fine Labels per Entity", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("(a) Overall Cardinality Distribution", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(range(1, max_card + 1))
    axes[0].grid(axis='y', alpha=0.3)

    # (b) By coarse category
    colors = {'Protagonist': '#2196F3', 'Antagonist': '#F44336', 'Innocent': '#4CAF50'}
    for coarse in COARSE_LABELS:
        subset = df[df['coarse_label'] == coarse]['num_fine_labels']
        axes[1].hist(subset, bins=bins, alpha=0.6, label=f'{coarse} (mean={subset.mean():.2f})',
                     color=colors[coarse], edgecolor='white')

    axes[1].set_xlabel("Number of Fine Labels per Entity", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("(b) Cardinality by Coarse Category", fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(range(1, max_card + 1))
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle("Fine-Grained Label Cardinality Distribution", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "a3_cardinality_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# A4. Entity Role Consistency / Diversity
# =============================================================================

def plot_a4_entity_diversity(df):
    """Analyze how many distinct role sets each entity mention receives."""
    # Group by normalized mention (lowercased to merge case variants)
    df_copy = df.copy()
    df_copy['mention_lower'] = df_copy['mention'].str.lower().str.strip()

    entity_roles = defaultdict(list)
    for _, row in df_copy.iterrows():
        fine_set = frozenset(row['fine_labels'])
        entity_roles[row['mention_lower']].append(fine_set)

    # Only entities appearing 2+ times
    multi_entities = {ent: roles for ent, roles in entity_roles.items() if len(roles) >= 2}

    if not multi_entities:
        print("  No multi-occurrence entities found, skipping A4")
        return

    diversity_counts = []
    entity_details = []
    for ent, roles in multi_entities.items():
        n_unique = len(set(roles))
        diversity_counts.append(n_unique)
        entity_details.append((ent, len(roles), n_unique))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) Histogram of diversity
    max_div = max(diversity_counts)
    bins = np.arange(0.5, max_div + 1.5, 1)
    axes[0].hist(diversity_counts, bins=bins, color='#7E57C2', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel("Number of Distinct Role Sets", fontsize=11)
    axes[0].set_ylabel("Number of Entities", fontsize=11)
    axes[0].set_title("(a) Role Diversity of Multi-Occurrence Entities", fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(1, max_div + 1))
    axes[0].grid(axis='y', alpha=0.3)

    total_multi = len(multi_entities)
    consistent = sum(1 for d in diversity_counts if d == 1)
    axes[0].text(0.95, 0.95,
                 f'Total: {total_multi} entities\nConsistent: {consistent} ({consistent/total_multi*100:.1f}%)',
                 transform=axes[0].transAxes, ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (b) Top-10 most role-diverse entities
    sorted_entities = sorted(entity_details, key=lambda x: (-x[2], -x[1]))[:15]
    names = [e[0][:25] for e in sorted_entities]
    occurrences = [e[1] for e in sorted_entities]
    unique_roles = [e[2] for e in sorted_entities]

    y_pos = np.arange(len(names))
    bars = axes[1].barh(y_pos, unique_roles, color='#FF7043', edgecolor='white', alpha=0.85)

    # Add occurrence count annotation
    for i, (occ, uniq) in enumerate(zip(occurrences, unique_roles)):
        axes[1].text(uniq + 0.1, i, f'({occ} occ.)', va='center', fontsize=9)

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].set_xlabel("Number of Distinct Role Sets", fontsize=11)
    axes[1].set_title("(b) Top-15 Most Role-Diverse Entities", fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle("Entity Role Consistency Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "a4_entity_role_diversity.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# A5. Text Length Distribution per Language
# =============================================================================

def plot_a5_text_length(df):
    """Violin/box plot of document word counts per language."""
    df_copy = df.copy()
    df_copy['word_count'] = df_copy['text'].apply(lambda x: len(x.split()))

    lang_order = [l for l in ["EN", "BG", "HI", "PT", "RU"] if l in df_copy['language'].unique()]
    df_copy['lang_name'] = df_copy['language'].map(LANG_NAMES)
    lang_name_order = [LANG_NAMES[l] for l in lang_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {'English': '#2196F3', 'Bulgarian': '#F44336', 'Hindi': '#FF9800',
               'Portuguese': '#4CAF50', 'Russian': '#9C27B0'}

    sns.violinplot(data=df_copy, x='lang_name', y='word_count', order=lang_name_order,
                   palette=palette, inner='box', ax=ax, cut=0)

    # Reference line for MAX_LENGTH (approximate: 512 tokens ~ 350-400 words)
    ax.axhline(y=400, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='~512 tokens (approx.)')

    # Add statistics
    for i, lang in enumerate(lang_order):
        subset = df_copy[df_copy['language'] == lang]['word_count']
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f'median={subset.median():.0f}\nmean={subset.mean():.0f}',
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Word Count", fontsize=12)
    ax.set_title("Document Word Count Distribution by Language", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "a5_text_length_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# A6. Domain Breakdown per Language
# =============================================================================

def plot_a6_domain_breakdown(df):
    """Stacked bar chart: domain proportions per language (only languages with domain info)."""
    # Only include entities whose domain could be determined from doc_id
    df_known = df[df['domain'] != 'Unknown'].copy()

    if df_known.empty:
        print("  Skipping A6: no domain info found in any doc_id")
        return

    lang_order = [l for l in ["EN", "BG", "HI", "PT", "RU"] if l in df_known['language'].unique()]
    domains = ["Ukraine-Russia War", "Climate Change"]

    cross = pd.crosstab(df_known['language'], df_known['domain'])
    cross = cross.reindex(index=lang_order, columns=domains, fill_value=0)
    cross = cross.loc[:, cross.sum() > 0]
    present_domains = cross.columns.tolist()

    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Ukraine-Russia War": '#F44336', "Climate Change": '#4CAF50'}
    bottom = np.zeros(len(lang_order))

    for domain in present_domains:
        values = cross_pct[domain].values
        counts = cross[domain].values
        ax.bar([LANG_NAMES.get(l, l) for l in lang_order], values, bottom=bottom,
               label=domain, color=colors.get(domain, '#9E9E9E'), edgecolor='white')

        for i, (val, count) in enumerate(zip(values, counts)):
            if count > 0:
                ax.text(i, bottom[i] + val / 2, f'{count}\n({val:.0f}%)',
                        ha='center', va='center', fontsize=9, fontweight='bold')

        bottom += values

    # Note which languages had no domain info
    all_langs = set(df['language'].unique())
    known_langs = set(df_known['language'].unique())
    missing_langs = all_langs - known_langs
    if missing_langs:
        missing_names = [LANG_NAMES.get(l, l) for l in sorted(missing_langs)]
        ax.text(0.5, -0.12,
                f"Note: {', '.join(missing_names)} excluded — domain not encoded in doc_id",
                ha='center', va='top', fontsize=9, fontstyle='italic', color='gray',
                transform=ax.transAxes)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Domain Distribution by Language (where annotated)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "a6_domain_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" Dataset Analysis — SemEval-2025 Task 10 ST1")
    print("=" * 60)

    print("\n[1/7] Loading data...")
    df = load_all_data()

    print(f"\n[2/7] A1: Fine label co-occurrence heatmap...")
    plot_a1_cooccurrence(df)

    print(f"\n[3/7] A2: Coarse distribution by language...")
    plot_a2_coarse_by_language(df)

    print(f"\n[4/7] A3: Cardinality distribution...")
    plot_a3_cardinality(df)

    print(f"\n[5/7] A4: Entity role diversity...")
    plot_a4_entity_diversity(df)

    print(f"\n[6/7] A5: Text length distribution...")
    plot_a5_text_length(df)

    print(f"\n[7/7] A6: Domain breakdown...")
    plot_a6_domain_breakdown(df)

    print("\n" + "=" * 60)
    print(f" All diagrams saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
