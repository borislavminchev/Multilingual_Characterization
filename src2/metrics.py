"""
Evaluation Metrics for Hierarchical Entity Role Classification

This module provides:
1. Coarse role evaluation (single-label classification)
2. Fine role evaluation (multi-label classification)
3. Hierarchical evaluation (combined coarse + fine)
4. Per-class performance breakdown

Usage:
    from metrics import evaluate_hierarchical_predictions
    results = evaluate_hierarchical_predictions(predictions_df)
"""

import ast
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from datasets import coarse_label2id, coarse_id2label, fine_label2id, fine_id2label


def parse_labels(labels):
    """Parse labels from string representation if needed."""
    if isinstance(labels, str):
        return ast.literal_eval(labels)
    return labels


# =============================================================================
# COARSE ROLE METRICS
# =============================================================================

def evaluate_coarse_predictions(y_true, y_pred):
    """
    Evaluate coarse role predictions (single-label classification).
    
    Args:
        y_true: list of ground truth coarse labels (names or IDs)
        y_pred: list of predicted coarse labels (names or IDs)
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert to IDs if names
    if isinstance(y_true[0], str):
        y_true = [coarse_label2id[label] for label in y_true]
    if isinstance(y_pred[0], str):
        y_pred = [coarse_label2id[label] for label in y_pred]
    
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    results['per_class'] = {}
    for class_id in range(len(coarse_id2label)):
        class_name = coarse_id2label[class_id]
        results['per_class'][class_name] = {
            'f1': per_class_f1[class_id],
            'precision': per_class_precision[class_id],
            'recall': per_class_recall[class_id],
        }
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return results


# =============================================================================
# FINE ROLE METRICS (MULTI-LABEL)
# =============================================================================

def compute_multilabel_f1_sample(y_true_set, y_pred_set):
    """
    Compute F1 score for a single sample (set-based).
    
    Args:
        y_true_set: set of ground truth labels
        y_pred_set: set of predicted labels
    
    Returns:
        tuple: (f1, precision, recall)
    """
    if len(y_true_set) == 0 and len(y_pred_set) == 0:
        return 1.0, 1.0, 1.0
    
    if len(y_pred_set) == 0:
        return 0.0, 0.0, 0.0
    
    if len(y_true_set) == 0:
        return 0.0, 0.0, 0.0
    
    tp = len(y_true_set & y_pred_set)
    precision = tp / len(y_pred_set)
    recall = tp / len(y_true_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1, precision, recall


def evaluate_fine_predictions(y_true_list, y_pred_list, respect_hierarchy=True, 
                               coarse_true=None, coarse_pred=None):
    """
    Evaluate fine role predictions (multi-label classification).
    
    Args:
        y_true_list: list of sets/lists of ground truth fine labels
        y_pred_list: list of sets/lists of predicted fine labels
        respect_hierarchy: if True, only evaluate fine labels valid for coarse category
        coarse_true: ground truth coarse labels (required if respect_hierarchy=True)
        coarse_pred: predicted coarse labels (for error analysis)
    
    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert to sets
    y_true_sets = [set(labels) if not isinstance(labels, set) else labels for labels in y_true_list]
    y_pred_sets = [set(labels) if not isinstance(labels, set) else labels for labels in y_pred_list]
    
    # Sample-level metrics
    sample_f1_scores = []
    sample_precision_scores = []
    sample_recall_scores = []
    
    for i in range(len(y_true_sets)):
        f1, prec, rec = compute_multilabel_f1_sample(y_true_sets[i], y_pred_sets[i])
        sample_f1_scores.append(f1)
        sample_precision_scores.append(prec)
        sample_recall_scores.append(rec)
    
    results = {
        'sample_f1_mean': np.mean(sample_f1_scores),
        'sample_f1_std': np.std(sample_f1_scores),
        'sample_precision_mean': np.mean(sample_precision_scores),
        'sample_recall_mean': np.mean(sample_recall_scores),
    }
    
    # Convert to binary matrix for sklearn metrics
    num_fine_labels = len(fine_label2id)
    y_true_binary = np.zeros((len(y_true_sets), num_fine_labels))
    y_pred_binary = np.zeros((len(y_pred_sets), num_fine_labels))
    
    for i, (true_set, pred_set) in enumerate(zip(y_true_sets, y_pred_sets)):
        for label in true_set:
            if label in fine_label2id:
                y_true_binary[i, fine_label2id[label]] = 1
        for label in pred_set:
            if label in fine_label2id:
                y_pred_binary[i, fine_label2id[label]] = 1
    
    # Micro/Macro F1
    results['micro_f1'] = f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), 
                                    average='binary', zero_division=0)
    
    # Per-class metrics
    results['per_class'] = {}
    for fine_name, fine_id in fine_label2id.items():
        class_true = y_true_binary[:, fine_id]
        class_pred = y_pred_binary[:, fine_id]
        
        if class_true.sum() > 0 or class_pred.sum() > 0:
            results['per_class'][fine_name] = {
                'f1': f1_score(class_true, class_pred, zero_division=0),
                'precision': precision_score(class_true, class_pred, zero_division=0),
                'recall': recall_score(class_true, class_pred, zero_division=0),
                'support': int(class_true.sum()),
                'predicted': int(class_pred.sum()),
            }
    
    return results


# =============================================================================
# HIERARCHICAL METRICS
# =============================================================================

def evaluate_hierarchical_predictions(df, pred_col='predicted_labels', gt_col='labels'):
    """
    Comprehensive evaluation of hierarchical predictions.
    
    Args:
        df: DataFrame with predictions and ground truth
        pred_col: column name for predictions (list format: [coarse, fine1, fine2, ...])
        gt_col: column name for ground truth
    
    Returns:
        dict: Comprehensive evaluation results
    """
    results = {}
    
    # Parse labels
    gt_labels = [parse_labels(l) for l in df[gt_col].tolist()]
    pred_labels = [parse_labels(l) for l in df[pred_col].tolist()]
    
    # Extract coarse and fine
    gt_coarse = [labels[0] for labels in gt_labels]
    pred_coarse = [labels[0] for labels in pred_labels]
    gt_fine = [set(labels[1:]) for labels in gt_labels]
    pred_fine = [set(labels[1:]) for labels in pred_labels]
    
    # Coarse evaluation
    results['coarse'] = evaluate_coarse_predictions(gt_coarse, pred_coarse)
    
    # Fine evaluation
    results['fine'] = evaluate_fine_predictions(gt_fine, pred_fine, 
                                                 coarse_true=gt_coarse, 
                                                 coarse_pred=pred_coarse)
    
    # Exact match (fine-grained labels match, as per official EMR definition)
    exact_matches = sum(
        1 for i in range(len(gt_fine))
        if gt_fine[i] == pred_fine[i]
    )
    results['exact_match_accuracy'] = exact_matches / len(gt_coarse)
    
    # Partial match (coarse correct)
    coarse_correct = sum(1 for gt, pred in zip(gt_coarse, pred_coarse) if gt == pred)
    results['coarse_accuracy'] = coarse_correct / len(gt_coarse)
    
    # Conditional fine accuracy (fine F1 when coarse is correct)
    correct_coarse_indices = [i for i in range(len(gt_coarse)) if gt_coarse[i] == pred_coarse[i]]
    if correct_coarse_indices:
        conditional_f1_scores = []
        for i in correct_coarse_indices:
            f1, _, _ = compute_multilabel_f1_sample(gt_fine[i], pred_fine[i])
            conditional_f1_scores.append(f1)
        results['conditional_fine_f1'] = np.mean(conditional_f1_scores)
    else:
        results['conditional_fine_f1'] = 0.0
    
    # Error analysis
    results['error_analysis'] = analyze_errors(gt_coarse, pred_coarse, gt_fine, pred_fine)
    
    return results


def analyze_errors(gt_coarse, pred_coarse, gt_fine, pred_fine):
    """
    Analyze common error patterns.
    
    Returns:
        dict: Error analysis results
    """
    analysis = {
        'coarse_errors': defaultdict(int),
        'fine_missing': defaultdict(int),  # Labels that should have been predicted
        'fine_extra': defaultdict(int),    # Labels incorrectly predicted
        'error_categories': {
            'coarse_wrong': 0,
            'fine_missing_only': 0,
            'fine_extra_only': 0,
            'fine_both': 0,
            'perfect': 0,
        }
    }
    
    for i in range(len(gt_coarse)):
        coarse_correct = gt_coarse[i] == pred_coarse[i]
        fine_missing = gt_fine[i] - pred_fine[i]
        fine_extra = pred_fine[i] - gt_fine[i]
        
        if not coarse_correct:
            analysis['error_categories']['coarse_wrong'] += 1
            analysis['coarse_errors'][(gt_coarse[i], pred_coarse[i])] += 1
        elif fine_missing and fine_extra:
            analysis['error_categories']['fine_both'] += 1
        elif fine_missing:
            analysis['error_categories']['fine_missing_only'] += 1
        elif fine_extra:
            analysis['error_categories']['fine_extra_only'] += 1
        else:
            analysis['error_categories']['perfect'] += 1
        
        for label in fine_missing:
            analysis['fine_missing'][label] += 1
        for label in fine_extra:
            analysis['fine_extra'][label] += 1
    
    # Convert defaultdicts to regular dicts
    analysis['coarse_errors'] = dict(analysis['coarse_errors'])
    analysis['fine_missing'] = dict(analysis['fine_missing'])
    analysis['fine_extra'] = dict(analysis['fine_extra'])
    
    return analysis


# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def print_evaluation_report(results, title="Evaluation Report"):
    """
    Print a formatted evaluation report.
    
    Args:
        results: Dictionary from evaluate_hierarchical_predictions()
        title: Report title
    """
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS")
    print("-"*40)
    print(f"  Exact Match Accuracy:    {results['exact_match_accuracy']:.4f}")
    print(f"  Coarse Accuracy:         {results['coarse_accuracy']:.4f}")
    print(f"  Conditional Fine F1:     {results['conditional_fine_f1']:.4f}")
    
    # Coarse metrics
    print("\n🏷️ COARSE ROLE METRICS")
    print("-"*40)
    coarse = results['coarse']
    print(f"  Accuracy:     {coarse['accuracy']:.4f}")
    print(f"  Macro F1:     {coarse['macro_f1']:.4f}")
    print(f"  Weighted F1:  {coarse['weighted_f1']:.4f}")
    
    print("\n  Per-class breakdown:")
    for class_name, metrics in coarse['per_class'].items():
        print(f"    {class_name:15} - F1: {metrics['f1']:.4f}, "
              f"P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}")
    
    # Fine metrics
    print("\n🔍 FINE ROLE METRICS")
    print("-"*40)
    fine = results['fine']
    print(f"  Sample F1 (mean ± std): {fine['sample_f1_mean']:.4f} ± {fine['sample_f1_std']:.4f}")
    print(f"  Micro F1:               {fine['micro_f1']:.4f}")
    print(f"  Sample Precision:       {fine['sample_precision_mean']:.4f}")
    print(f"  Sample Recall:          {fine['sample_recall_mean']:.4f}")
    
    if fine['per_class']:
        print("\n  Top fine labels by support:")
        sorted_fine = sorted(fine['per_class'].items(), 
                            key=lambda x: x[1]['support'], reverse=True)[:10]
        for label, metrics in sorted_fine:
            print(f"    {label:20} - F1: {metrics['f1']:.4f}, "
                  f"Support: {metrics['support']}, Predicted: {metrics['predicted']}")
    
    # Error analysis
    print("\n⚠️ ERROR ANALYSIS")
    print("-"*40)
    errors = results['error_analysis']
    categories = errors['error_categories']
    total = sum(categories.values())
    
    print(f"  Perfect predictions:     {categories['perfect']:4d} ({categories['perfect']/total*100:.1f}%)")
    print(f"  Coarse errors:           {categories['coarse_wrong']:4d} ({categories['coarse_wrong']/total*100:.1f}%)")
    print(f"  Fine missing only:       {categories['fine_missing_only']:4d} ({categories['fine_missing_only']/total*100:.1f}%)")
    print(f"  Fine extra only:         {categories['fine_extra_only']:4d} ({categories['fine_extra_only']/total*100:.1f}%)")
    print(f"  Fine both miss+extra:    {categories['fine_both']:4d} ({categories['fine_both']/total*100:.1f}%)")
    
    if errors['coarse_errors']:
        print("\n  Most common coarse confusions:")
        sorted_confusions = sorted(errors['coarse_errors'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
        for (gt, pred), count in sorted_confusions:
            print(f"    {gt} → {pred}: {count}")
    
    print("\n" + "="*80)


def evaluate_from_csv(csv_path, pred_col='predicted_labels', gt_col='labels'):
    """
    Convenience function to evaluate predictions from a CSV file.
    
    Args:
        csv_path: Path to CSV file with predictions
        pred_col: Column name for predictions
        gt_col: Column name for ground truth
    
    Returns:
        dict: Evaluation results
    """
    df = pd.read_csv(csv_path)
    results = evaluate_hierarchical_predictions(df, pred_col=pred_col, gt_col=gt_col)
    print_evaluation_report(results, title=f"Evaluation: {csv_path}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate hierarchical predictions")
    parser.add_argument('--input', type=str, required=True, help='Input CSV with predictions')
    parser.add_argument('--pred_col', type=str, default='predicted_labels', help='Predictions column')
    parser.add_argument('--gt_col', type=str, default='labels', help='Ground truth column')
    args = parser.parse_args()
    
    evaluate_from_csv(args.input, pred_col=args.pred_col, gt_col=args.gt_col)