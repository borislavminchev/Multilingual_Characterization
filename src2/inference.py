"""
End-to-End Inference Pipeline

This script performs hierarchical inference:
1. Loads the coarse classifier and predicts coarse labels
2. Loads the fine classifier and predicts fine labels (using coarse predictions as masks)
3. Combines predictions into final output: [coarse, fine1, fine2, ...]

Prerequisites:
    - Run train_coarse.py to train coarse classifier
    - Run train_fine.py to train fine classifier

Usage:
    python inference.py
    python inference.py --input path/to/data.csv --output path/to/predictions.csv
"""

import os
import argparse
import ast
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from config import (
    COARSE_PREDICTIONS_TEST, FINE_CHECKPOINT_DIR, COARSE_CHECKPOINT_DIR,
    FINAL_PREDICTIONS_PATH, MODEL_NAME, MAX_LENGTH, FINE_THRESHOLD,
    FINE_GAP_RATIO, FINE_MIN_LABELS
)
from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from datasets import (
    FineRoleDataset, FineRoleInferenceDataset, collate_fn_fine_role, collate_fn_fine_inference,
    coarse_label2id, coarse_id2label, fine_label2id, fine_id2label
)
from hierarchical_model import FineRoleClassifier


def load_fine_classifier(checkpoint_dir, tokenizer, device):
    """Load trained fine classifier from checkpoint."""
    print(f"🧠 Loading fine classifier from: {checkpoint_dir}")
    
    # Load base model
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Initialize classifier structure
    classifier = FineRoleClassifier(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        threshold=FINE_THRESHOLD
    )
    
    # Load trained weights
    checkpoint_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        classifier.load_state_dict(state_dict)
        print("✅ Loaded model weights from pytorch_model.bin")
    else:
        # Try loading from safetensors or other formats
        safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            classifier.load_state_dict(state_dict)
            print("✅ Loaded model weights from model.safetensors")
        else:
            print("⚠️ Warning: No checkpoint found, using randomly initialized weights")
    
    classifier.to(device)
    classifier.eval()
    return classifier


def apply_hybrid_prediction(probs, coarse_label, taxonomy_manager, 
                            threshold=0.3, gap_ratio=0.5, min_labels=1):
    """
    Hybrid prediction strategy: threshold + relative fallback + guaranteed minimum.
    
    Args:
        probs: numpy array of shape (num_fine_labels,) - sigmoid probabilities
        coarse_label: int - coarse label ID for this sample
        taxonomy_manager: TaxonomyManager instance for getting valid fine labels
        threshold: float - primary threshold for predictions
        gap_ratio: float - ratio of max probability for relative threshold
        min_labels: int - minimum number of predictions to guarantee
    
    Returns:
        predictions: numpy array of shape (num_fine_labels,) - binary predictions
    """
    num_fine = len(probs)
    predictions = np.zeros(num_fine, dtype=int)
    
    # Get valid fine label indices for this coarse category
    valid_fine_indices = taxonomy_manager.get_fine_indices_for_coarse(coarse_label)
    
    if len(valid_fine_indices) == 0:
        return predictions  # No valid fine labels for this coarse
    
    # Create mask for valid labels
    valid_mask = np.zeros(num_fine, dtype=bool)
    valid_mask[valid_fine_indices] = True
    
    # Get valid probabilities
    valid_probs = probs.copy()
    valid_probs[~valid_mask] = -np.inf
    
    # Step 1: Apply primary threshold
    threshold_preds = (probs >= threshold) & valid_mask
    
    if threshold_preds.sum() > 0:
        predictions = threshold_preds.astype(int)
        return predictions
    
    # Step 2: Fallback to relative threshold
    max_valid_prob = valid_probs[valid_mask].max() if valid_mask.sum() > 0 else 0
    
    if max_valid_prob > 0:
        relative_threshold = gap_ratio * max_valid_prob
        relative_preds = (probs >= relative_threshold) & valid_mask
        
        if relative_preds.sum() > 0:
            predictions = relative_preds.astype(int)
            return predictions
    
    # Step 3: Guarantee minimum predictions (top-k)
    # Sort valid indices by probability (descending)
    valid_indices_sorted = sorted(valid_fine_indices, key=lambda x: probs[x], reverse=True)
    
    for idx in valid_indices_sorted[:min_labels]:
        predictions[idx] = 1
    
    return predictions


def predict_fine_labels(classifier, dataloader, device, threshold=0.3,
                        gap_ratio=0.5, min_labels=1):
    """
    Generate fine label predictions for a dataloader using hybrid strategy.
    
    Returns:
        predictions: list of lists of fine label names
        probabilities: numpy array of shape (N, num_fine_labels)
    """
    from taxonomy_manager import TaxonomyManager
    
    all_predictions = []
    all_probabilities = []
    
    # Initialize taxonomy manager for coarse-to-fine mapping
    taxonomy_manager = TaxonomyManager(
        model=classifier.base_model,
        tokenizer=classifier.tokenizer,
        device=device
    )
    
    classifier.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting fine labels"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coarse_labels = batch['coarse_labels'].to(device)
            
            # Get predictions
            outputs = classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                coarse_labels=coarse_labels
            )
            
            # Apply sigmoid to masked logits
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            coarse_labels_np = coarse_labels.cpu().numpy()
            
            all_probabilities.append(probs)
            
            # Apply hybrid prediction for each sample
            for i in range(len(probs)):
                preds = apply_hybrid_prediction(
                    probs[i], 
                    coarse_labels_np[i],
                    taxonomy_manager,
                    threshold=threshold,
                    gap_ratio=gap_ratio,
                    min_labels=min_labels
                )
                
                # Convert to label names
                fine_labels = []
                for j in range(len(preds)):
                    if preds[j] == 1:
                        fine_labels.append(fine_id2label[j])
                all_predictions.append(fine_labels)
    
    return all_predictions, np.vstack(all_probabilities)


def combine_predictions(df, fine_predictions):
    """
    Combine coarse and fine predictions into final output format.
    
    Output format: [coarse_label, fine_label1, fine_label2, ...]
    """
    final_labels = []
    
    for idx, row in df.iterrows():
        coarse_label = row['predicted_coarse']
        fine_labels = fine_predictions[idx]
        
        # Combine: coarse first, then fine labels
        combined = [coarse_label] + fine_labels
        final_labels.append(combined)
    
    return final_labels


def evaluate_predictions(df, final_predictions):
    """
    Evaluate predictions against ground truth if available.
    """
    if 'labels' not in df.columns:
        print("⚠️ No ground truth labels found, skipping evaluation")
        return
    
    print("\n📊 Evaluation Results:")
    print("-" * 50)
    
    # Parse ground truth
    gt_coarse = []
    gt_fine = []
    pred_coarse = []
    pred_fine = []
    
    for idx, row in df.iterrows():
        labels = row['labels']
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)
        
        gt_coarse.append(labels[0])
        gt_fine.append(set(labels[1:]))
        
        pred_labels = final_predictions[idx]
        pred_coarse.append(pred_labels[0])
        pred_fine.append(set(pred_labels[1:]))
    
    # Coarse accuracy
    coarse_correct = sum(1 for gt, pred in zip(gt_coarse, pred_coarse) if gt == pred)
    coarse_acc = coarse_correct / len(gt_coarse)
    print(f"Coarse Accuracy: {coarse_acc:.4f}")
    
    # Fine-level metrics (per-sample)
    fine_f1_scores = []
    for gt_set, pred_set in zip(gt_fine, pred_fine):
        if len(gt_set) == 0 and len(pred_set) == 0:
            fine_f1_scores.append(1.0)
        elif len(gt_set) == 0 or len(pred_set) == 0:
            fine_f1_scores.append(0.0)
        else:
            tp = len(gt_set & pred_set)
            precision = tp / len(pred_set) if pred_set else 0
            recall = tp / len(gt_set) if gt_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fine_f1_scores.append(f1)
    
    avg_fine_f1 = np.mean(fine_f1_scores)
    print(f"Fine Role F1 (sample avg): {avg_fine_f1:.4f}")
    
    # Exact match (both coarse and fine correct)
    exact_matches = sum(
        1 for i in range(len(gt_coarse))
        if gt_coarse[i] == pred_coarse[i] and gt_fine[i] == pred_fine[i]
    )
    exact_match_acc = exact_matches / len(gt_coarse)
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
    
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Entity Role Classification Inference")
    parser.add_argument('--input', type=str, default=COARSE_PREDICTIONS_TEST,
                        help='Input CSV with coarse predictions')
    parser.add_argument('--output', type=str, default=FINAL_PREDICTIONS_PATH,
                        help='Output CSV path for final predictions')
    parser.add_argument('--threshold', type=float, default=FINE_THRESHOLD,
                        help='Threshold for fine label prediction')
    parser.add_argument('--gap_ratio', type=float, default=FINE_GAP_RATIO,
                        help='Relative threshold ratio for fallback prediction')
    parser.add_argument('--min_labels', type=int, default=FINE_MIN_LABELS,
                        help='Minimum number of fine labels to predict per sample')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    args = parser.parse_args()
    
    print("="*100)
    print("HIERARCHICAL ENTITY ROLE CLASSIFICATION - INFERENCE")
    print("="*100)
    
    # Check input exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        print("\nPlease run train_coarse.py first to generate coarse predictions.")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Using device: {device}")
    
    # Load data with coarse predictions
    print(f"\n📂 Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   Total samples: {len(df)}")
    
    if 'predicted_coarse' not in df.columns:
        print("❌ Error: 'predicted_coarse' column not found!")
        print("   Please run train_coarse.py first.")
        return
    
    # Initialize tokenizer
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [ENTITY_START_TOKEN, ENTITY_END_TOKEN]
    })
    
    # Load fine classifier
    fine_classifier = load_fine_classifier(FINE_CHECKPOINT_DIR, tokenizer, device)
    
    # Create dataset and dataloader
    print("\n📊 Creating inference dataset...")
    
    # Check if ground truth labels are available (for evaluation)
    has_labels = 'labels' in df.columns
    
    if has_labels:
        dataset = FineRoleDataset(df, tokenizer, max_length=MAX_LENGTH, use_predicted_coarse=True)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn_fine_role
        )
    else:
        dataset = FineRoleInferenceDataset(df, tokenizer, max_length=MAX_LENGTH)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn_fine_inference
        )
    
    # Generate fine predictions
    print("\n🔮 Generating fine label predictions...")
    print(f"   Hybrid strategy: threshold={args.threshold}, gap_ratio={args.gap_ratio}, min_labels={args.min_labels}")
    fine_predictions, fine_probs = predict_fine_labels(
        fine_classifier, dataloader, device, 
        threshold=args.threshold,
        gap_ratio=args.gap_ratio,
        min_labels=args.min_labels
    )
    
    # Combine predictions
    print("\n🔗 Combining coarse and fine predictions...")
    final_predictions = combine_predictions(df, fine_predictions)
    
    # Add to dataframe
    df['predicted_labels'] = [str(labels) for labels in final_predictions]
    df['predicted_fine_labels'] = [str(labels) for labels in fine_predictions]
    
    # Add fine probabilities for each class
    for fine_name, fine_id in fine_label2id.items():
        safe_name = fine_name.lower().replace(' ', '_')
        df[f'fine_proba_{safe_name}'] = fine_probs[:, fine_id]
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Saved predictions to: {args.output}")
    
    # Evaluate if ground truth available
    if has_labels:
        evaluate_predictions(df, final_predictions)
    
    # Summary statistics
    print("\n📈 Prediction Summary:")
    print("-" * 50)
    
    # Coarse distribution
    coarse_counts = df['predicted_coarse'].value_counts()
    print("Coarse label distribution:")
    for label, count in coarse_counts.items():
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Fine label counts
    all_fine = []
    for labels in fine_predictions:
        all_fine.extend(labels)
    
    if all_fine:
        from collections import Counter
        fine_counts = Counter(all_fine)
        print("\nTop 10 fine labels:")
        for label, count in fine_counts.most_common(10):
            print(f"   {label}: {count}")
    
    # Average number of fine labels per sample
    avg_fine = np.mean([len(labels) for labels in fine_predictions])
    print(f"\nAvg fine labels per sample: {avg_fine:.2f}")
    
    print("\n" + "="*100)
    print("INFERENCE COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()