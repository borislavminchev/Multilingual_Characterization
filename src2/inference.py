"""
End-to-End Inference Pipeline (with Soft Conditioning Support)

This script performs hierarchical inference with two modes:
1. Hard masking (legacy): Uses coarse labels to mask invalid fine labels
2. Soft conditioning (new): Uses coarse probabilities for soft hierarchy conditioning

Prerequisites:
    - Run train_coarse.py to train coarse classifier
    - Run train_fine.py to train fine classifier

Usage:
    python inference.py                    # Auto-detect mode from checkpoint
    python inference.py --soft             # Force soft conditioning
    python inference.py --hard             # Force hard masking
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
    FINE_GAP_RATIO, FINE_MIN_LABELS, FINE_MAX_LABELS,
    USE_SOFT_CONDITIONING, CARDINALITY_WEIGHT, TARGET_CARDINALITY,
    NUM_COARSE_LABELS, NUM_FINE_LABELS
)
from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from datasets import (
    FineRoleDataset, FineRoleInferenceDataset, collate_fn_fine_role, collate_fn_fine_inference,
    SoftConditionedFineRoleDataset, SoftConditionedInferenceDataset, collate_fn_soft_conditioned,
    coarse_label2id, coarse_id2label, fine_label2id, fine_id2label
)
from hierarchical_model import FineRoleClassifier, SoftConditionedFineClassifier


def load_fine_classifier(checkpoint_dir, tokenizer, device, use_soft=False):
    """Load trained fine classifier from checkpoint."""
    print(f"🧠 Loading fine classifier from: {checkpoint_dir}")
    
    # Load base model
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(device)

    # Initialize classifier based on mode
    if use_soft:
        classifier = SoftConditionedFineClassifier(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            threshold=FINE_THRESHOLD,
            num_coarse=NUM_COARSE_LABELS,
            num_fine=NUM_FINE_LABELS,
            cardinality_weight=CARDINALITY_WEIGHT,
            target_cardinality=TARGET_CARDINALITY
        )
    else:
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


def smart_prediction(probs, coarse_label, taxonomy_manager, 
                     threshold=0.5, gap_ratio=0.7, min_labels=1, max_labels=2):
    """
    Smart prediction strategy for HARD MASKING mode.
    Uses coarse label to filter valid fine labels.
    
    Args:
        probs: numpy array of shape (num_fine_labels,) - sigmoid probabilities
        coarse_label: int - coarse label ID for this sample
        taxonomy_manager: TaxonomyManager instance for getting valid fine labels
        threshold: float - primary threshold for predictions
        gap_ratio: float - adaptive threshold as ratio of top probability
        min_labels: int - minimum number of predictions to guarantee
        max_labels: int - maximum number of predictions allowed
    
    Returns:
        predictions: numpy array of shape (num_fine_labels,) - binary predictions
    """
    num_fine = len(probs)
    predictions = np.zeros(num_fine, dtype=int)
    
    # Get valid fine label indices for this coarse category
    valid_fine_indices = taxonomy_manager.get_fine_indices_for_coarse(coarse_label)
    
    if len(valid_fine_indices) == 0:
        return predictions
    
    # Sort valid indices by probability (descending)
    sorted_valid = sorted(
        [(probs[i], i) for i in valid_fine_indices], 
        key=lambda x: x[0], 
        reverse=True
    )
    
    if not sorted_valid:
        return predictions
    
    top_prob, top_idx = sorted_valid[0]
    
    # Calculate adaptive threshold
    adaptive_threshold = gap_ratio * top_prob
    effective_threshold = max(threshold, adaptive_threshold)
    
    selected_indices = []
    
    for prob, idx in sorted_valid:
        if len(selected_indices) >= max_labels:
            break
        
        if prob >= effective_threshold:
            selected_indices.append(idx)
        elif len(selected_indices) < min_labels:
            selected_indices.append(idx)
    
    # Ensure min_labels
    if len(selected_indices) < min_labels:
        for prob, idx in sorted_valid:
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) >= min_labels:
                    break
    
    for idx in selected_indices:
        predictions[idx] = 1
    
    return predictions


def soft_prediction(probs, threshold=0.5, gap_ratio=0.7, min_labels=1, max_labels=3):
    """
    Smart prediction strategy for SOFT CONDITIONING mode.
    No coarse-based filtering - the soft conditioning already handles hierarchy.
    
    Args:
        probs: numpy array of shape (num_fine_labels,) - sigmoid probabilities
        threshold: float - primary threshold for predictions
        gap_ratio: float - adaptive threshold as ratio of top probability
        min_labels: int - minimum number of predictions to guarantee
        max_labels: int - maximum number of predictions allowed
    
    Returns:
        predictions: numpy array of shape (num_fine_labels,) - binary predictions
    """
    num_fine = len(probs)
    predictions = np.zeros(num_fine, dtype=int)
    
    # Sort all labels by probability
    sorted_all = sorted(
        [(probs[i], i) for i in range(num_fine)], 
        key=lambda x: x[0], 
        reverse=True
    )
    
    if not sorted_all:
        return predictions
    
    top_prob, top_idx = sorted_all[0]
    
    # Adaptive threshold
    adaptive_threshold = gap_ratio * top_prob
    effective_threshold = max(threshold, adaptive_threshold)
    
    selected_indices = []
    
    for prob, idx in sorted_all:
        if len(selected_indices) >= max_labels:
            break
        
        if prob >= effective_threshold:
            selected_indices.append(idx)
        elif len(selected_indices) < min_labels:
            selected_indices.append(idx)
    
    # Ensure min_labels
    if len(selected_indices) < min_labels:
        for prob, idx in sorted_all:
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) >= min_labels:
                    break
    
    for idx in selected_indices:
        predictions[idx] = 1
    
    return predictions


def predict_fine_labels_hard(classifier, dataloader, device, threshold=0.5,
                             gap_ratio=0.7, min_labels=1, max_labels=2):
    """
    Generate fine label predictions using HARD MASKING mode.
    """
    from taxonomy_manager import TaxonomyManager
    
    all_predictions = []
    all_probabilities = []
    
    taxonomy_manager = TaxonomyManager(
        model=classifier.base_model,
        tokenizer=classifier.tokenizer,
        device=device
    )
    
    classifier.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting (hard masking)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coarse_labels = batch['coarse_labels'].to(device)
            
            outputs = classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                coarse_labels=coarse_labels
            )
            
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            coarse_labels_np = coarse_labels.cpu().numpy()
            
            all_probabilities.append(probs)
            
            for i in range(len(probs)):
                preds = smart_prediction(
                    probs[i], 
                    coarse_labels_np[i],
                    taxonomy_manager,
                    threshold=threshold,
                    gap_ratio=gap_ratio,
                    min_labels=min_labels,
                    max_labels=max_labels
                )
                
                fine_labels = []
                for j in range(len(preds)):
                    if preds[j] == 1:
                        fine_labels.append(fine_id2label[j])
                all_predictions.append(fine_labels)
    
    return all_predictions, np.vstack(all_probabilities)


def predict_fine_labels_soft(classifier, dataloader, device, threshold=0.5,
                             gap_ratio=0.7, min_labels=1, max_labels=3):
    """
    Generate fine label predictions using SOFT CONDITIONING mode.
    """
    all_predictions = []
    all_probabilities = []
    
    classifier.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting (soft conditioning)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            coarse_probs = batch['coarse_probs'].to(device)
            coarse_labels = batch['coarse_labels'].to(device)
            
            outputs = classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                coarse_probs=coarse_probs,
                coarse_labels=coarse_labels  # Fallback
            )
            
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            
            all_probabilities.append(probs)
            
            for i in range(len(probs)):
                preds = soft_prediction(
                    probs[i],
                    threshold=threshold,
                    gap_ratio=gap_ratio,
                    min_labels=min_labels,
                    max_labels=max_labels
                )
                
                fine_labels = []
                for j in range(len(preds)):
                    if preds[j] == 1:
                        fine_labels.append(fine_id2label[j])
                all_predictions.append(fine_labels)
    
    return all_predictions, np.vstack(all_probabilities)


def combine_predictions(df, fine_predictions):
    """
    Combine coarse and fine predictions into final output format.
    """
    final_labels = []
    
    for idx, row in df.iterrows():
        coarse_label = row['predicted_coarse']
        fine_labels = fine_predictions[idx]
        
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
    
    # Fine-level metrics
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
    
    # Exact match (fine-grained labels only, as per official EMR definition)
    exact_matches = sum(
        1 for i in range(len(gt_fine))
        if gt_fine[i] == pred_fine[i]
    )
    exact_match_acc = exact_matches / len(gt_coarse)
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
    
    # Additional metrics
    avg_pred_count = np.mean([len(preds) for preds in pred_fine])
    avg_gt_count = np.mean([len(gt) for gt in gt_fine])
    print(f"Avg Pred Labels/Sample: {avg_pred_count:.2f}")
    print(f"Avg GT Labels/Sample: {avg_gt_count:.2f}")
    
    print("-" * 50)


def detect_training_mode(checkpoint_dir):
    """
    Detect training mode from checkpoint directory.
    Returns 'soft', 'hard', or None if cannot detect.
    """
    mode_file = os.path.join(checkpoint_dir, 'training_mode.txt')
    if os.path.exists(mode_file):
        with open(mode_file, 'r') as f:
            for line in f:
                if line.startswith('mode='):
                    return line.strip().split('=')[1]
    
    # Check directory name
    if '_soft' in checkpoint_dir:
        return 'soft'
    elif '_hard' in checkpoint_dir:
        return 'hard'
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Entity Role Classification Inference")
    parser.add_argument('--input', type=str, default=COARSE_PREDICTIONS_TEST,
                        help='Input CSV with coarse predictions')
    parser.add_argument('--output', type=str, default=FINAL_PREDICTIONS_PATH,
                        help='Output CSV path for final predictions')
    parser.add_argument('--soft', action='store_true',
                        help='Use soft conditioning mode')
    parser.add_argument('--hard', action='store_true',
                        help='Use hard masking mode')
    parser.add_argument('--threshold', type=float, default=FINE_THRESHOLD,
                        help='Threshold for fine label prediction')
    parser.add_argument('--gap_ratio', type=float, default=FINE_GAP_RATIO,
                        help='Relative threshold ratio for fallback prediction')
    parser.add_argument('--min_labels', type=int, default=FINE_MIN_LABELS,
                        help='Minimum number of fine labels to predict per sample')
    parser.add_argument('--max_labels', type=int, default=FINE_MAX_LABELS,
                        help='Maximum number of fine labels to predict per sample')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    args = parser.parse_args()
    
    # Determine mode
    if args.soft:
        use_soft = True
    elif args.hard:
        use_soft = False
    else:
        # Auto-detect from config or checkpoint
        use_soft = USE_SOFT_CONDITIONING
    
    mode_str = "SOFT CONDITIONING" if use_soft else "HARD MASKING"
    
    # Determine checkpoint directory
    checkpoint_dir = FINE_CHECKPOINT_DIR + ("_soft" if use_soft else "_hard")
    
    # Fallback to old directory if new one doesn't exist
    if not os.path.exists(checkpoint_dir):
        if os.path.exists(FINE_CHECKPOINT_DIR):
            checkpoint_dir = FINE_CHECKPOINT_DIR
            print(f"⚠️ Using legacy checkpoint: {checkpoint_dir}")
            # Detect mode from legacy checkpoint
            detected_mode = detect_training_mode(checkpoint_dir)
            if detected_mode:
                use_soft = (detected_mode == 'soft')
                mode_str = "SOFT CONDITIONING" if use_soft else "HARD MASKING"
    
    print("="*100)
    print(f"HIERARCHICAL ENTITY ROLE CLASSIFICATION - INFERENCE ({mode_str})")
    print("="*100)
    
    # Check input exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        print("\nPlease run train_coarse.py first to generate coarse predictions.")
        return
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint not found: {checkpoint_dir}")
        print("\nPlease run train_fine.py first.")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Using device: {device}")
    
    # Load data
    print(f"\n📂 Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"   Total samples: {len(df)}")
    
    if 'predicted_coarse' not in df.columns:
        print("❌ Error: 'predicted_coarse' column not found!")
        return
    
    # Initialize tokenizer
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [ENTITY_START_TOKEN, ENTITY_END_TOKEN]
    })
    
    # Load fine classifier
    fine_classifier = load_fine_classifier(checkpoint_dir, tokenizer, device, use_soft=use_soft)
    
    # Create dataset and dataloader
    print("\n📊 Creating inference dataset...")
    has_labels = 'labels' in df.columns
    
    if use_soft:
        if has_labels:
            dataset = SoftConditionedFineRoleDataset(df, tokenizer, max_length=MAX_LENGTH, use_predicted_coarse=True)
        else:
            dataset = SoftConditionedInferenceDataset(df, tokenizer, max_length=MAX_LENGTH)
        collate_fn = collate_fn_soft_conditioned
    else:
        if has_labels:
            dataset = FineRoleDataset(df, tokenizer, max_length=MAX_LENGTH, use_predicted_coarse=True)
        else:
            dataset = FineRoleInferenceDataset(df, tokenizer, max_length=MAX_LENGTH)
        collate_fn = collate_fn_fine_role if has_labels else collate_fn_fine_inference
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn
    )
    
    # Generate fine predictions
    print("\n🔮 Generating fine label predictions...")
    print(f"   Mode: {mode_str}")
    print(f"   Parameters: threshold={args.threshold}, gap_ratio={args.gap_ratio}, "
          f"min_labels={args.min_labels}, max_labels={args.max_labels}")
    
    if use_soft:
        fine_predictions, fine_probs = predict_fine_labels_soft(
            fine_classifier, dataloader, device,
            threshold=args.threshold,
            gap_ratio=args.gap_ratio,
            min_labels=args.min_labels,
            max_labels=args.max_labels
        )
    else:
        fine_predictions, fine_probs = predict_fine_labels_hard(
            fine_classifier, dataloader, device,
            threshold=args.threshold,
            gap_ratio=args.gap_ratio,
            min_labels=args.min_labels,
            max_labels=args.max_labels
        )
    
    # Combine predictions
    print("\n🔗 Combining coarse and fine predictions...")
    final_predictions = combine_predictions(df, fine_predictions)
    
    # Add to dataframe
    df['predicted_labels'] = [str(labels) for labels in final_predictions]
    df['predicted_fine_labels'] = [str(labels) for labels in fine_predictions]
    
    # Add fine probabilities
    for fine_name, fine_id in fine_label2id.items():
        safe_name = fine_name.lower().replace(' ', '_')
        df[f'fine_prob_{safe_name}'] = fine_probs[:, fine_id]
    
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
    
    # Average number of fine labels
    avg_fine = np.mean([len(labels) for labels in fine_predictions])
    print(f"\nAvg fine labels per sample: {avg_fine:.2f}")
    
    print("\n" + "="*100)
    print(f"INFERENCE COMPLETE ({mode_str})")
    print("="*100)


if __name__ == "__main__":
    main()