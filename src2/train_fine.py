"""
Train Fine Role Classifier (Unified with Masking)

This script trains the unified fine role classifier that predicts multi-label
fine roles while respecting the hierarchical structure (masking based on coarse labels).

Prerequisites:
    - Run train_coarse.py first to generate coarse predictions

Usage:
    python train_fine.py
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from config import (
    COARSE_PREDICTIONS_TRAIN, COARSE_PREDICTIONS_VAL, COARSE_PREDICTIONS_TEST,
    FINE_CHECKPOINT_DIR, PREDICTIONS_ROOT,
    MODEL_NAME, MAX_LENGTH,
    FINE_NUM_EPOCHS, FINE_BATCH_SIZE, FINE_LEARNING_RATE,
    FINE_WARMUP_STEPS, FINE_NUM_UNFROZEN_LAYERS, FINE_THRESHOLD,
    NUM_FINE_LABELS,
    FINE_LOSS_TYPE, ASL_GAMMA_NEG, ASL_GAMMA_POS, ASL_CLIP, ASL_ENTROPY_WEIGHT
)
from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from datasets import (
    FineRoleDataset, collate_fn_fine_role,
    fine_label2id, fine_id2label, coarse_label2id
)
from hierarchical_model import FineRoleClassifier


def compute_fine_class_counts(df, label_col='labels'):
    """
    Compute class counts for each fine label (for class-balanced loss).
    
    Returns:
        list: Count for each fine label (length = NUM_FINE_LABELS)
    """
    counts = Counter()
    for raw in df[label_col].tolist():
        labels = ast.literal_eval(raw) if isinstance(raw, str) else raw
        for label in labels[1:]:  # Skip coarse label
            if label in fine_label2id:
                counts[fine_label2id[label]] += 1
    
    return [counts.get(i, 0) for i in range(NUM_FINE_LABELS)]


def stable_sigmoid(x):
    """Numerically stable sigmoid function."""
    # For positive values, use standard sigmoid
    # For negative values, use exp(x) / (1 + exp(x)) to avoid overflow
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    result = np.zeros_like(x, dtype=np.float64)
    
    # For positive x: 1 / (1 + exp(-x))
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    
    # For negative x: exp(x) / (1 + exp(x))
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1 + exp_x)
    
    return result


def compute_multilabel_metrics(eval_pred):
    """
    Compute multi-label metrics for HuggingFace Trainer.
    
    Note: This computes metrics over ALL fine labels. The masking is already
    applied in the model's forward pass (logits for invalid labels are -inf).
    """
    logits, labels = eval_pred
    
    # Handle tuple labels (when multiple label columns are returned)
    if isinstance(labels, tuple):
        # fine_labels is the first element based on our collate function
        labels = labels[0] if len(labels) > 0 else labels
    
    # Ensure labels is a numpy array
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Apply stable sigmoid and threshold
    probs = stable_sigmoid(logits.astype(np.float64))
    predictions = (probs >= FINE_THRESHOLD).astype(int)
    labels = labels.astype(int)
    
    # Flatten for micro-averaging (but only consider non-zero entries for proper evaluation)
    # We compute both micro and macro F1
    micro_f1 = f1_score(labels.flatten(), predictions.flatten(), average='micro', zero_division=0)
    macro_f1 = f1_score(labels.flatten(), predictions.flatten(), average='macro', zero_division=0)
    
    # Per-sample metrics (for samples with at least one positive label)
    sample_f1_scores = []
    for i in range(len(labels)):
        if labels[i].sum() > 0:  # Only consider samples with positive labels
            sample_f1 = f1_score(labels[i], predictions[i], average='micro', zero_division=0)
            sample_f1_scores.append(sample_f1)
    
    avg_sample_f1 = np.mean(sample_f1_scores) if sample_f1_scores else 0.0
    
    # Precision and recall
    micro_precision = precision_score(labels.flatten(), predictions.flatten(), average='micro', zero_division=0)
    micro_recall = recall_score(labels.flatten(), predictions.flatten(), average='micro', zero_division=0)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'sample_f1': avg_sample_f1,
        'precision': micro_precision,
        'recall': micro_recall,
    }


class FineRoleTrainer(Trainer):
    """
    Custom Trainer for FineRoleClassifier to handle multi-label evaluation properly.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to handle our custom model outputs."""
        # Extract labels - keep copies for later
        fine_labels = inputs.pop('fine_labels', None)
        coarse_labels = inputs.pop('coarse_labels', None)
        
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            coarse_labels=coarse_labels,
            fine_labels=fine_labels
        )
        
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step to properly return labels for metrics."""
        # Extract labels before they're removed
        fine_labels = inputs.get('fine_labels', None)
        coarse_labels = inputs.get('coarse_labels', None)
        
        # Move inputs to device
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # Forward pass
            fine_labels_tensor = inputs.pop('fine_labels', None)
            coarse_labels_tensor = inputs.pop('coarse_labels', None)
            
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                coarse_labels=coarse_labels_tensor,
                fine_labels=fine_labels_tensor
            )
            
            loss = outputs.loss
            logits = outputs.logits
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return fine_labels as the labels for metrics computation
        return (loss, logits, fine_labels_tensor)


def main():
    print("="*100)
    print("FINE ROLE CLASSIFIER TRAINING (UNIFIED WITH MASKING)")
    print("="*100)
    
    # Check if coarse predictions exist
    if not os.path.exists(COARSE_PREDICTIONS_TRAIN):
        print("❌ Error: Coarse predictions not found!")
        print(f"   Expected: {COARSE_PREDICTIONS_TRAIN}")
        print("\n   Please run train_coarse.py first.")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Using device: {device}")
    
    # Load coarse predictions
    print("\n📂 Loading coarse predictions...")
    train_df = pd.read_csv(COARSE_PREDICTIONS_TRAIN)
    val_df = pd.read_csv(COARSE_PREDICTIONS_VAL)
    test_df = pd.read_csv(COARSE_PREDICTIONS_TEST)
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Verify coarse predictions are present
    if 'predicted_coarse' not in train_df.columns:
        print("❌ Error: 'predicted_coarse' column not found in training data!")
        return
    
    # Initialize tokenizer
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [ENTITY_START_TOKEN, ENTITY_END_TOKEN]
    })
    
    # Load base model
    print(f"🧠 Loading base model: {MODEL_NAME}")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model = base_model.to(device)
    
    # Create datasets (using ground-truth coarse for training)
    print("\n📊 Creating datasets...")
    train_dataset = FineRoleDataset(train_df, tokenizer, max_length=MAX_LENGTH, use_predicted_coarse=False)
    val_dataset = FineRoleDataset(val_df, tokenizer, max_length=MAX_LENGTH, use_predicted_coarse=False)
    test_dataset = FineRoleDataset(test_df, tokenizer, max_length=MAX_LENGTH, use_predicted_coarse=False)
    
    # Compute fine class counts for class-balanced loss
    fine_class_counts = compute_fine_class_counts(train_df)
    print(f"\n📈 Fine label distribution (top 10):")
    sorted_counts = sorted(enumerate(fine_class_counts), key=lambda x: x[1], reverse=True)
    for idx, count in sorted_counts[:10]:
        print(f"   {fine_id2label[idx]}: {count}")
    
    # Initialize classifier
    print("\n🏗️ Initializing FineRoleClassifier...")
    print(f"   Loss type: {FINE_LOSS_TYPE}")
    if FINE_LOSS_TYPE in ['asl', 'asl_optimized']:
        print(f"   ASL params: gamma_neg={ASL_GAMMA_NEG}, gamma_pos={ASL_GAMMA_POS}, clip={ASL_CLIP}")
        if FINE_LOSS_TYPE == 'asl_optimized':
            print(f"   Entropy regularization: weight={ASL_ENTROPY_WEIGHT}")
    
    classifier = FineRoleClassifier(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        class_counts=fine_class_counts,
        num_unfrozen_layers=FINE_NUM_UNFROZEN_LAYERS,
        threshold=FINE_THRESHOLD,
        loss_type=FINE_LOSS_TYPE,
        gamma_neg=ASL_GAMMA_NEG,
        gamma_pos=ASL_GAMMA_POS,
        clip=ASL_CLIP,
        entropy_weight=ASL_ENTROPY_WEIGHT
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=FINE_CHECKPOINT_DIR,
        num_train_epochs=FINE_NUM_EPOCHS,
        per_device_train_batch_size=FINE_BATCH_SIZE,
        per_device_eval_batch_size=FINE_BATCH_SIZE,
        learning_rate=FINE_LEARNING_RATE,
        warmup_steps=FINE_WARMUP_STEPS,
        save_strategy='epoch',
        eval_strategy='epoch',
        logging_steps=50,
        logging_dir='./logs/fine',
        load_best_model_at_end=True,
        metric_for_best_model='micro_f1',
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to='none',
    )
    
    # Create custom Trainer
    trainer = FineRoleTrainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_fine_role,
        compute_metrics=compute_multilabel_metrics,
    )
    
    # Train
    print("\n" + "="*100)
    print("STARTING TRAINING")
    print("="*100)
    train_result = trainer.train()
    
    print(f"\n{'='*100}")
    print("TRAINING COMPLETED")
    print(f"{'='*100}")
    print(f"Train Loss: {train_result.training_loss:.4f}")
    
    # Evaluate on validation set
    print(f"\n{'='*100}")
    print("EVALUATING ON VALIDATION SET")
    print(f"{'='*100}")
    val_results = trainer.evaluate(val_dataset, metric_key_prefix='val')
    print(f"Val Micro-F1:  {val_results['val_micro_f1']:.4f}")
    print(f"Val Macro-F1:  {val_results['val_macro_f1']:.4f}")
    print(f"Val Sample-F1: {val_results['val_sample_f1']:.4f}")
    print(f"Val Precision: {val_results['val_precision']:.4f}")
    print(f"Val Recall:    {val_results['val_recall']:.4f}")
    
    # Evaluate on test set
    print(f"\n{'='*100}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*100}")
    test_results = trainer.evaluate(test_dataset, metric_key_prefix='test')
    print(f"Test Micro-F1:  {test_results['test_micro_f1']:.4f}")
    print(f"Test Macro-F1:  {test_results['test_macro_f1']:.4f}")
    print(f"Test Sample-F1: {test_results['test_sample_f1']:.4f}")
    print(f"Test Precision: {test_results['test_precision']:.4f}")
    print(f"Test Recall:    {test_results['test_recall']:.4f}")
    
    # Save final model
    print(f"\n{'='*100}")
    print("SAVING FINAL MODEL")
    print(f"{'='*100}")
    trainer.save_model(FINE_CHECKPOINT_DIR)
    print(f"✅ Model saved to: {FINE_CHECKPOINT_DIR}")
    
    print(f"\n{'='*100}")
    print("FINE CLASSIFIER TRAINING COMPLETE")
    print(f"{'='*100}")
    print("\nNext step: Run inference.py to generate final predictions")


if __name__ == "__main__":
    main()