"""
Train Coarse Role Classifier

This script trains the coarse role classifier (Protagonist/Antagonist/Innocent)
and generates predictions on train/val/test sets for use by the fine classifier.

Usage:
    python train_coarse.py
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from config import (
    TRAIN_DATA_RATIONALE, VAL_DATA_RATIONALE, TEST_DATA_RATIONALE,
    COARSE_CHECKPOINT_DIR, PREDICTIONS_ROOT,
    COARSE_PREDICTIONS_TRAIN, COARSE_PREDICTIONS_VAL, COARSE_PREDICTIONS_TEST,
    MODEL_NAME, MAX_LENGTH,
    COARSE_NUM_EPOCHS, COARSE_BATCH_SIZE, COARSE_LEARNING_RATE,
    COARSE_WARMUP_STEPS, COARSE_NUM_UNFROZEN_LAYERS
)
from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from datasets import (
    EntityFramingDataset, collate_fn_entity_framing,
    coarse_label2id, coarse_id2label
)
from hierarchical_model import CoarseRoleClassifier


def compute_class_counts_from_df(df, label_col='labels', coarse_map=coarse_label2id):
    """Compute class counts for class-balanced loss."""
    cnt = Counter()
    for raw in df[label_col].tolist():
        labels = ast.literal_eval(raw) if isinstance(raw, str) else raw
        coarse = labels[0]
        cnt[coarse_map[coarse]] += 1
    num_classes = len(coarse_map)
    return [cnt.get(i, 0) for i in range(num_classes)]


def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def generate_predictions(trainer, dataset, df, output_path, device):
    """
    Generate predictions on a dataset and save to CSV.
    
    Adds columns:
    - predicted_coarse: predicted coarse label name
    - predicted_coarse_id: predicted coarse label ID
    - coarse_confidence: confidence score of prediction
    - coarse_proba_*: probability for each coarse class
    """
    print(f"\nGenerating predictions for {len(dataset)} samples...")
    
    predictions_output = trainer.predict(dataset)
    logits = predictions_output.predictions
    
    # Convert to probabilities
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predicted_ids = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Create output dataframe
    output_df = df.copy()
    output_df['predicted_coarse'] = [coarse_id2label[pid] for pid in predicted_ids]
    output_df['predicted_coarse_id'] = predicted_ids
    output_df['coarse_confidence'] = confidences
    
    # Add per-class probabilities
    for class_name, class_id in coarse_label2id.items():
        output_df[f'coarse_proba_{class_name.lower()}'] = probs[:, class_id]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to {output_path}")
    
    # Print accuracy if ground truth is available
    if 'labels' in df.columns:
        gt_coarse = []
        for raw in df['labels'].tolist():
            labels = ast.literal_eval(raw) if isinstance(raw, str) else raw
            gt_coarse.append(labels[0])
        gt_ids = [coarse_label2id[c] for c in gt_coarse]
        acc = accuracy_score(gt_ids, predicted_ids)
        f1 = f1_score(gt_ids, predicted_ids, average='weighted', zero_division=0)
        print(f"   Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return output_df


def main():
    print("="*100)
    print("COARSE ROLE CLASSIFIER TRAINING")
    print("="*100)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Using device: {device}")
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(TRAIN_DATA_RATIONALE)
    val_df = pd.read_csv(VAL_DATA_RATIONALE)
    test_df = pd.read_csv(TEST_DATA_RATIONALE)
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
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
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = EntityFramingDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset = EntityFramingDataset(val_df, tokenizer, max_length=MAX_LENGTH)
    test_dataset = EntityFramingDataset(test_df, tokenizer, max_length=MAX_LENGTH)
    
    # Compute class counts for class-balanced loss
    class_counts = compute_class_counts_from_df(train_df)
    print(f"\n📈 Class distribution:")
    for class_name, class_id in coarse_label2id.items():
        print(f"   {class_name}: {class_counts[class_id]}")
    
    # Initialize classifier
    print("\n🏗️ Initializing CoarseRoleClassifier...")
    classifier = CoarseRoleClassifier(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        class_counts=class_counts,
        num_unfrozen_layers=COARSE_NUM_UNFROZEN_LAYERS
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=COARSE_CHECKPOINT_DIR,
        num_train_epochs=COARSE_NUM_EPOCHS,
        per_device_train_batch_size=COARSE_BATCH_SIZE,
        per_device_eval_batch_size=COARSE_BATCH_SIZE,
        learning_rate=COARSE_LEARNING_RATE,
        warmup_steps=COARSE_WARMUP_STEPS,
        save_strategy='epoch',
        eval_strategy='epoch',
        logging_steps=50,
        logging_dir='./logs/coarse',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to='none',  # Disable wandb/tensorboard
    )
    
    # Create Trainer
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_entity_framing,
        compute_metrics=compute_metrics,
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
    print(f"Val Accuracy:  {val_results['val_accuracy']:.4f}")
    print(f"Val F1 Score:  {val_results['val_f1']:.4f}")
    print(f"Val Precision: {val_results['val_precision']:.4f}")
    print(f"Val Recall:    {val_results['val_recall']:.4f}")
    
    # Evaluate on test set
    print(f"\n{'='*100}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*100}")
    test_results = trainer.evaluate(test_dataset, metric_key_prefix='test')
    print(f"Test Accuracy:  {test_results['test_accuracy']:.4f}")
    print(f"Test F1 Score:  {test_results['test_f1']:.4f}")
    print(f"Test Precision: {test_results['test_precision']:.4f}")
    print(f"Test Recall:    {test_results['test_recall']:.4f}")
    
    # Generate predictions on all splits
    print(f"\n{'='*100}")
    print("GENERATING PREDICTIONS FOR FINE CLASSIFIER")
    print(f"{'='*100}")
    
    os.makedirs(PREDICTIONS_ROOT, exist_ok=True)
    
    generate_predictions(trainer, train_dataset, train_df, COARSE_PREDICTIONS_TRAIN, device)
    generate_predictions(trainer, val_dataset, val_df, COARSE_PREDICTIONS_VAL, device)
    generate_predictions(trainer, test_dataset, test_df, COARSE_PREDICTIONS_TEST, device)
    
    print(f"\n{'='*100}")
    print("COARSE CLASSIFIER TRAINING COMPLETE")
    print(f"{'='*100}")
    print(f"\n📁 Model saved to: {COARSE_CHECKPOINT_DIR}")
    print(f"📁 Predictions saved to: {PREDICTIONS_ROOT}/")
    print("\nNext step: Run train_fine.py to train the fine role classifier")


if __name__ == "__main__":
    main()