# 1. Initialize global configuration. 
# 2. Call data_utils to load and split data. 
# 3. Orchestrate the OOF training loop (Tier 1). 
# 4. Orchestrate Tier 2 training (Option A/B setup). 
# 5. Run final inference on the test set. 
# 6. Call metrics.py for final evaluation.

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
from hierarchical_model import CoarseRoleClassifier
from config import TRAIN_DATA_RATIONALE, VAL_DATA_RATIONALE, TEST_DATA_RATIONALE
from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from datasets import EntityFramingDataset, collate_fn_entity_framing
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from collections import Counter
from datasets import coarse_label2id


def clalculate_class_weights(train_dataset, device):
    
    coarse_label_counts = Counter()
    for idx in range(len(train_dataset)):
        coarse_label = train_dataset[idx]['coarse_labels']
        coarse_label_counts[coarse_label] += 1
    
    # Calculate inverse frequency weights normalized
    num_classes = len(coarse_label2id)
    total_samples = sum(coarse_label_counts.values())
    class_weights = []
    for class_id in range(num_classes):
        if class_id in coarse_label_counts:
            # Inverse frequency: total / (num_classes * class_frequency)
            weight = total_samples / (num_classes * coarse_label_counts[class_id])
        else:
            # If class not in training data, assign neutral weight
            weight = 1.0
        class_weights.append(weight)
    
    print(f"\nClass Weights (inverse frequency):")
    for class_id, weight in enumerate(class_weights):
        class_name = list(coarse_label2id.keys())[class_id]
        count = coarse_label_counts.get(class_id, 0)
        print(f"  Class {class_id} ({class_name}): weight={weight:.4f}, count={count}")

    return torch.tensor(class_weights, dtype=torch.float32, device=device)        


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_df = pd.read_csv(TRAIN_DATA_RATIONALE)
    val_df = pd.read_csv(VAL_DATA_RATIONALE)
    test_df = pd.read_csv(TEST_DATA_RATIONALE)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    tokenizer.add_special_tokens({'additional_special_tokens': [ENTITY_START_TOKEN, ENTITY_END_TOKEN]})
    
    # Load model and resize embeddings to match tokenizer vocab size
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    train_dataset = EntityFramingDataset(train_df, tokenizer=tokenizer)
    val_dataset = EntityFramingDataset(val_df, tokenizer=tokenizer)
    test_dataset = EntityFramingDataset(test_df, tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_entity_framing)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_entity_framing)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_entity_framing)
    
    class_weights = clalculate_class_weights(train_dataset, device=device)
    
    classifier = CoarseRoleClassifier(model, tokenizer, device=device, class_weights=class_weights, num_unfrozen_layers=3)

    # for batch_idx, batch in enumerate(train_dataloader):
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     logits = classifier(input_ids, attention_mask)
        
    #     print(f"\n[Batch {batch_idx}] Raw logits shape: {logits.shape}, dtype: {logits.dtype}")
    #     print(f"[Batch {batch_idx}] Raw logits sample: {logits[0]}")
        
    #     # Apply softmax to get probabilities
    #     probs = torch.softmax(logits, dim=-1)
    #     print(f"[Batch {batch_idx}] Probabilities sum: {probs[0].sum():.4f} (should be 1.0)")
        
    #     # Get predictions and confidence scores
    #     predicted_classes = torch.argmax(logits, dim=-1)
    #     predicted_confidence = torch.max(probs, dim=-1).values
        
    #     # Get actual labels (coarse_labels)
    #     actual_labels = batch['coarse_labels']
        
    #     print(f"[Batch {batch_idx}] Predictions: {predicted_classes}, Actuals: {actual_labels}")
    #     print(f"[Batch {batch_idx}] Confidence: {predicted_confidence}")


    # =====================================================
    # TRAINING WITH HUGGINGFACE TRAINER
    # =====================================================

    print("\n" + "="*100)
    print("STARTING TRAINING WITH HUGGINGFACE TRAINER")
    print("="*100)

    # Define compute metrics function
    def compute_metrics(eval_pred):
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results/coarse_role_classifier',
        num_train_epochs=8,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=100,
        save_strategy='best',
        eval_strategy='epoch',
        logging_steps=50,
        logging_dir='./logs',
        load_best_model_at_end=True,
        save_total_limit=2,
        remove_unused_columns=False,
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
    print("Starting training...")
    train_result = trainer.train()

    print(f"\n{'='*100}")
    print("TRAINING COMPLETED")
    print(f"{'='*100}")
    print(f"Train Loss: {train_result.training_loss:.4f}")

    # ========== TEST EVALUATION ==========
    print(f"\n{'='*100}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*100}")

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset, metric_key_prefix='test')

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_results['test_accuracy']:.4f}")
    print(f"  F1 Score:  {test_results['test_f1']:.4f}")
    print(f"  Precision: {test_results['test_precision']:.4f}")
    print(f"  Recall:    {test_results['test_recall']:.4f}")
    print(f"{'='*100}")

    # Get predictions with confidence scores
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    confidence_scores = np.max(probs.numpy(), axis=1)

    print(f"Average Confidence: {np.mean(confidence_scores):.4f}")
    print(f"Min Confidence: {np.min(confidence_scores):.4f}")
    print(f"Max Confidence: {np.max(confidence_scores):.4f}")
    print(f"{'='*100}")