import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import gc
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import nltk

# Ensure nltk resources
nltk.download('punkt', quiet=True)

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_ROOT = "./data"
TRAIN_DATA_PARENT = os.path.join(DATA_ROOT, "target_4_December_release")
VAL_DATA_PARENT = os.path.join(DATA_ROOT, "cleaned_dev_10_january_2025")
# Using VAL as TEST based on your script snippet
TEST_DATA_PARENT = VAL_DATA_PARENT 
TAXONOMY_FILE = os.path.join(DATA_ROOT, "taxonomy.json")
TEST_DATA_RESULTS = os.path.join(DATA_ROOT, "results")
ENSEMBLE_PREDICTION_FILE = "ensemble_predictions_standardized.tsv"

# === ENSEMBLE CONFIGURATION ===
# All models now use 'sentence' mode to ensure consistent test set size (604 samples).
# The ensemble diversity comes from the different model IDs and hyperparameters.
ENSEMBLE_CONFIGS = [
    {
        "name": "xlmr_sentence_v1",
        "model_id": "xlm-roberta-base",
        "data_mode": "paragraph",
        "batch_size": 16,
        "lr": 2e-5
    },
    {
        "name": "distilbert_sentence",
        "model_id": "distilbert-base-multilingual-cased", # Replaced xlm-roberta-base with DistilBERT for architectural diversity
        "data_mode": "paragraph",
        "batch_size": 32, # DistilBERT is smaller, can use a larger batch size
        "lr": 3e-5
    },
    {
        "name": "mbert_sentence",
        "model_id": "bert-base-multilingual-cased", # Different model architecture
        "data_mode": "paragraph",
        "batch_size": 16, 
        "lr": 3e-5 
    }
]

# ==========================================
# DATA LOADING UTILITIES
# ==========================================

def load_taxonomy():
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    main_categories = [cat["name"] for cat in taxonomy]
    label2id = {label: i for i, label in enumerate(main_categories)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

# Load Labels immediately
label2id, id2label = load_taxonomy()
available_languages = [d for d in os.listdir(TRAIN_DATA_PARENT) if os.path.isdir(os.path.join(TRAIN_DATA_PARENT, d))]

def split_sentences_with_offsets(text):
    """Splits text into sentences using NLTK and returns (start, end, text) tuples.
    The end index is inclusive (points to the last character)."""
    parts = []
    sentences = nltk.sent_tokenize(text)
    search_offset = 0
    for s in sentences:
        idx = text.find(s, search_offset)
        # Fallback if find fails (shouldn't happen often if using search_offset)
        if idx == -1: 
            try:
                 idx = text.index(s)
            except ValueError:
                 # Skip if sentence is genuinely not found
                 search_offset += len(s) 
                 continue
        
        # Calculate the INCLUSIVE end index: start + length - 1
        inclusive_end = idx + len(s) - 1 
        
        parts.append((idx, inclusive_end, s))
        search_offset = inclusive_end + 1
    return parts

def expand_to_smaller_units(df):
    """
    Expands document-level annotations into sentence-level annotations.
    Only sentence splitting is used for standardization.
    """
    new_rows = []
    # Alignment buffer to forgive minor NLTK segmentation errors
    buffer = 2 

    for _, row in df.iterrows():
        text, mention = row["text"], row["mention"]
        orig_start, orig_end = row["start"], row["end"]
        
        units = split_sentences_with_offsets(text)

        for unit_start, unit_end, unit_text in units:
            # Check if the original annotation starts within the sentence unit boundary (with buffer)
            if not (unit_start - buffer <= orig_start <= unit_end + buffer): 
                continue
            
            # --- Recalculate local offsets relative to the unit_text ---
            
            # 1. Global to Local Start
            local_start = orig_start - unit_start
            
            # 2. Local End (length is inclusive: end - start + 1)
            mention_length = orig_end - orig_start + 1
            local_end = local_start + mention_length - 1

            # 3. Sanity check: verify the mention is still correct at local indices
            # Correct misalignment if necessary by finding the mention within the unit_text
            try:
                 # We only check up to local_end + 1 because python slicing is exclusive
                 if unit_text[local_start : local_end + 1] != mention:
                     raise IndexError # Force correction logic
            except (IndexError, TypeError):
                 # Find the mention within the unit text
                 idx = unit_text.find(mention)
                 if idx == -1: continue # If mention can't be found, skip
                 
                 local_start, local_end = idx, idx + len(mention) - 1

            new_row = row.copy()
            new_row["text"] = unit_text
            new_row["start"] = local_start
            new_row["end"] = local_end
            new_row["orig_start"] = orig_start
            new_row["orig_end"] = orig_end
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)

def load_annotations(annotation_path, docs_root, labeled=True):
    data = []
    if not os.path.exists(annotation_path): return pd.DataFrame()
    
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if labeled:
                if len(parts) < 5: 
                    print(f"Skipping malformed labeled line: {line.strip()}")
                    continue
                doc_id, mention, start, end, *labels = parts
                label = labels[0]
            else:
                if len(parts) < 4:
                    print(f"Skipping malformed unlabeled line: {line.strip()}")
                    continue
                doc_id, mention, start, end = parts[:4]
                label = None
            
            try:
                start, end = int(start), int(end)
            except ValueError:
                print(f"Skipping line due to invalid start/end indices: {line.strip()}")
                continue

            text_path = os.path.join(docs_root, doc_id)
            if not os.path.exists(text_path): continue
            
            with open(text_path, "r", encoding="utf-8") as doc_file:
                text = doc_file.read()
            
            entry = {"doc_id": doc_id, "text": text, "mention": mention, "start": start, "end": end}
            if labeled: entry["label"] = label
            data.append(entry)
    return pd.DataFrame(data)

def load_data_for_config():
    """Loads and splits the data using only sentence-level expansion."""
    print(f"\nProcessing Data for Sentence Mode (Standardized)")
    all_train, all_test = [], []
    
    for lang in available_languages:
        train_root = os.path.join(TRAIN_DATA_PARENT, lang, "raw-documents")
        train_ann = os.path.join(TRAIN_DATA_PARENT, lang, "subtask-1-annotations.txt")
        val_root = os.path.join(VAL_DATA_PARENT, lang, "subtask-1-documents")
        val_ann = os.path.join(VAL_DATA_PARENT, lang, "subtask-1-annotations.txt")
        
        test_root = val_root
        test_ann = val_ann

        train_df_doc = load_annotations(train_ann, train_root, labeled=True)
        test_df_doc = load_annotations(test_ann, test_root, labeled=False)

        if not train_df_doc.empty: all_train.append(expand_to_smaller_units(train_df_doc))
        if not test_df_doc.empty: all_test.append(expand_to_smaller_units(test_df_doc))

    train_full = pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame()
    test_full = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
    
    # Internal validation split
    if not train_full.empty:
        if 'label' in train_full.columns:
            train_split, val_split = train_test_split(train_full, test_size=0.15, random_state=42, stratify=train_full['label'])
        else:
            print("Warning: 'label' column missing for stratification. Performing simple split.")
            train_split, val_split = train_test_split(train_full, test_size=0.15, random_state=42)
    else:
        train_split, val_split = pd.DataFrame(), pd.DataFrame()
        
    return train_split, val_split, test_full

# ==========================================
# DATASET CLASS
# ==========================================
class EntityFramingDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len=256, labeled=True):
        self.df = df
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.labeled = labeled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text, mention = row["text"], row["mention"]
        # Special tokens to highlight entity
        marked_text = text.replace(mention, f"[ENTITY] {mention} [/ENTITY]")
        
        inputs = self.tokenizer(
            marked_text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        
        if self.labeled:
            item["labels"] = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1}

# ==========================================
# TRAINING & PREDICTION ENGINE
# ==========================================
# Load the data once for all models since the split strategy is now identical
TRAIN_DF_GLOBAL, VAL_DF_GLOBAL, TEST_DF_GLOBAL = load_data_for_config()
print(f"GLOBAL DATASET LOADED - Train Size: {len(TRAIN_DF_GLOBAL)} | Val Size: {len(VAL_DF_GLOBAL)} | Test Size: {len(TEST_DF_GLOBAL)}")


def train_and_predict_single_model(config, run_id, train_df, val_df, test_df):
    print(f"\n{'='*10} Starting Run {run_id+1}: {config['name']} {'='*10}")
    
    # The sizes should now be consistent:
    # print(f"Train Size: {len(train_df)} | Val Size: {len(val_df)} | Test Size: {len(test_df)}")

    if train_df.empty or test_df.empty:
        print(f"Skipping run {config['name']} due to empty training or test data.")
        num_classes = len(label2id)
        if not test_df.empty:
             dummy_logits = np.zeros((len(test_df), num_classes))
        else:
             return np.array([]), pd.DataFrame()
        return dummy_logits, test_df


    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    special_tokens = {'additional_special_tokens': ['[ENTITY]', '[/ENTITY]']}
    tokenizer.add_special_tokens(special_tokens)

    # 3. Datasets
    train_dataset = EntityFramingDataset(train_df, tokenizer, label2id, labeled=True)
    val_dataset = EntityFramingDataset(val_df, tokenizer, label2id, labeled=True)
    test_dataset = EntityFramingDataset(test_df, tokenizer, label2id, labeled=False)

    # 4. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_id'],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Smart Freezing (Optional: Keep Classifier + Last Layer + Pooler)
    for name, param in model.named_parameters():
        if not any(x in name for x in ['pooler', 'classifier', 'layer.11', 'layer.23']): 
            param.requires_grad = False
    
    model.to(DEVICE)

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=f"./results/{config['name']}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config['lr'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=3, 
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"./logs/{config['name']}",
        logging_steps=50,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(f"Generating predictions for {config['name']}...")
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions 
    
    # Clean up memory
    del model, trainer, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return logits, test_df

# ==========================================
# MAIN EXECUTION: ENSEMBLE LOGIC
# ==========================================
def main():
    ensemble_logits = []
    reference_test_df = None
    reference_shape = None 

    # --- Phase 1: Train all models and collect Logits ---
    for i, config in enumerate(ENSEMBLE_CONFIGS):
        # Pass the globally loaded (and size-aligned) data
        logits, test_df_out = train_and_predict_single_model(
            config, 
            run_id=i, 
            train_df=TRAIN_DF_GLOBAL, 
            val_df=VAL_DF_GLOBAL, 
            test_df=TEST_DF_GLOBAL
        )
        
        if logits.size == 0:
             print(f"Skipping ensemble contribution for {config['name']} due to empty predictions.")
             continue

        current_shape = logits.shape

        # Since we use the same split strategy, the shapes SHOULD match now.
        if reference_shape is None:
            reference_shape = current_shape
            reference_test_df = test_df_out
        elif current_shape != reference_shape:
            # This should no longer happen but is kept for safety
            print(f"❌ WARNING: Prediction shape mismatch for {config['name']}! Expected shape {reference_shape}, got {current_shape}. Skipping contribution to ensemble.")
            continue 
        
        # Apply Softmax to convert Logits -> Probabilities
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        ensemble_logits.append(probs)

    if not ensemble_logits:
        print("❌ Ensemble failed: No models generated successful predictions.")
        return

    print("\n" + "="*30)
    print("CALCULATING ENSEMBLE VOTES")
    print("="*30)

    # --- Phase 2: Soft Voting (Averaging Probabilities) ---
    all_probs_stack = np.array(ensemble_logits)
    avg_probs = np.mean(all_probs_stack, axis=0)
    final_pred_indices = np.argmax(avg_probs, axis=1)
    final_labels = [id2label[idx] for idx in final_pred_indices]

    # --- Phase 3: Export Results ---
    reference_test_df["predicted_label"] = final_labels
    reference_test_df["confidence"] = np.max(avg_probs, axis=1)

    os.makedirs(TEST_DATA_RESULTS, exist_ok=True)
    output_file = os.path.join(TEST_DATA_RESULTS, ENSEMBLE_PREDICTION_FILE)

    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in reference_test_df.iterrows():
            f.write(f"{row['doc_id']}\t{row['mention']}\t{row['orig_start']}\t{row['orig_end']}\t{row['predicted_label']}\n")

    print(f"✅ Ensemble predictions saved to: {output_file}")
    print(reference_test_df[['mention', 'predicted_label', 'confidence']].head(10))
    
    return output_file


# ==========================================
# EVALUATION LOGIC
# ==========================================

print("\n=== Multilingual Evaluation (Updated for Doc/Paragraph/Sentence Modes) ===")

def evaluate_predictions_per_language(languages, label2id, prediction_file_path):
    """
    Evaluates the final ensemble predictions against the original ground truth annotations.
    """
    all_eval_dfs = []
    language_scores = []
    
    if not os.path.exists(prediction_file_path):
        print(f"⚠️ Prediction file not found at {prediction_file_path}. Cannot perform evaluation.")
        return None, None
        
    for lang in languages:
        print(f"\n🌍 Evaluating language: {lang}")

        # Use the single ensemble prediction file
        predictions_file = prediction_file_path

        test_ann_path = os.path.join(TEST_DATA_PARENT, lang, "subtask-1-annotations.txt")
        if not os.path.exists(test_ann_path):
            print(f"⚠️ Missing ground truth for {lang}, skipping.")
            continue

        # ------------------------------------------------------------
        # Load predictions (UPDATED FORMAT)
        # ------------------------------------------------------------
        predictions_df = pd.read_csv(
            predictions_file,
            sep="\t",
            names=["doc_id", "mention", "orig_start", "orig_end", "predicted_label"]
        )
        
        # Filter predictions to only include the current language documents (optional, but good practice)
        # However, the ensemble file might not contain doc_id prefixes, so we trust the merge key.

        # ------------------------------------------------------------
        # Load ground truth
        # ------------------------------------------------------------
        ground_truth_data = []
        with open(test_ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    doc_id, mention, start, end, *labels = parts
                    try:
                        ground_truth_data.append({
                            "doc_id": doc_id,
                            "mention": mention,
                            "orig_start": int(start),
                            "orig_end": int(end),
                            "true_label": labels[0]
                        })
                    except ValueError:
                        # Skip if start/end are not integers
                        continue

        ground_truth_df = pd.DataFrame(ground_truth_data)

        # ------------------------------------------------------------
        # Convert types (for robust merging)
        # ------------------------------------------------------------
        # Already handled by pd.read_csv and the try/except block above, but ensuring consistency
        
        # ------------------------------------------------------------
        # Merge ON GLOBAL OFFSETS (critical fix)
        # ------------------------------------------------------------
        eval_df = predictions_df.merge(
            ground_truth_df,
            on=["doc_id", "mention", "orig_start", "orig_end"],
            how="inner"
        )

        if eval_df.empty:
            print(f"⚠️ No matching examples found for {lang}. Check prediction file format/offsets.")
            continue

        # ------------------------------------------------------------
        # Convert labels to IDs
        # ------------------------------------------------------------
        y_true = [label2id[label] for label in eval_df["true_label"]]
        y_pred = [label2id[label] for label in eval_df["predicted_label"]]

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )

        language_scores.append({
            "Language": lang,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Samples": len(eval_df)
        })
        all_eval_dfs.append(eval_df)

        print(f"→ {lang} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

    # ------------------------------------------------------------
    # Combine and compute overall metrics
    # ------------------------------------------------------------
    if not all_eval_dfs:
        print("⚠️ No evaluation data collected across languages.")
        return None, None

    combined_eval = pd.concat(all_eval_dfs, ignore_index=True)

    combined_y_true = [label2id[label] for label in combined_eval["true_label"]]
    combined_y_pred = [label2id[label] for label in combined_eval["predicted_label"]]

    overall_acc = accuracy_score(combined_y_true, combined_y_pred)
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
        combined_y_true, combined_y_pred, average="micro", zero_division=0
    )

    print("\n=== 🌐 Overall Multilingual Evaluation ===")
    print(f"Accuracy : {overall_acc:.4f}")
    print(f"Precision: {overall_prec:.4f}")
    print(f"Recall   : {overall_rec:.4f}")
    print(f"F1       : {overall_f1:.4f}")

    language_report = pd.DataFrame(language_scores).sort_values("F1", ascending=False)

    print("\n=== Per-Language Summary ===")
    print(language_report.to_string(index=False))

    return combined_eval, language_report


if __name__ == "__main__":
    # 1. Run the training and generate the ensemble prediction file
    output_file_path = main()
    
    # 2. Run multilingual evaluation on the generated file
    if output_file_path:
        combined_eval_df, language_report_df = evaluate_predictions_per_language(
            available_languages, label2id, output_file_path
        )

        # Optional: per-class metrics
        if combined_eval_df is not None:
            y_true = [label2id[label] for label in combined_eval_df["true_label"]]
            y_pred = [label2id[label] for label in combined_eval_df["predicted_label"]]

            class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            performance_df = pd.DataFrame({
                "Category": list(label2id.keys()),
                "Precision": class_precision,
                "Recall": class_recall,
                "F1-Score": class_f1,
                "Support": class_support
            })

            print("\n=== Per-Category Performance (Multilingual Combined) ===")
            print(performance_df.sort_values("F1-Score", ascending=False).to_string(index=False))