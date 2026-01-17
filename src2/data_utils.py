import pandas as pd
import json
import os
import nltk
from sklearn.model_selection import train_test_split, KFold
from config import TAXONOMY_FILE, TRAIN_DATA_PARENT, VAL_DATA_PARENT, TEST_DATA_PARENT

# Download NLTK punkt tokenizer
nltk.download("punkt", quiet=True)

ENTITY_START_TOKEN = "[ENTITY]"
ENTITY_END_TOKEN = "[/ENTITY]"

def split_sentences_with_offsets(text):
    """Splits text into sentences using NLTK and returns (start, end, text) tuples.
    The end index is inclusive (points to the last character)."""
    parts = []
    sentences = nltk.sent_tokenize(text)
    search_offset = 0
    for s in sentences:
        idx = text.find(s, search_offset)
        if idx == -1: 
            continue
        
        end = idx + len(s)
        if s.strip():
            parts.append((idx, end, s))
        search_offset = end
    
    return parts

def split_paragraphs_with_offsets(text):
    """Splits text into paragraphs and returns (start, end, text) tuples.
    The end index is inclusive (points to the last character)."""
    parts = []
    offset = 0
    for raw in text.split("\n"):
        idx = text.find(raw, offset)
        if idx == -1: 
            continue
        
        end = idx + len(raw)
        if raw.strip():
            parts.append((idx, end, raw))
        offset = end

    return parts

def expand_to_smaller_units(df, mode):
    if mode not in ["paragraph", "sentence"]:
        raise ValueError(f"Unsupported mode: {mode}. Use 'paragraph' or 'sentence'.")
    
    new_rows = []
    splitter = split_paragraphs_with_offsets if mode == "paragraph" else split_sentences_with_offsets

    for _, row in df.iterrows():
        text, mention = row["text"], row["mention"]
        orig_start, orig_end = row["start"], row["end"]
        
        
        units = splitter(text)

        found = False
        for unit_start, unit_end, unit_text in units:
            # Check if unit contains the entire mention span
            if not (unit_start <= orig_start and orig_end <= unit_end):
                continue

            found = True
            
            mention_length = orig_end - orig_start
            local_start = orig_start - unit_start
            local_end = local_start + mention_length

            new_row = row.copy()
            new_row.update({
                "text": unit_text, 
                "start": local_start, 
                "end": local_end, 
                "orig_start": orig_start, 
                "orig_end": orig_end
            })
            new_rows.append(new_row)
        
    
    return pd.DataFrame(new_rows)

def load_annotations(annotation_path, docs_root, labeled=True):
    """Load annotations from file."""
    data = []
    if not os.path.exists(annotation_path):
        return pd.DataFrame()
    
    num = 0
    
    with open(annotation_path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.strip().split("\t")
            
            doc_id, mention, start, end = parts[:4]

            labels = parts[4:] 
            
            start, end = int(start), int(end)

            text_path = os.path.join(docs_root, doc_id)
            
            with open(text_path, "r", encoding="utf-8", newline="") as doc_file:
                text = doc_file.read()

            if text[start:end] != mention:
                start, end = recaulculate_offsets(text, mention, start, end)
            
            entry = {"doc_id": doc_id, "text": text, "mention": mention, "start": start, "end": end}
            if labeled:
                entry["labels"] = labels
            data.append(entry)

    return pd.DataFrame(data)

def recaulculate_offsets(text, mention, orig_start, orig_end):
    """Recalculate offsets if original offsets do not match the mention text by gradually/iteratively searching nearby the original offsets and increasing the search radius."""
    search_radius = 0
    max_radius = min(orig_start, len(text) - orig_end + 1)
    while search_radius <= max_radius:
        start_search = max(0, orig_start - search_radius)
        end_search = min(len(text), orig_end + search_radius)
        idx = text.find(mention, start_search, end_search)
        if idx != -1:
            return idx, idx + len(mention)
        search_radius += 1
    return orig_start, orig_end
    

def load_taxonomy():
    with open(TAXONOMY_FILE, "r", encoding="utf-8", newline="") as f:
        taxonomy = json.load(f)
    
    return taxonomy

def load_coarse_labels():
    taxonomy = load_taxonomy()
    
    return [
        {k: v for k, v in label.items() if k != "subtypes"}
        for label in taxonomy
    ]


def load_fine_labels():
    taxonomy = load_taxonomy()

    return [
        {"parent": label['name'], **subtype}
        for label in taxonomy
        for subtype in label.get("subtypes", [])
    ]


def load_multilingual_data_by_mode(mode="document", labeled=True):
    assert mode in ["document", "paragraph", "sentence"], f"Invalid mode: {mode}"

    trainf_df_all, val_df_all, test_df_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for lang in available_languages:
        print(f"\n📘 Loading {mode}-level data for language: {lang}")

        train_root = os.path.join(TRAIN_DATA_PARENT, lang, "raw-documents")
        train_ann = os.path.join(TRAIN_DATA_PARENT, lang, "subtask-1-annotations.txt")

        val_root = os.path.join(VAL_DATA_PARENT, lang, "subtask-1-documents")
        val_ann = os.path.join(VAL_DATA_PARENT, lang, "subtask-1-annotations.txt")

        test_root = os.path.join(TEST_DATA_PARENT, lang, "subtask-1-documents")
        test_ann = os.path.join(TEST_DATA_PARENT, lang, "subtask-1-annotations.txt")

        if not (os.path.exists(train_ann) and os.path.exists(val_ann)):
            print(f"⚠️ Skipping {lang}: missing annotation files")
            continue

        # Load document-level data
        train_df_doc = load_annotations(train_ann, train_root, labeled=labeled)
        val_df_doc = load_annotations(val_ann, val_root, labeled=labeled)
        test_df_doc = load_annotations(test_ann, test_root, labeled=labeled)
        print(f"  - Loaded {len(train_df_doc)} train, {len(val_df_doc)} val, {len(test_df_doc)} test documents.")
        # Transform based on mode
        if mode == "document":
            train_df, val_df, test_df = train_df_doc, val_df_doc, test_df_doc
        else:
            train_df = expand_to_smaller_units(train_df_doc, mode)
            val_df = expand_to_smaller_units(val_df_doc, mode)
            test_df = expand_to_smaller_units(test_df_doc, mode) if not test_df_doc.empty else pd.DataFrame()

        print(f"  - Expanded to {len(train_df)} train, {len(val_df)} val, {len(test_df)} test {mode}s.")
       
        diff = set(zip(train_df_doc['doc_id'], train_df_doc['mention'])) - set(zip(train_df['doc_id'], train_df['mention']))
        print(diff)
        trainf_df_all = pd.concat([trainf_df_all, train_df], ignore_index=True)
        val_df_all = pd.concat([val_df_all, val_df], ignore_index=True)
        test_df_all = pd.concat([test_df_all, test_df], ignore_index=True)

        

    return trainf_df_all, val_df_all, test_df_all

def add_text_size_features(df):
    """Add text size metrics to dataframe (text length and word count)."""
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    return df

def load_data(mode="document"):
    train_df_full, val_df, test_df = load_multilingual_data_by_mode(mode=mode, labeled=True)

    print(f"\n✅ Loaded dataset sizes: Train={len(train_df_full)}, Val={len(val_df)}, Test={len(test_df)}")

    train_df, new_val_df = train_test_split(train_df_full, test_size=0.2, random_state=42)
    val_df = new_val_df

    # Add text size features
    train_df = add_text_size_features(train_df)
    val_df = add_text_size_features(val_df)
    test_df = add_text_size_features(test_df)

    print(f"\n✅ Final dataset sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"📊 Avg text length - Train: {train_df['text_length'].mean():.1f}, Val: {val_df['text_length'].mean():.1f}, Test: {test_df['text_length'].mean():.1f}")
    print(f"📊 Avg word count - Train: {train_df['word_count'].mean():.1f}, Val: {val_df['word_count'].mean():.1f}, Test: {test_df['word_count'].mean():.1f}")

    return train_df, val_df, test_df

# Load Labels immediately
# label2id, id2label = load_taxonomy()
available_languages = [d for d in os.listdir(TRAIN_DATA_PARENT) if os.path.isdir(os.path.join(TRAIN_DATA_PARENT, d))]

def insert_entity_tokens(df):
    doc_id, text, mention, start, end = df['doc_id'], df['text'], df['mention'], df['start'], df['end']
    modified_text = f"{text[:start]}{ENTITY_START_TOKEN}{text[start:end]}{ENTITY_END_TOKEN}{text[end:]}"
    df['text'] = modified_text
    return df

# K-Fold Generator: Implement a function to yield K-Fold splits (train/val/OOF indices).

def k_fold_split(df, k=5, random_state=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    for train_index, val_index in kf.split(df):
        yield df.iloc[train_index], df.iloc[val_index]