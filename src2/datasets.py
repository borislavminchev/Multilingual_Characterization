# 1. EntityFramingDataset: Takes a DataFrame and tokenizer. 
# Crucially, applies the entity markers ([ENTITY], [/ENTITY]) to the text. 
# 2. compute_metrics: Defines the evaluation function used by the Hugging Face Trainer (e.g., micro-F1).

import torch
import ast
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN, load_data, load_coarse_labels, load_fine_labels

coarse_labels = [label["name"] for label in load_coarse_labels()]
fine_labels = [label["name"] for label in load_fine_labels()]

coarse_label2id = {label: i for i, label in enumerate(coarse_labels)}
fine_label2id = {label: i for i, label in enumerate(fine_labels)}

coarse_id2label = {i: label for i, label in enumerate(coarse_labels)}
fine_id2label = {i: label for i, label in enumerate(fine_labels)}


def collate_fn_entity_framing(batch):
    """Custom collate function to handle variable-length fine_labels."""
    collated = {}
    
    # Collate standard tensors
    tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    
    # Collate coarse_labels as a tensor and use as 'labels' for the Trainer
    collated['coarse_labels'] = torch.tensor([item['coarse_labels'] for item in batch])
    
    # Keep fine_labels as list (variable length)
    collated['fine_labels'] = [item['fine_labels'] for item in batch]
    
    # Keep original labels for reference only (not used by Trainer)
    collated['original_labels'] = [item['labels'] for item in batch]
    
    return collated


class EntityFramingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        labels = row['labels']
        
        # Parse labels if they come as a string representation of a list
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        # Insert entity markers around the mention
        start, end = row['start'], row['end']
        marked_text = f"{text[:start]} {ENTITY_START_TOKEN} {text[start:end]} {ENTITY_END_TOKEN} {text[end:]}"

        # Tokenize the text with entity markers
        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        item['coarse_labels'] = coarse_label2id[labels[0]]
        item['fine_labels'] = [fine_label2id[label] for label in labels[1:]]
        item['labels'] = labels  # Original labels for reference
        return item
    
# train_df, val_df, test_df = load_data(mode="document")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer.add_special_tokens({'additional_special_tokens': [ENTITY_START_TOKEN, ENTITY_END_TOKEN]})


# train_dataset = EntityFramingDataset(train_df, tokenizer)
# print("train_dataset[0]:", train_dataset[0])

# print("Fine Labels:", load_fine_labels())
# print("Coarse Labels:", load_coarse_labels())
