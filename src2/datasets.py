# PyTorch datasets and collate functions for the two stages.
# All of them mark the entity span with [ENTITY] ... [/ENTITY] before tokenizing;
# the fine datasets additionally carry coarse labels (hard) or coarse probabilities (soft).

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

NUM_FINE_LABELS = len(fine_labels)


def collate_fn_entity_framing(batch):
    """Custom collate function to handle variable-length fine_labels."""
    collated = {}

    tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    collated['coarse_labels'] = torch.tensor([item['coarse_labels'] for item in batch])
    collated['fine_labels'] = [item['fine_labels'] for item in batch]  # variable length, kept as list
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

        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        start, end = row['start'], row['end']
        marked_text = f"{text[:start]} {ENTITY_START_TOKEN} {text[start:end]} {ENTITY_END_TOKEN} {text[end:]}"

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
        item['labels'] = labels
        return item

def collate_fn_fine_role(batch):
    """
    Collate function for FineRoleDataset.
    coarse_labels -> (B,); fine_labels -> multi-hot (B, num_fine_labels).
    """
    collated = {}

    tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    collated['coarse_labels'] = torch.tensor([item['coarse_labels'] for item in batch])
    collated['fine_labels'] = torch.stack([item['fine_labels'] for item in batch])

    if 'original_labels' in batch[0]:
        collated['original_labels'] = [item['original_labels'] for item in batch]

    return collated


class FineRoleDataset(Dataset):
    """
    Dataset for fine-role multi-label classification.
    
    Features:
    - Converts fine labels to multi-hot encoding (tensor of shape num_fine_labels)
    - Includes coarse labels for masking during training
    - Can use either ground-truth coarse labels (training) or predicted coarse labels (inference)
    """
    
    def __init__(self, dataframe, tokenizer, max_length=512, use_predicted_coarse=False):
        """
        Args:
            dataframe: DataFrame with columns: text, start, end, labels, 
                       and optionally 'predicted_coarse' for inference
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            use_predicted_coarse: If True, use 'predicted_coarse' column instead of ground-truth
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_predicted_coarse = use_predicted_coarse
        self.num_fine_labels = NUM_FINE_LABELS
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        labels = row['labels']

        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        start, end = int(row['start']), int(row['end'])
        marked_text = f"{text[:start]} {ENTITY_START_TOKEN} {text[start:end]} {ENTITY_END_TOKEN} {text[end:]}"

        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Predicted coarse at inference, ground-truth coarse during training
        if self.use_predicted_coarse and 'predicted_coarse' in row:
            predicted_coarse = row['predicted_coarse']
            if isinstance(predicted_coarse, str):
                item['coarse_labels'] = coarse_label2id[predicted_coarse]
            else:
                item['coarse_labels'] = int(predicted_coarse)
        else:
            item['coarse_labels'] = coarse_label2id[labels[0]]

        fine_label_tensor = torch.zeros(self.num_fine_labels, dtype=torch.float32)
        for label in labels[1:]:  # labels[0] is the coarse role
            if label in fine_label2id:
                fine_label_tensor[fine_label2id[label]] = 1.0

        item['fine_labels'] = fine_label_tensor
        item['original_labels'] = labels

        return item


class FineRoleInferenceDataset(Dataset):
    """
    Dataset for fine-role inference when only predicted coarse labels are available.
    
    Similar to FineRoleDataset but designed for inference scenarios where
    ground-truth fine labels are not available.
    """
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        """
        Args:
            dataframe: DataFrame with columns: text, start, end, predicted_coarse
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_fine_labels = NUM_FINE_LABELS
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']

        start, end = int(row['start']), int(row['end'])
        marked_text = f"{text[:start]} {ENTITY_START_TOKEN} {text[start:end]} {ENTITY_END_TOKEN} {text[end:]}"

        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        predicted_coarse = row['predicted_coarse']
        if isinstance(predicted_coarse, str):
            item['coarse_labels'] = coarse_label2id[predicted_coarse]
        else:
            item['coarse_labels'] = int(predicted_coarse)

        return item


def collate_fn_fine_inference(batch):
    """Collate function for fine-role inference (no fine_labels)."""
    collated = {}
    
    tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    
    collated['coarse_labels'] = torch.tensor([item['coarse_labels'] for item in batch])
    
    return collated


# Soft-conditioned datasets: carry coarse probabilities instead of a hard coarse label.

NUM_COARSE_LABELS = len(coarse_labels)


class SoftConditionedFineRoleDataset(Dataset):
    """
    Dataset for fine-role classification with soft coarse conditioning.
    
    Instead of using hard coarse labels for masking, this dataset provides
    coarse probabilities (from a pre-trained coarse classifier) that are
    passed to the fine classifier for soft conditioning.
    
    Features:
    - Converts fine labels to multi-hot encoding
    - Includes coarse probabilities for soft hierarchy conditioning
    - Falls back to one-hot coarse labels if probabilities not available
    """
    
    def __init__(self, dataframe, tokenizer, max_length=512, use_predicted_coarse=False):
        """
        Args:
            dataframe: DataFrame with columns: text, start, end, labels,
                       and optionally 'coarse_probs' (list/array of 3 probabilities)
                       or 'coarse_prob_protagonist', 'coarse_prob_antagonist', 'coarse_prob_innocent'
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            use_predicted_coarse: If True, use predicted coarse info instead of ground-truth
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_predicted_coarse = use_predicted_coarse
        self.num_fine_labels = NUM_FINE_LABELS
        self.num_coarse_labels = NUM_COARSE_LABELS

        self.has_coarse_probs = self._check_coarse_probs_available()

    def _check_coarse_probs_available(self):
        """True if the dataframe carries coarse probabilities (single list column or per-class columns)."""
        if 'coarse_probs' in self.dataframe.columns:
            return True
        prob_cols = ['coarse_prob_protagonist', 'coarse_prob_antagonist', 'coarse_prob_innocent']
        return all(col in self.dataframe.columns for col in prob_cols)
    
    def _get_coarse_probs(self, row):
        """Extract coarse probabilities from the row (list column or per-class columns)."""
        if 'coarse_probs' in row and row['coarse_probs'] is not None:
            probs = row['coarse_probs']
            if isinstance(probs, str):
                probs = ast.literal_eval(probs)
            return torch.tensor(probs, dtype=torch.float32)

        if all(col in row.index for col in ['coarse_prob_protagonist', 'coarse_prob_antagonist', 'coarse_prob_innocent']):
            return torch.tensor([
                row['coarse_prob_protagonist'],
                row['coarse_prob_antagonist'],
                row['coarse_prob_innocent']
            ], dtype=torch.float32)

        # Alternate column naming written by train_coarse.py
        if all(col in row.index for col in ['coarse_proba_protagonist', 'coarse_proba_antagonist', 'coarse_proba_innocent']):
            return torch.tensor([
                row['coarse_proba_protagonist'],
                row['coarse_proba_antagonist'],
                row['coarse_proba_innocent']
            ], dtype=torch.float32)

        return None
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']
        labels = row['labels']

        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        start, end = int(row['start']), int(row['end'])
        marked_text = f"{text[:start]} {ENTITY_START_TOKEN} {text[start:end]} {ENTITY_END_TOKEN} {text[end:]}"

        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.use_predicted_coarse and 'predicted_coarse' in row:
            predicted_coarse = row['predicted_coarse']
            if isinstance(predicted_coarse, str):
                coarse_label_id = coarse_label2id[predicted_coarse]
            else:
                coarse_label_id = int(predicted_coarse)
        else:
            coarse_label_id = coarse_label2id[labels[0]]

        item['coarse_labels'] = coarse_label_id

        # Soft conditioning signal; fall back to a one-hot of the coarse label
        coarse_probs = self._get_coarse_probs(row)
        if coarse_probs is not None:
            item['coarse_probs'] = coarse_probs
        else:
            one_hot = torch.zeros(self.num_coarse_labels, dtype=torch.float32)
            one_hot[coarse_label_id] = 1.0
            item['coarse_probs'] = one_hot

        fine_label_tensor = torch.zeros(self.num_fine_labels, dtype=torch.float32)
        for label in labels[1:]:
            if label in fine_label2id:
                fine_label_tensor[fine_label2id[label]] = 1.0

        item['fine_labels'] = fine_label_tensor
        item['original_labels'] = labels

        return item


class SoftConditionedInferenceDataset(Dataset):
    """
    Dataset for soft-conditioned fine-role inference.
    
    Uses coarse probabilities from a pre-trained coarse classifier
    for soft hierarchy conditioning during inference.
    """
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        """
        Args:
            dataframe: DataFrame with columns: text, start, end, predicted_coarse,
                       and coarse probability columns
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_fine_labels = NUM_FINE_LABELS
        self.num_coarse_labels = NUM_COARSE_LABELS
    
    def _get_coarse_probs(self, row):
        """Extract coarse probabilities from the row (list column or per-class columns)."""
        if 'coarse_probs' in row and row['coarse_probs'] is not None:
            probs = row['coarse_probs']
            if isinstance(probs, str):
                probs = ast.literal_eval(probs)
            return torch.tensor(probs, dtype=torch.float32)

        if all(col in row.index for col in ['coarse_prob_protagonist', 'coarse_prob_antagonist', 'coarse_prob_innocent']):
            return torch.tensor([
                row['coarse_prob_protagonist'],
                row['coarse_prob_antagonist'],
                row['coarse_prob_innocent']
            ], dtype=torch.float32)

        # Alternate column naming written by train_coarse.py
        if all(col in row.index for col in ['coarse_proba_protagonist', 'coarse_proba_antagonist', 'coarse_proba_innocent']):
            return torch.tensor([
                row['coarse_proba_protagonist'],
                row['coarse_proba_antagonist'],
                row['coarse_proba_innocent']
            ], dtype=torch.float32)

        return None
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['text']

        start, end = int(row['start']), int(row['end'])
        marked_text = f"{text[:start]} {ENTITY_START_TOKEN} {text[start:end]} {ENTITY_END_TOKEN} {text[end:]}"

        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        predicted_coarse = row['predicted_coarse']
        if isinstance(predicted_coarse, str):
            coarse_label_id = coarse_label2id[predicted_coarse]
        else:
            coarse_label_id = int(predicted_coarse)

        item['coarse_labels'] = coarse_label_id

        coarse_probs = self._get_coarse_probs(row)
        if coarse_probs is not None:
            item['coarse_probs'] = coarse_probs
        else:
            one_hot = torch.zeros(self.num_coarse_labels, dtype=torch.float32)
            one_hot[coarse_label_id] = 1.0
            item['coarse_probs'] = one_hot

        return item


def collate_fn_soft_conditioned(batch):
    """
    Collate function for soft-conditioned datasets:
    coarse_labels (B,), coarse_probs (B, num_coarse), fine_labels multi-hot if present.
    """
    collated = {}

    tensor_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    collated['coarse_labels'] = torch.tensor([item['coarse_labels'] for item in batch])
    collated['coarse_probs'] = torch.stack([item['coarse_probs'] for item in batch])

    if 'fine_labels' in batch[0]:
        collated['fine_labels'] = torch.stack([item['fine_labels'] for item in batch])

    if 'original_labels' in batch[0]:
        collated['original_labels'] = [item['original_labels'] for item in batch]

    return collated
