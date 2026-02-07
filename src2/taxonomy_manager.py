# Semantic Engine. Parses taxonomy.json and uses a Sentence-Transformer to generate 
# the parent and subtype anchor vectors.

import torch
import numpy as np
from data_utils import load_taxonomy, load_coarse_labels, load_fine_labels
from datasets import coarse_label2id, fine_label2id


class TaxonomyManager:
    def __init__(self, model, tokenizer, device='cpu'):
        self.taxonomy = load_taxonomy()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.anchors = self.generate_anchors()
         
        # Cache tensor anchors
        self._parent_anchors_cache = None
        self._subtype_anchors_cache = None

    def __process_item(self, item):
        name = item["name"]
        description = item.get("description", "None")
        example = item.get("example", "None")

        full_text = name
        # full_text = f"Role: {name}. Description: {description} Example: {example}".strip()


        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, padding=True)
        # Move inputs to the correct device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        # Get the [CLS] token representation as the embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
        return name, vector

    def generate_anchors(self) -> dict:
        anchors = {}
        
        for label in self.taxonomy:
            name, vector = self.__process_item(label)
            anchors[name] = vector
            
            for subtype in label.get("subtypes", []):
                subtype_name, subtype_vector = self.__process_item(subtype)
                anchors[subtype_name] = subtype_vector
        return anchors

    def get_parent_anchors(self):
        if self._parent_anchors_cache is not None:
            return self._parent_anchors_cache

        parent_labels = [label['name'] for label in load_coarse_labels()]
        self._parent_anchors_cache = {k: v for k, v in self.anchors.items() if k in parent_labels}
        return self._parent_anchors_cache

    def get_subtype_anchors(self, parent_label):
        if self._subtype_anchors_cache is not None:
            return self._subtype_anchors_cache

        subtype_labels = [label['name'] for label in load_fine_labels() if label['parent'] == parent_label]
        self._subtype_anchors_cache = {k: v for k, v in self.anchors.items() if k in subtype_labels}
        return self._subtype_anchors_cache
    
    def get_all_fine_anchors(self):
        """
        Returns all fine label anchors ordered by fine_label2id.
        Returns:
            dict: {fine_label_name: anchor_vector} for all 22 fine labels
        """
        fine_labels = load_fine_labels()
        fine_label_names = [label['name'] for label in fine_labels]
        return {k: v for k, v in self.anchors.items() if k in fine_label_names}
    
    def get_coarse_to_fine_mapping(self):
        """
        Returns a mapping from coarse label IDs to lists of valid fine label IDs.
        Returns:
            dict: {coarse_id: [fine_id1, fine_id2, ...]}
        """
        fine_labels = load_fine_labels()
        mapping = {}
        
        for coarse_name, coarse_id in coarse_label2id.items():
            valid_fine_ids = []
            for fine_label in fine_labels:
                if fine_label['parent'] == coarse_name:
                    fine_id = fine_label2id[fine_label['name']]
                    valid_fine_ids.append(fine_id)
            mapping[coarse_id] = valid_fine_ids
        
        return mapping
    
    def get_coarse_to_fine_mask(self, device='cpu'):
        """
        Returns a tensor mask where mask[coarse_id] is a boolean tensor indicating
        which fine labels are valid for that coarse category.
        
        Returns:
            torch.Tensor: Shape (num_coarse, num_fine) boolean mask
        """
        num_coarse = len(coarse_label2id)
        num_fine = len(fine_label2id)
        
        mask = torch.zeros(num_coarse, num_fine, dtype=torch.bool, device=device)
        mapping = self.get_coarse_to_fine_mapping()
        
        for coarse_id, fine_ids in mapping.items():
            for fine_id in fine_ids:
                mask[coarse_id, fine_id] = True
        
        return mask
    
    def get_fine_indices_for_coarse(self, coarse_label):
        """
        Get the fine label indices that belong to a specific coarse category.
        
        Args:
            coarse_label: Either the coarse label name (str) or coarse_id (int)
        Returns:
            list: List of fine label IDs valid for this coarse category
        """
        if isinstance(coarse_label, str):
            coarse_id = coarse_label2id[coarse_label]
        else:
            coarse_id = coarse_label
        
        mapping = self.get_coarse_to_fine_mapping()
        return mapping.get(coarse_id, [])
    
    def get_fine_label_names_for_coarse(self, coarse_label):
        """
        Get the fine label names that belong to a specific coarse category.
        
        Args:
            coarse_label: The coarse label name (str)
        Returns:
            list: List of fine label names valid for this coarse category
        """
        fine_labels = load_fine_labels()
        return [label['name'] for label in fine_labels if label['parent'] == coarse_label]
    

# from transformers import AutoModel, AutoTokenizer
# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# taxonomy_manager = TaxonomyManager(model, tokenizer)

# parent_anchors = taxonomy_manager.get_parent_anchors()
# subtype_anchors = taxonomy_manager.get_subtype_anchors("Antagonist")

# print("Subtype Anchors:", subtype_anchors)
