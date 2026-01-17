# Semantic Engine. Parses taxonomy.json and uses a Sentence-Transformer to generate 
# the parent and subtype anchor vectors.

import torch
import numpy as np
from data_utils import load_taxonomy, load_coarse_labels, load_fine_labels


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

    

# from transformers import AutoModel, AutoTokenizer
# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# taxonomy_manager = TaxonomyManager(model, tokenizer)

# parent_anchors = taxonomy_manager.get_parent_anchors()
# subtype_anchors = taxonomy_manager.get_subtype_anchors("Antagonist")

# print("Subtype Anchors:", subtype_anchors)
