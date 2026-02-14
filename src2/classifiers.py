"""
Classifier Models for Hierarchical Entity Role Classification

This module contains:
- CoarseRoleClassifier: Single-label classification (3 classes)
- FineRoleClassifier: Multi-label classification with hard masking (22 labels)
- SoftConditionedFineClassifier: Multi-label with soft hierarchy conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from taxonomy_manager import TaxonomyManager
from datasets import coarse_label2id, fine_label2id, coarse_id2label, fine_id2label
from transformers.modeling_outputs import SequenceClassifierOutput

from losses import (
    FocalLoss, MultiLabelFocalLoss, 
    AsymmetricLoss, AsymmetricLossOptimized,
    CardinalityRegularizer
)

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def entity_span_pooling(hidden_states, input_ids, entity_start_id, entity_end_id):
    """
    Pool entity representations from hidden states.
    
    Args:
        hidden_states: (B, T, H) transformer hidden states
        input_ids: (B, T) input token IDs
        entity_start_id: Token ID for [ENTITY] marker
        entity_end_id: Token ID for [/ENTITY] marker
    Returns:
        (B, H) pooled entity representations
    """
    batch_size, seq_len, hidden_dim = hidden_states.size()
    pooled_outputs = []

    for i in range(batch_size):
        ids = input_ids[i]

        start_positions = (ids == entity_start_id).nonzero(as_tuple=True)[0]
        end_positions = (ids == entity_end_id).nonzero(as_tuple=True)[0]

        # Fallback: no entity markers found
        if len(start_positions) == 0 or len(end_positions) == 0:
            pooled_outputs.append(hidden_states[i, 0])  # [CLS]
            continue

        start = start_positions[0].item() + 1
        end = end_positions[0].item()

        if start >= end:
            pooled_outputs.append(hidden_states[i, 0])
            continue

        span_embeddings = hidden_states[i, start:end]
        pooled_outputs.append(span_embeddings.mean(dim=0))

    return torch.stack(pooled_outputs)


def multi_view_entity_pooling(hidden_states, input_ids, entity_start_id, entity_end_id):
    """
    Multi-view entity representation pooling.
    
    Combines multiple representations of the entity for richer features:
    1. Entity span mean pooling (semantic content of entity)
    2. Entity start marker hidden state (left context boundary)
    3. Entity end marker hidden state (right context boundary)
    4. Entity span max pooling (salient features)
    5. CLS token (global document context)
    
    Args:
        hidden_states: (B, T, H) transformer hidden states
        input_ids: (B, T) input token IDs
        entity_start_id: Token ID for [ENTITY] marker
        entity_end_id: Token ID for [/ENTITY] marker
    Returns:
        (B, 5*H) concatenated multi-view entity representations
    """
    batch_size, seq_len, hidden_dim = hidden_states.size()
    
    entity_span_means = []
    entity_start_hiddens = []
    entity_end_hiddens = []
    entity_span_maxs = []
    cls_tokens = []

    for i in range(batch_size):
        ids = input_ids[i]
        cls_token = hidden_states[i, 0]  # [CLS] token
        cls_tokens.append(cls_token)

        start_positions = (ids == entity_start_id).nonzero(as_tuple=True)[0]
        end_positions = (ids == entity_end_id).nonzero(as_tuple=True)[0]

        # Fallback: no entity markers found - use CLS for all views
        if len(start_positions) == 0 or len(end_positions) == 0:
            entity_span_means.append(cls_token)
            entity_start_hiddens.append(cls_token)
            entity_end_hiddens.append(cls_token)
            entity_span_maxs.append(cls_token)
            continue

        start_marker_pos = start_positions[0].item()
        end_marker_pos = end_positions[0].item()
        
        # Entity span is between the markers (exclusive of markers)
        entity_start = start_marker_pos + 1
        entity_end = end_marker_pos

        # Get marker hidden states
        entity_start_hiddens.append(hidden_states[i, start_marker_pos])
        entity_end_hiddens.append(hidden_states[i, end_marker_pos])

        if entity_start >= entity_end:
            # Empty entity span - use marker states
            entity_span_means.append(hidden_states[i, start_marker_pos])
            entity_span_maxs.append(hidden_states[i, start_marker_pos])
            continue

        span_embeddings = hidden_states[i, entity_start:entity_end]
        
        # Mean pooling
        entity_span_means.append(span_embeddings.mean(dim=0))
        
        # Max pooling
        entity_span_maxs.append(span_embeddings.max(dim=0)[0])

    # Stack all views
    entity_span_mean = torch.stack(entity_span_means)      # (B, H)
    entity_start_hidden = torch.stack(entity_start_hiddens)  # (B, H)
    entity_end_hidden = torch.stack(entity_end_hiddens)      # (B, H)
    entity_span_max = torch.stack(entity_span_maxs)          # (B, H)
    cls_token = torch.stack(cls_tokens)                      # (B, H)

    # Concatenate all views: (B, 5*H)
    multi_view = torch.cat([
        entity_span_mean,
        entity_start_hidden,
        entity_end_hidden,
        entity_span_max,
        cls_token
    ], dim=1)

    return multi_view


class MultiViewFusion(nn.Module):
    """
    Learnable fusion layer for multi-view entity representations.
    
    Takes concatenated multi-view representations and fuses them into
    a single representation using attention-based or MLP-based fusion.
    """
    
    def __init__(self, hidden_size, num_views=5, fusion_type='attention', dropout=0.1):
        """
        Args:
            hidden_size: Hidden dimension of each view
            num_views: Number of views being fused (default 5)
            fusion_type: 'attention', 'mlp', or 'gated'
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_views = num_views
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # Self-attention over views
            self.view_attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
            )
        elif fusion_type == 'mlp':
            # MLP fusion
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_size * num_views, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
            )
        elif fusion_type == 'gated':
            # Gated fusion with learnable gates per view
            self.gates = nn.Sequential(
                nn.Linear(hidden_size * num_views, num_views),
                nn.Sigmoid(),
            )
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        logger.info(f"Initialized MultiViewFusion with {fusion_type} fusion, {num_views} views")
    
    def forward(self, multi_view_repr):
        """
        Args:
            multi_view_repr: (B, num_views * hidden_size) concatenated representations
        Returns:
            (B, hidden_size) fused representation
        """
        batch_size = multi_view_repr.size(0)
        
        if self.fusion_type == 'attention':
            # Reshape to (B, num_views, hidden_size)
            views = multi_view_repr.view(batch_size, self.num_views, self.hidden_size)
            
            # Compute attention weights (B, num_views, 1)
            attn_scores = self.view_attention(views)
            attn_weights = F.softmax(attn_scores, dim=1)
            
            # Weighted sum (B, hidden_size)
            fused = (views * attn_weights).sum(dim=1)
            return self.output_proj(fused)
        
        elif self.fusion_type == 'mlp':
            return self.fusion_mlp(multi_view_repr)
        
        elif self.fusion_type == 'gated':
            # Reshape to (B, num_views, hidden_size)
            views = multi_view_repr.view(batch_size, self.num_views, self.hidden_size)
            
            # Compute gates (B, num_views)
            gates = self.gates(multi_view_repr)
            
            # Apply gates and sum (B, hidden_size)
            fused = (views * gates.unsqueeze(-1)).sum(dim=1)
            return self.output_proj(fused)


# =============================================================================
# SEMANTIC SIMILARITY HEADS
# =============================================================================

class SemanticSimilarityHead(nn.Module):
    """Semantic similarity head for coarse-level classification."""
    
    def __init__(self, model, tokenizer, input_dim, parent_label=None, 
                 freeze_anchors=True, similarity_metric='cosine', device='cpu'):
        super().__init__()
        
        self.taxonomy_manager = TaxonomyManager(model, tokenizer, device=device)
        self.parent_label = parent_label
        self.input_dim = input_dim
        self.similarity_metric = similarity_metric
        
        anchor_vectors_dict = (
            self.taxonomy_manager.get_parent_anchors() 
            if parent_label is None 
            else self.taxonomy_manager.get_subtype_anchors(parent_label)
        )
        
        label2id = coarse_label2id if parent_label is None else fine_label2id
        anchor_vectors_list = [
            anchor_vectors_dict[label] 
            for label in sorted(label2id.keys(), key=lambda x: label2id[x])
        ]
        self.num_classes = len(anchor_vectors_list)

        if freeze_anchors:
            self.register_buffer('anchor_vectors', torch.tensor(anchor_vectors_list).detach())
        else:
            self.anchor_vectors = nn.Parameter(torch.tensor(anchor_vectors_list), requires_grad=True)
        
        self.projection = nn.Linear(input_dim, input_dim)
        self.temperature = 15.0
        
        logger.info(f"Initialized SemanticSimilarityHead with {self.num_classes} classes")

    def forward(self, entity_vectors):
        projected = self.projection(entity_vectors)
        
        if self.similarity_metric == 'cosine':
            projected_norm = F.normalize(projected, p=2, dim=1)
            anchors_norm = F.normalize(self.anchor_vectors, p=2, dim=1)
            logits = torch.matmul(projected_norm, anchors_norm.t())
            logits = logits * self.temperature
        else:
            logits = torch.matmul(projected, self.anchor_vectors.t())
        
        return logits


class FineSemanticSimilarityHead(nn.Module):
    """Semantic similarity head for fine-grained multi-label classification."""
    
    def __init__(self, model, tokenizer, input_dim, freeze_anchors=True, 
                 similarity_metric='cosine', device='cpu'):
        super().__init__()
        
        self.taxonomy_manager = TaxonomyManager(model, tokenizer, device=device)
        self.input_dim = input_dim
        self.similarity_metric = similarity_metric
        self.device = device
        
        anchor_vectors_dict = self.taxonomy_manager.get_all_fine_anchors()
        anchor_vectors_list = [
            anchor_vectors_dict[label] 
            for label in sorted(fine_label2id.keys(), key=lambda x: fine_label2id[x])
        ]
        self.num_classes = len(anchor_vectors_list)
        
        if freeze_anchors:
            self.register_buffer('anchor_vectors', torch.tensor(anchor_vectors_list).detach())
        else:
            self.anchor_vectors = nn.Parameter(torch.tensor(anchor_vectors_list), requires_grad=True)
        
        self.projection = nn.Linear(input_dim, input_dim)
        self.temperature = 15.0
        
        self.register_buffer(
            'coarse_to_fine_mask', 
            self.taxonomy_manager.get_coarse_to_fine_mask(device=device)
        )
        
        logger.info(f"Initialized FineSemanticSimilarityHead with {self.num_classes} fine classes")
    
    def forward(self, entity_vectors, coarse_labels=None):
        projected = self.projection(entity_vectors)
        
        if self.similarity_metric == 'cosine':
            projected_norm = F.normalize(projected, p=2, dim=1)
            anchors_norm = F.normalize(self.anchor_vectors, p=2, dim=1)
            logits = torch.matmul(projected_norm, anchors_norm.t())
            logits = logits * self.temperature
        else:
            logits = torch.matmul(projected, self.anchor_vectors.t())
        
        if coarse_labels is not None:
            mask = self.coarse_to_fine_mask[coarse_labels]
        else:
            mask = torch.ones(
                entity_vectors.size(0), self.num_classes, 
                dtype=torch.bool, device=entity_vectors.device
            )
        
        return logits, mask


# =============================================================================
# HIERARCHY AFFINITY LAYER (FOR SOFT CONDITIONING)
# =============================================================================

class HierarchyAffinityLayer(nn.Module):
    """
    Learnable affinity layer that maps coarse probabilities to fine label priors.
    
    Instead of hard masking, this layer learns soft relationships between
    coarse categories and fine labels.
    """
    
    def __init__(self, num_coarse=3, num_fine=22, init_from_taxonomy=True, device='cpu'):
        super().__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.device = device
        
        self.affinity = nn.Parameter(torch.zeros(num_coarse, num_fine, device=device))
        
        if init_from_taxonomy:
            self._init_from_taxonomy()
        else:
            nn.init.xavier_uniform_(self.affinity)
        
        logger.info(f"Initialized HierarchyAffinityLayer: ({num_coarse}, {num_fine})")
    
    def _init_from_taxonomy(self):
        """Initialize affinity matrix based on known taxonomy hierarchy."""
        init_valid = 2.0
        init_invalid = -2.0
        
        self.affinity.data.fill_(init_invalid)
        
        # Protagonist subtypes
        protagonist_fine = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous']
        for fine_name in protagonist_fine:
            if fine_name in fine_label2id:
                self.affinity.data[0, fine_label2id[fine_name]] = init_valid
        
        # Antagonist subtypes
        antagonist_fine = ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor',
                          'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot']
        for fine_name in antagonist_fine:
            if fine_name in fine_label2id:
                self.affinity.data[1, fine_label2id[fine_name]] = init_valid
        
        # Innocent subtypes
        innocent_fine = ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
        for fine_name in innocent_fine:
            if fine_name in fine_label2id:
                self.affinity.data[2, fine_label2id[fine_name]] = init_valid
        
        logger.info("Initialized affinity matrix from taxonomy hierarchy")
    
    def forward(self, coarse_probs):
        """
        Compute soft fine label prior from coarse probabilities.
        
        Args:
            coarse_probs: (B, num_coarse) softmax probabilities for coarse labels
        Returns:
            fine_prior: (B, num_fine) soft prior over fine labels
        """
        fine_prior = torch.matmul(coarse_probs, self.affinity)
        return fine_prior


# =============================================================================
# CLASSIFIER MODELS
# =============================================================================

class MultiViewCoarseClassifier(nn.Module):
    """
    Multi-View Coarse Role Classifier for single-label classification (3 classes).
    
    Uses multiple views of the entity representation for richer features:
    1. Entity span mean pooling (semantic content)
    2. Entity start marker hidden state (left context boundary)
    3. Entity end marker hidden state (right context boundary)
    4. Entity span max pooling (salient features)
    5. CLS token (global document context)
    
    These views are fused using attention, MLP, or gated fusion.
    """
    
    def __init__(self, base_model, tokenizer, device='cpu', num_unfrozen_layers=4, 
                 class_counts=None, label_smoothing=0.0, focal_gamma=2.0, beta=0.9999,
                 fusion_type='attention', dropout=0.1):
        """
        Args:
            base_model: Pre-trained transformer model
            tokenizer: Tokenizer with entity markers
            device: Device to run on
            num_unfrozen_layers: Number of transformer layers to fine-tune (from top)
            class_counts: List of sample counts per class for class-balanced loss
            label_smoothing: Label smoothing factor (0 = no smoothing)
            focal_gamma: Focusing parameter for focal loss
            beta: Class balance beta for effective number of samples
            fusion_type: How to fuse multi-view representations ('attention', 'mlp', 'gated')
            dropout: Dropout probability
        """
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.num_classes = 3  # Protagonist, Antagonist, Innocent
        self.num_views = 5    # Number of entity views
        
        # Freeze encoder layers (keep last num_unfrozen_layers trainable)
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        logger.info(f"Frozen {total_layers - num_unfrozen_layers}/{total_layers} encoder layers")
        
        hidden_size = base_model.config.hidden_size
        
        # Multi-view fusion layer
        self.fusion = MultiViewFusion(
            hidden_size=hidden_size,
            num_views=self.num_views,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # Classification head after fusion
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_classes)
        )
        
        # Loss function with optional label smoothing
        if label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.use_focal = False
            logger.info(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
        else:
            self.loss_fn = FocalLoss(
                samples_per_cls=class_counts, 
                beta=beta, 
                gamma=focal_gamma, 
                reduction='mean',  # Use mean for better stability
                device=device
            )
            self.use_focal = True
            logger.info(f"Using FocalLoss with gamma={focal_gamma}, beta={beta}")
        
        logger.info(f"Initialized MultiViewCoarseClassifier with {fusion_type} fusion")
        self.to(device)

    def forward(self, input_ids, attention_mask, coarse_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get multi-view entity representation (B, 5*H)
        multi_view_repr = multi_view_entity_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )
        
        # Fuse views into single representation (B, H)
        fused_repr = self.fusion(multi_view_repr)
        
        # Classify
        logits = self.classifier(fused_repr)
        
        # Compute loss
        loss = None
        if coarse_labels is not None:
            loss = self.loss_fn(logits, coarse_labels)
        
        return SequenceClassifierOutput(loss=loss, logits=logits)


class CoarseRoleClassifier(nn.Module):
    """
    Coarse Role Classifier for single-label classification (3 classes).
    
    Supports two modes:
    1. use_cls_head=True: Standard classification head (simpler, often better)
    2. use_cls_head=False: Semantic similarity with coarse label anchors (legacy)
    
    NOTE: For better performance, consider using MultiViewCoarseClassifier instead.
    """
    
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine', 
                 device='cpu', num_unfrozen_layers=2, class_counts=None, 
                 use_cls_head=True, label_smoothing=0.0, focal_gamma=2.0, beta=0.9999):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cls_head = use_cls_head
        self.num_classes = 3  # Protagonist, Antagonist, Innocent
        
        # Freeze encoder layers
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        hidden_size = base_model.config.hidden_size
        
        if use_cls_head:
            # Standard classification head (simpler and often more effective)
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, self.num_classes)
            )
            logger.info("Using standard classification head for coarse classifier")
        else:
            # Semantic similarity head (legacy)
            self.semantic_head = SemanticSimilarityHead(
                model=base_model,
                tokenizer=tokenizer,
                input_dim=hidden_size,
                freeze_anchors=freeze_anchors,
                similarity_metric=similarity_metric,
                device=device
            )
            logger.info("Using semantic similarity head for coarse classifier")
        
        # Loss function with optional label smoothing
        if label_smoothing > 0:
            # Use CrossEntropyLoss with label smoothing (simpler, often better)
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.use_focal = False
            logger.info(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
        else:
            # Use FocalLoss for class imbalance
            self.loss_fn = FocalLoss(
                samples_per_cls=class_counts, 
                beta=beta, 
                gamma=focal_gamma, 
                reduction='sum', 
                device=device
            )
            self.use_focal = True
            logger.info(f"Using FocalLoss with gamma={focal_gamma}, beta={beta}")
        
        self.to(device)

    def forward(self, input_ids, attention_mask, coarse_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get entity representation
        entity_vectors = entity_span_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )
        
        # Compute logits
        if self.use_cls_head:
            logits = self.classifier(entity_vectors)
        else:
            logits = self.semantic_head(entity_vectors)
        
        # Compute loss
        loss = None
        if coarse_labels is not None:
            loss = self.loss_fn(logits, coarse_labels)
        
        return SequenceClassifierOutput(loss=loss, logits=logits)


class FineRoleClassifier(nn.Module):
    """
    Fine Role Classifier for multi-label classification with hard masking.
    
    Uses semantic similarity with fine label anchors and masks invalid
    labels based on coarse predictions.
    """
    
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine',
                 device='cpu', num_unfrozen_layers=2, class_counts=None, threshold=0.5,
                 loss_type='asl_optimized', gamma_neg=4.0, gamma_pos=1.0, clip=0.05,
                 entropy_weight=0.1):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.loss_type = loss_type
        
        # Freeze encoder layers
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.semantic_head = FineSemanticSimilarityHead(
            model=base_model,
            tokenizer=tokenizer,
            input_dim=base_model.config.hidden_size,
            freeze_anchors=freeze_anchors,
            similarity_metric=similarity_metric,
            device=device
        )
        
        # Select loss function
        if loss_type == 'focal':
            self.loss_fn = MultiLabelFocalLoss(
                samples_per_cls=class_counts, beta=0.9999, gamma=2.0,
                reduction='mean', device=device
            )
        elif loss_type == 'asl':
            self.loss_fn = AsymmetricLoss(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, reduction='mean'
            )
        elif loss_type == 'asl_optimized':
            self.loss_fn = AsymmetricLossOptimized(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip,
                samples_per_cls=class_counts, beta=0.9999, reduction='mean',
                device=device, entropy_weight=entropy_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        logger.info(f"Initialized FineRoleClassifier with {loss_type} loss")
        self.to(device)
    
    def forward(self, input_ids, attention_mask, coarse_labels=None, fine_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        entity_vectors = entity_span_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )
        
        logits, mask = self.semantic_head(entity_vectors, coarse_labels)
        
        # Apply mask for inference
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        
        loss = None
        if fine_labels is not None:
            loss = self.loss_fn(logits, fine_labels, mask=mask)
        
        return SequenceClassifierOutput(loss=loss, logits=masked_logits)
    
    def predict(self, input_ids, attention_mask, coarse_labels):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, coarse_labels=coarse_labels)
            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities >= self.threshold).float()
        return predictions, probabilities


class SoftConditionedFineClassifier(nn.Module):
    """
    Soft Conditioned Fine Role Classifier combining:
    1. Soft hierarchy conditioning (coarse probs → fine prior via learnable affinity)
    2. Cardinality regularization (constrain number of predictions)
    3. Asymmetric Loss (handle class imbalance)
    
    Key differences from FineRoleClassifier:
    - Uses coarse probabilities (soft) instead of coarse labels (hard)
    - Learns hierarchy relationship rather than hard masking
    - Adds cardinality constraint to prevent over/under-prediction
    """
    
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine',
                 device='cpu', num_unfrozen_layers=2, class_counts=None, threshold=0.5,
                 loss_type='asl_optimized', gamma_neg=4.0, gamma_pos=1.0, clip=0.05,
                 entropy_weight=0.1, cardinality_weight=0.3, target_cardinality=1.5,
                 num_coarse=3, num_fine=22):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.loss_type = loss_type
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        
        # Freeze encoder layers
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Entity to fine logits projection
        hidden_size = base_model.config.hidden_size
        self.entity_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_fine)
        )
        
        # Hierarchy affinity layer
        self.hierarchy_affinity = HierarchyAffinityLayer(
            num_coarse=num_coarse,
            num_fine=num_fine,
            init_from_taxonomy=True,
            device=device
        )
        
        # Cardinality regularizer
        self.cardinality_reg = CardinalityRegularizer(
            target_cardinality=target_cardinality,
            weight=cardinality_weight
        )
        
        # Base loss function
        if loss_type == 'focal':
            self.loss_fn = MultiLabelFocalLoss(
                samples_per_cls=class_counts, beta=0.9999, gamma=2.0,
                reduction='mean', device=device
            )
        elif loss_type == 'asl':
            self.loss_fn = AsymmetricLoss(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, reduction='mean'
            )
        elif loss_type == 'asl_optimized':
            self.loss_fn = AsymmetricLossOptimized(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip,
                samples_per_cls=class_counts, beta=0.9999, reduction='mean',
                device=device, entropy_weight=entropy_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        logger.info(f"Initialized SoftConditionedFineClassifier: "
                   f"cardinality_weight={cardinality_weight}, target={target_cardinality}")
        
        self.to(device)
    
    def forward(self, input_ids, attention_mask, coarse_probs=None, coarse_labels=None, 
                fine_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        entity_vectors = entity_span_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )
        
        entity_logits = self.entity_projection(entity_vectors)
        
        # Compute hierarchy prior
        if coarse_probs is not None:
            hierarchy_prior = self.hierarchy_affinity(coarse_probs)
        elif coarse_labels is not None:
            coarse_one_hot = F.one_hot(coarse_labels, num_classes=self.num_coarse).float()
            hierarchy_prior = self.hierarchy_affinity(coarse_one_hot)
        else:
            batch_size = entity_vectors.size(0)
            hierarchy_prior = torch.zeros(batch_size, self.num_fine, device=self.device)
        
        fine_logits = entity_logits + hierarchy_prior
        
        loss = None
        if fine_labels is not None:
            base_loss = self.loss_fn(fine_logits, fine_labels, mask=None)
            probs = torch.sigmoid(fine_logits)
            cardinality_loss = self.cardinality_reg(probs, mask=None)
            loss = base_loss + cardinality_loss
        
        return SequenceClassifierOutput(loss=loss, logits=fine_logits)
    
    def predict(self, input_ids, attention_mask, coarse_probs=None, coarse_labels=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, attention_mask, 
                coarse_probs=coarse_probs, 
                coarse_labels=coarse_labels
            )
            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities >= self.threshold).float()
        return predictions, probabilities