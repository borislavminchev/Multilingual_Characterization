"""
Single import point for the model components.

Losses live in losses.py and classifiers/heads in classifiers.py; this module
just re-exports them so callers can do `from hierarchical_model import ...`.
"""

# Re-export from losses.py
from losses import (
    FocalLoss,
    ClassBalancedCrossEntropy,
    MultiLabelFocalLoss,
    AsymmetricLoss,
    AsymmetricLossOptimized,
    CardinalityRegularizer
)

# Re-export from classifiers.py
from classifiers import (
    entity_span_pooling,
    SemanticSimilarityHead,
    FineSemanticSimilarityHead,
    HierarchyAffinityLayer,
    CoarseRoleClassifier,
    FineRoleClassifier,
    SoftConditionedFineClassifier
)

__all__ = [
    'FocalLoss',
    'ClassBalancedCrossEntropy',
    'MultiLabelFocalLoss',
    'AsymmetricLoss',
    'AsymmetricLossOptimized',
    'CardinalityRegularizer',
    'entity_span_pooling',
    'SemanticSimilarityHead',
    'FineSemanticSimilarityHead',
    'HierarchyAffinityLayer',
    'CoarseRoleClassifier',
    'FineRoleClassifier',
    'SoftConditionedFineClassifier',
]