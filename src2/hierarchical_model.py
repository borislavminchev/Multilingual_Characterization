"""
Hierarchical Model Components for Entity Role Classification

This module re-exports all classifier and loss components for backward compatibility.
The actual implementations have been moved to separate modules:
- losses.py: All loss functions (FocalLoss, ASL, etc.)
- classifiers.py: All classifier models (CoarseRoleClassifier, FineRoleClassifier, etc.)

Usage:
    from hierarchical_model import CoarseRoleClassifier, FineRoleClassifier
    # or
    from classifiers import CoarseRoleClassifier, FineRoleClassifier
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
    # Utility functions
    entity_span_pooling,
    multi_view_entity_pooling,
    # Heads
    SemanticSimilarityHead,
    FineSemanticSimilarityHead,
    HierarchyAffinityLayer,
    MultiViewFusion,
    # Classifiers
    CoarseRoleClassifier,
    MultiViewCoarseClassifier,
    FineRoleClassifier,
    SoftConditionedFineClassifier
)

# All public exports
__all__ = [
    # Losses
    'FocalLoss',
    'ClassBalancedCrossEntropy',
    'MultiLabelFocalLoss',
    'AsymmetricLoss',
    'AsymmetricLossOptimized',
    'CardinalityRegularizer',
    # Utility functions
    'entity_span_pooling',
    'multi_view_entity_pooling',
    # Heads
    'SemanticSimilarityHead',
    'FineSemanticSimilarityHead',
    'HierarchyAffinityLayer',
    'MultiViewFusion',
    # Classifiers
    'CoarseRoleClassifier',
    'MultiViewCoarseClassifier',
    'FineRoleClassifier',
    'SoftConditionedFineClassifier',
]
