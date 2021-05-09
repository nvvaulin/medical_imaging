import numpy as np
import torch


class WeightedClassRandomSampler(torch.utils.data.WeightedRandomSampler):
    def __init__(self, labels, class_weights=None, label_names=None, names_weights=None):
        if class_weights is None:
            class_weights = [names_weights.get(i, None) for i in label_names]
        mask = np.array([not (i is None) for i in class_weights])
        if mask.sum() < len(mask):
            labels = labels[:, mask]
            labels = np.concatenate((labels, (labels.max(1) == 0)[:, None]), 1)
        assert (labels.sum(1).max() != 1).sum() == 0, 'for weighted classes labels should be one hot encoded'
        class_ratios = labels.mean(0)

        class_weights = np.array(class_weights, dtype=np.float32)
        if mask.sum() < len(mask):
            class_weights = class_weights[mask]
            class_weights = np.concatenate((class_weights, np.array([1. - class_weights.sum()])))
        else:
            class_weights /=class_weights.sum()

        weights = ((class_weights / class_ratios)[None, :] * labels).max(1)
        super().__init__(weights, len(labels))