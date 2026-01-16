import numpy as np
import pandas as pd


def min_max_scale(X: np.ndarray, feature_min: np.ndarray = None, feature_max: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply min-max scaling to features, scaling each feature to [0, 1]."""
    if feature_min is None:
        feature_min = X.min(axis=0)
    if feature_max is None:
        feature_max = X.max(axis=0)

    range_vals = feature_max - feature_min
    range_vals[range_vals == 0] = 1.0

    X_scaled = (X - feature_min) / range_vals
    return X_scaled, feature_min, feature_max


def min_max_scale_single(features: np.ndarray, feature_min: np.ndarray, feature_max: np.ndarray) -> np.ndarray:
    """Apply min-max scaling to a single feature vector using pre-computed min/max."""
    range_vals = feature_max - feature_min
    range_vals[range_vals == 0] = 1.0
    return (features - feature_min) / range_vals


def convert_str_to_int(y: np.ndarray, class_mapping: dict[str, int] = None) -> tuple[np.ndarray, dict[str, int]]:
    """Convert string labels to integers."""
    if class_mapping is None:
        unique_labels = sorted(np.unique(y))
        class_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    y_int = np.array([class_mapping[label] for label in y])
    return y_int, class_mapping


def undersample(X: np.ndarray, y: np.ndarray, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Undersample to balance classes by reducing majority classes to match minority."""
    np.random.seed(random_seed)

    unique_classes = np.unique(y)
    class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
    min_count = min(class_counts.values())

    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        sampled_indices = np.random.choice(cls_indices, size=min_count, replace=False)
        balanced_indices.extend(sampled_indices)

    np.random.shuffle(balanced_indices)
    return X[balanced_indices], y[balanced_indices]
