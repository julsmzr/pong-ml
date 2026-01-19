import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

from src.data.loader import load_training_data, TARGET_COL
from src.data.preparation import min_max_scale, convert_str_to_int, undersample
from src.models.weighted_forest.clf import WeightedForest, euclidean_distance


VERBOSE = False


def vprint(message: str) -> None:
    if VERBOSE:
        print(message)

def train_weighted_forest(
    learning_decay: float = 0.9,
    accuracy_goal: float = 0.8,
    epochs: int = 3,
    random_state: int = 42,
    test_size: float = 0.1,
    initializer_low: float = 0,
    initializer_high: float = 1,
    num_start_cells: int = 4,
    similarity_threshold: float = 2.0,
) -> tuple[WeightedForest, pd.DataFrame, pd.Series, dict]:
    """Train a Weighted Forest classifier on Pong data."""
    print("Running offline training for Weighted Forest")

    print("Loading data...")
    X, y = load_training_data(random_state=random_state)
    feature_columns = list(X.columns)

    vprint(f"  Total samples: {len(X)}")
    vprint(f"  Features: {feature_columns}")
    vprint(f"  Classes: {sorted(y.unique())}")

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    print("Preparing data...")
    X_np, feature_min, feature_max = min_max_scale(X_np)
    y_np, class_mapping = convert_str_to_int(y_np)
    X_np, y_np = undersample(X_np, y_np, random_seed=random_state)

    vprint(f"  After undersampling: {len(X_np)} samples")
    vprint(f"  Class distribution (balanced): {np.bincount(y_np)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np,
        test_size=test_size,
        random_state=random_state,
        stratify=y_np
    )
    vprint(f"  Training samples: {len(X_train)}")
    vprint(f"  Test samples: {len(X_test)}")

    print(f"Training Weighted Forest...")
    vprint(f"  Hyperparameters:")
    vprint(f"    num_features: {X_train.shape[1]}")
    vprint(f"    num_classes: {len(class_mapping)}")
    vprint(f"    learning_decay: {learning_decay}")
    vprint(f"    accuracy_goal: {accuracy_goal}")
    vprint(f"    epochs: {epochs}")

    clf = WeightedForest(
        num_features=X_train.shape[1],
        num_classes=len(class_mapping),
        distance_function=euclidean_distance,
        learning_decay=learning_decay,
        accuracy_goal=accuracy_goal,
        initializer_low=initializer_low,
        initializer_high=initializer_high,
        random_seed=random_state,
        num_start_cells=num_start_cells,
        similarity_threshold=similarity_threshold
    )

    epoch_accuracies = clf.fit(X_train, y_train, epochs=epochs)

    vprint(f"  Final training accuracy: {epoch_accuracies[-1]:.4f}")
    vprint(f"  Final number of cells: {len(clf.cells)}")

    print(f"Evaluating model...")
    y_train_pred = clf.predict(X_train).astype(int)
    y_test_pred = clf.predict(X_test).astype(int)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Balanced Accuracy: {test_balanced_acc:.4f}")

    # Convert back to string labels for classification report
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    y_test_str = pd.Series(y_test).map(reverse_mapping)
    y_test_pred_str = pd.Series(y_test_pred).map(reverse_mapping)

    vprint("  Classification Report (Test Set):")
    vprint(classification_report(y_test_str, y_test_pred_str))

    vprint("  Confusion Matrix (Test Set):")
    vprint(confusion_matrix(y_test_str, y_test_pred_str))

    print(f"Saving model...")

    models_dir = "models/wf"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/weighted_forest_pong.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Model saved to: {model_path}")

    metadata = {
        'model_type': 'WeightedForest',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'test_balanced_accuracy': test_balanced_acc,
        'num_cells': len(clf.cells),
        'hyperparameters': {
            'learning_decay': learning_decay,
            'accuracy_goal': accuracy_goal,
            'epochs': epochs,
            'random_state': random_state,
            'initializer_low': initializer_low,
            'initializer_high': initializer_high
        },
        'features': feature_columns,
        'target': TARGET_COL,
        'classes': list(class_mapping.keys()),
        'class_mapping': class_mapping,
        'scaler': {
            'feature_min': feature_min,
            'feature_max': feature_max
        }
    }

    metadata_path = f"{models_dir}/weighted_forest_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved to: {metadata_path}")

    print("Training Finished for Weighted Forest\n")


    return clf, X_test, y_test, {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_balanced_acc': test_balanced_acc,
        'epoch_accuracies': epoch_accuracies
    }


def main(
    learning_decay: float = 0.95,
    accuracy_goal: float = 0.65,
    epochs: int = 3,
    random_state: int = 42,
    test_size: float = 0.1,
    initializer_low: float = 0,
    initializer_high: float = 1,
    num_start_cells: int = 7,
    similarity_threshold: float = 1.25
) -> None:
    train_weighted_forest(
        learning_decay=learning_decay,
        accuracy_goal=accuracy_goal,
        epochs=epochs,
        random_state=random_state,
        test_size=test_size,
        initializer_low=initializer_low,
        initializer_high=initializer_high,
        num_start_cells=num_start_cells,
        similarity_threshold=similarity_threshold,
    )


if __name__ == "__main__":
    main()
