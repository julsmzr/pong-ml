import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.data.loader import load_training_data_class_balanced, TARGET_COL
from src.models.weighted_forest.clf import Weighted_Forest, euclidean_distance


def train_weighted_forest(
    learning_decay: float = 0.9,
    accuracy_goal: float = 0.8,
    epochs: int = 3,
    random_state: int = 42,
    test_size: float = 0.2
) -> tuple[Weighted_Forest, pd.DataFrame, pd.Series, dict]:
    """Train a Weighted Forest classifier on Pong data."""
    print("=" * 60)
    print("WEIGHTED FOREST TRAINING - PONG")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    X, y = load_training_data_class_balanced(random_state=random_state)
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {list(X.columns)}")
    print(f"  Classes: {sorted(y.unique())}")
    print(f"  Class distribution (balanced):\n{y.value_counts()}")

    # Convert to numpy and encode labels
    X_np = X.to_numpy()
    class_mapping = {'D': 0, 'I': 1, 'U': 2}
    y_np = y.map(class_mapping).to_numpy()

    # Split data
    print(f"\n[2/5] Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np,
        test_size=test_size,
        random_state=random_state,
        stratify=y_np
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Train model
    print(f"\n[3/5] Training Weighted Forest...")
    print(f"  Hyperparameters:")
    print(f"    num_features: {X_train.shape[1]}")
    print(f"    num_classes: {len(class_mapping)}")
    print(f"    learning_decay: {learning_decay}")
    print(f"    accuracy_goal: {accuracy_goal}")
    print(f"    epochs: {epochs}")

    clf = Weighted_Forest(
        num_features=X_train.shape[1],
        num_classes=len(class_mapping),
        distance_function=euclidean_distance,
        learning_decay=learning_decay,
        accuracy_goal=accuracy_goal
    )

    epoch_accuracies = clf.fit(X_train, y_train, epochs=epochs)

    print(f"  Final training accuracy: {epoch_accuracies[-1]:.4f}")
    print(f"  Final number of cells: {len(clf.cells)}")

    # Evaluate
    print(f"\n[4/5] Evaluating model...")
    y_train_pred = clf.predict(X_train).astype(int)
    y_test_pred = clf.predict(X_test).astype(int)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Convert back to string labels for classification report
    reverse_mapping = {0: 'D', 1: 'I', 2: 'U'}
    y_test_str = pd.Series(y_test).map(reverse_mapping)
    y_test_pred_str = pd.Series(y_test_pred).map(reverse_mapping)

    print("\n  Classification Report (Test Set):")
    print(classification_report(y_test_str, y_test_pred_str))

    print("\n  Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test_str, y_test_pred_str))

    # Save model
    print(f"\n[5/5] Saving model...")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/weighted_forest_pong.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Model saved to: {model_path}")

    # Also save metadata
    metadata = {
        'model_type': 'WeightedForest',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'num_cells': len(clf.cells),
        'hyperparameters': {
            'learning_decay': learning_decay,
            'accuracy_goal': accuracy_goal,
            'epochs': epochs,
            'random_state': random_state
        },
        'features': list(X.columns),
        'target': TARGET_COL,
        'classes': sorted(y.unique()),
        'class_mapping': class_mapping
    }

    metadata_path = f"{models_dir}/weighted_forest_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return clf, X_test, y_test, {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'epoch_accuracies': epoch_accuracies
    }


def main() -> None:
    clf, X_test, y_test, metrics = train_weighted_forest(
        learning_decay=0.95, 
        accuracy_goal=0.65, 
        epochs=5, 
        random_state=42,
        test_size=0.2
    )


if __name__ == "__main__":
    main()
