import os
import pandas as pd
import pickle
from river import tree, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.data.loader import load_training_data_class_balanced, TARGET_COL


def train_hoeffding_tree(
    grace_period: int = 200,
    delta: float = 1e-7,
    tau: float = 0.05,
    leaf_prediction: str = "nba",
    random_state: int = 42,
    test_size: float = 0.2
) -> tuple[tree.HoeffdingTreeClassifier, pd.DataFrame, pd.Series, dict]:
    """Offline pretraining of Hoeffding Tree on existing data (online tuning during gameplay can be added later)."""
    print("=" * 60)
    print("HOEFFDING TREE OFFLINE PRETRAINING - PONG")
    print("=" * 60)
    print("NOTE: This pretains the model on existing data.")
    print("      Online learning during gameplay will be added in next step.")

    # Load data
    print("\n[1/5] Loading data...")
    X, y = load_training_data_class_balanced(random_state=random_state)
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {list(X.columns)}")
    print(f"  Classes: {sorted(y.unique())}")  
    print(f"  Class distribution (balanced):\n{y.value_counts()}")

    # Split data
    print(f"\n[2/5] Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Train model
    print(f"\n[3/5] Training Hoeffding Tree (online learning)...")
    print(f"  Hyperparameters:")
    print(f"    grace_period: {grace_period}")
    print(f"    delta: {delta}")
    print(f"    tau: {tau}")
    print(f"    leaf_prediction: {leaf_prediction}")

    clf = tree.HoeffdingTreeClassifier(
        grace_period=grace_period,
        delta=delta,
        tau=tau,
        leaf_prediction=leaf_prediction,
        nominal_attributes=None
    )

    # Train incrementally
    train_metric = metrics.Accuracy()
    for idx in range(len(X_train)):
        x_dict = X_train.iloc[idx].to_dict()
        y_true = y_train.iloc[idx]

        if idx > 0:
            y_pred = clf.predict_one(x_dict)
            train_metric.update(y_true, y_pred)

        clf.learn_one(x_dict, y_true)

        if (idx + 1) % 10000 == 0:
            print(f"    Processed {idx + 1}/{len(X_train)} samples...")

    train_acc = train_metric.get()
    print(f"  Progressive validation accuracy: {train_acc:.4f}")

    # Evaluate
    print(f"\n[4/5] Evaluating model...")
    y_test_pred = []
    for idx in range(len(X_test)):
        x_dict = X_test.iloc[idx].to_dict()
        y_pred = clf.predict_one(x_dict)
        y_test_pred.append(y_pred)

    test_acc = sum(y_test.iloc[i] == y_test_pred[i] for i in range(len(y_test))) / len(y_test)

    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    print("\n  Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    print("\n  Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))

    # Save model
    print(f"\n[5/5] Saving model...")

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/hoeffding_tree_pong.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Model saved to: {model_path}")

    # Also save metadata
    metadata = {
        'model_type': 'HoeffdingTreeClassifier',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'hyperparameters': {
            'grace_period': grace_period,
            'delta': delta,
            'tau': tau,
            'leaf_prediction': leaf_prediction,
            'random_state': random_state
        },
        'features': list(X.columns),
        'target': TARGET_COL,
        'classes': sorted(y.unique())
    }

    metadata_path = f"{models_dir}/hoeffding_tree_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return clf, X_test, y_test, {
        'train_acc': train_acc,
        'test_acc': test_acc
    }


def main() -> None:
    clf, X_test, y_test, metrics = train_hoeffding_tree(
        grace_period=50, 
        delta=1e-7,
        tau=0.1,
        leaf_prediction='nba',
        random_state=42,
        test_size=0.2
    )


if __name__ == "__main__":
    main()
