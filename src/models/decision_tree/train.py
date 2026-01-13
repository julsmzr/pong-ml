import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from data.loader import load_training_data, FEATURE_COLS, TARGET_COL


def train_decision_tree(
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    test_size: float = 0.2
) -> tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series, dict]:
    """Train a Decision Tree classifier on Pong data."""
    print("=" * 60)
    print("DECISION TREE OFFLINE TRAINING - PONG")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    X, y = load_training_data(random_state=random_state)
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {list(X.columns)}")
    print(f"  Classes: {sorted(y.unique())}")
    print(f"  Class distribution:\n{y.value_counts()}")

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
    print(f"\n[3/5] Training Decision Tree...")
    print(f"  Hyperparameters:")
    print(f"    max_depth: {max_depth}")
    print(f"    min_samples_split: {min_samples_split}")
    print(f"    min_samples_leaf: {min_samples_leaf}")

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion='gini'
    )

    clf.fit(X_train, y_train)
    print(f"  Tree depth: {clf.get_depth()}")
    print(f"  Number of leaves: {clf.get_n_leaves()}")

    # Evaluate
    print(f"\n[4/5] Evaluating model...")
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    print("\n  Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    print("\n  Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))

    # Feature importance
    print("\n  Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))

    # Save model
    print(f"\n[5/5] Saving model...")
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "decision_tree_pong.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Model saved to: {model_path}")

    # Also save metadata
    metadata = {
        'model_type': 'DecisionTreeClassifier',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'tree_depth': clf.get_depth(),
        'n_leaves': clf.get_n_leaves(),
        'hyperparameters': {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        },
        'features': FEATURE_COLS,
        'target': TARGET_COL,
        'classes': sorted(y.unique())
    }

    metadata_path = models_dir / "decision_tree_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return clf, X_test, y_test, {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'feature_importance': feature_importance
    }


def visualize_results(clf: DecisionTreeClassifier, feature_importance: pd.DataFrame, save_path: Path | None = None) -> None:
    """Create visualization of feature importance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(feature_importance['feature'], feature_importance['importance'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Decision Tree Feature Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    plt.show()


def main() -> None:
    # Train the model
    clf, X_test, y_test, metrics = train_decision_tree(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        test_size=0.2
    )

    # Visualize feature importance
    visualize_results(
        clf,
        metrics['feature_importance'],
        save_path=Path(__file__).parent.parent.parent.parent / "models" / "dt_feature_importance.png"
    )


if __name__ == "__main__":
    main()
