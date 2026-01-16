import os
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from src.data.loader import load_training_data, TARGET_COL

VERBOSE = False


def vprint(message: str) -> None:
    if VERBOSE:
        print(message)

def train_decision_tree(
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    test_size: float = 0.2
) -> tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series, dict]:
    """Train a Decision Tree classifier on Pong data."""
    print("Running offline training for Decision Tree")

    print("Loading data...")
    X, y = load_training_data(random_state=random_state)

    vprint(f"  Total samples: {len(X)}")
    vprint(f"  Features: {list(X.columns)}")
    vprint(f"  Classes: {sorted(y.unique())}")
    vprint(f"  Class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    vprint(f"  Training samples: {len(X_train)}")
    vprint(f"  Test samples: {len(X_test)}")

    print(f"Training Decision Tree...")
    vprint(f"  Hyperparameters:")
    vprint(f"    max_depth: {max_depth}")
    vprint(f"    min_samples_split: {min_samples_split}")
    vprint(f"    min_samples_leaf: {min_samples_leaf}")

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion='gini',
        class_weight='balanced'  # Handle class imbalance
    )

    clf.fit(X_train, y_train)
    vprint(f"  Tree depth: {clf.get_depth()}")
    vprint(f"  Number of leaves: {clf.get_n_leaves()}")

    print(f"Evaluating model...")
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    vprint("  Classification Report (Test Set):")
    vprint(classification_report(y_test, y_test_pred))

    vprint("  Confusion Matrix (Test Set):")
    vprint(confusion_matrix(y_test, y_test_pred))

    vprint("  Feature Importances:")
    feature_names = list(X_train.columns)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    vprint(feature_importance.to_string(index=False))

    print(f"Saving model...")
    models_dir = "models/dt"
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/decision_tree_pong.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Model saved to: {model_path}")

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
        'features': feature_names,
        'target': TARGET_COL,
        'classes': sorted(y.unique())
    }

    metadata_path =f"{models_dir}/decision_tree_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Metadata saved to: {metadata_path}")

    print("Training Finished for Decision Tree\n")

    return clf, X_test, y_test, {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'feature_importance': feature_importance
    }


def visualize_results(feature_importance: pd.DataFrame, save_path: str | None = None) -> None:
    """Create visualization of feature importance."""
    _, ax = plt.subplots(figsize=(10, 6))

    ax.barh(feature_importance['feature'], feature_importance['importance'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Decision Tree Feature Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    plt.show()


def main(visualize_results: bool = False) -> None:
    _, _, _, metrics = train_decision_tree(
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        test_size=0.2
    )

    if not visualize_results:
        return
    
    visualize_results(
        metrics['feature_importance'],
        save_path="models/dt_feature_importance.png"
    )

if __name__ == "__main__":
    main()
