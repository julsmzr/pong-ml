import os
import time
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from river.tree import HoeffdingTreeClassifier

from src.data.loader import load_training_data
from src.data.preparation import min_max_scale, convert_str_to_int, undersample
from src.models.weighted_forest.clf import WeightedForest, euclidean_distance
from src.evaluation.statistical import run_rskf


def _train_and_evaluate(
    model_name: str,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    inference_times: dict[str, list[float]],
) -> float:
    """Train a model and return test accuracy."""
    preds: list[int] | NDArray[np.int64]

    if model_name == "DecisionTree":
        clf = DecisionTreeClassifier(
            max_depth=20, min_samples_split=10, random_state=42
        )
        clf.fit(X_train, y_train)
        start_time = time.time()
        preds = clf.predict(X_test)
        end_time = time.time()

    elif model_name == "HoeffdingTree":
        clf = HoeffdingTreeClassifier()
        for x, y_label in zip(X_train, y_train):
            clf.learn_one(dict(enumerate(x)), int(y_label))
        start_time = time.time()
        preds = [clf.predict_one(dict(enumerate(x))) for x in X_test]
        end_time = time.time()

    elif model_name == "WeightedForest":
        clf = WeightedForest(
            X_train.shape[1],
            len(np.unique(y_train)),
            euclidean_distance,
            accuracy_goal=0.65,
            random_seed=42,
        )
        clf.fit(X_train, y_train, epochs=3)
        start_time = time.time()
        preds = clf.predict(X_test).astype(int)
        end_time = time.time()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    inference_times[model_name].append(end_time - start_time)
    return float(accuracy_score(y_test, preds))


def run_offline_evaluation(
    n_repeats: int = 10,
    n_splits: int = 5,
    random_state: int = 42,
    output_dir: str = "models",
) -> dict[str, object]:
    """Run offline evaluation with Repeated Stratified K-Fold CV."""
    X, y = load_training_data(random_state=random_state)
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    X_np, _, _ = min_max_scale(X_np)
    y_np, _ = convert_str_to_int(y_np)
    X_np, y_np = undersample(X_np, y_np, random_seed=random_state)

    models = {
        "DecisionTree": "DecisionTree",
        "HoeffdingTree": "HoeffdingTree",
        "WeightedForest": "WeightedForest",
    }

    inference_times: dict[str, list[float]] = {model: [] for model in models}

    def train_eval_fn(
        model_name: str,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_test: NDArray[np.float64],
        y_test: NDArray[np.int64],
    ) -> float:
        return _train_and_evaluate(
            model_name, X_train, y_train, X_test, y_test, inference_times
        )

    output = run_rskf(
        train_eval_fn,
        models,
        X_np,
        y_np,
        n_repeats=n_repeats,
        n_splits=n_splits,
        random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)

    cv_rows = []
    for model_name, scores in output["results"].items():
        for fold_idx, score in enumerate(scores):
            cv_rows.append({
                "model": model_name,
                "fold": fold_idx,
                "accuracy": score,
            })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(output_dir, "offline_cv_results.csv"), index=False)

    time_rows = []
    for model_name, times in inference_times.items():
        time_rows.append({
            "model": model_name,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
        })
    time_df = pd.DataFrame(time_rows)
    time_df.to_csv(os.path.join(output_dir, "offline_inference_times.csv"), index=False)

    stats_rows = [{
        "test": "friedman",
        "statistic": output["friedman"]["statistic"],
        "p_value": output["friedman"]["p_value"],
    }]
    if output.get("posthoc"):
        for pair, result in output["posthoc"].items():
            stats_rows.append({
                "test": f"wilcoxon_{pair}",
                "statistic": None,
                "p_value": result["p_corrected"],
                "significant": result["significant"],
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(output_dir, "offline_statistical_tests.csv"), index=False)

    print(f"Results saved to {output_dir}/offline_*.csv")

    return {
        "cv_results": output,
        "inference_times": inference_times,
    }
