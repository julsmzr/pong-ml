import numpy as np
from scipy import stats
from itertools import combinations


def friedman_test(results: dict[str, list[float]]) -> tuple[float, float]:
    """Perform Friedman test on model results."""
    data = np.array(list(results.values()))
    statistic, pvalue = stats.friedmanchisquare(*data)
    return statistic, pvalue


def wilcoxon_posthoc(results: dict[str, list[float]], alpha: float = 0.05) -> dict:
    """Perform pairwise Wilcoxon signed-rank tests with Hommel correction."""
    model_names = list(results.keys())
    pairs = list(combinations(model_names, 2))
    pvalues = []

    for m1, m2 in pairs:
        _, pval = stats.wilcoxon(results[m1], results[m2], alternative='two-sided')
        pvalues.append(pval)

    corrected = hommel_correction(pvalues, alpha)

    comparisons = {}
    for i, (m1, m2) in enumerate(pairs):
        comparisons[f"{m1}_vs_{m2}"] = {
            'p_value': pvalues[i],
            'p_corrected': corrected[i],
            'significant': corrected[i] < alpha
        }

    return comparisons


def hommel_correction(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """Apply Hommel's step-up procedure for multiple comparisons."""
    n = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvals = np.array(pvalues)[sorted_indices]

    corrected = np.zeros(n)
    for i in range(n):
        corrected[sorted_indices[i]] = min(1.0, sorted_pvals[i] * n / (i + 1))

    return corrected.tolist()


def run_rskf(train_eval_fn, models: dict, X, y, n_repeats: int = 10, n_splits: int = 5, random_state: int = 42) -> dict:
    """Run cross-validation and statistical tests."""
    from sklearn.model_selection import StratifiedKFold

    results = {name: [] for name in models.keys()}

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + rep)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for name, model in models.items():
                score = train_eval_fn(model, X_train, y_train, X_test, y_test)
                results[name].append(score)

    friedman_stat, friedman_p = friedman_test(results)

    output = {
        'results': results,
        'friedman': {'statistic': friedman_stat, 'p_value': friedman_p}
    }

    if friedman_p < 0.05:
        output['posthoc'] = wilcoxon_posthoc(results)

    return output


def print_results(output: dict) -> None:
    """Print formatted statistical test results."""

    print("Model Performance (mean ± std):")
    for name, scores in output['results'].items():
        print(f"  {name}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    print(f"\nFriedman Test:")
    print(f"  Statistic: {output['friedman']['statistic']:.4f}")
    print(f"  P-value: {output['friedman']['p_value']:.4f}")

    if 'posthoc' in output:
        print("\nPost-hoc Pairwise Comparisons (Wilcoxon + Hommel):")
        for pair, result in output['posthoc'].items():
            sig = "*" if result['significant'] else ""
            print(f"  {pair}: p={result['p_corrected']:.4f} {sig}")
