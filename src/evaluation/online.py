import os
from dataclasses import dataclass
import pandas as pd

from src.game.pong import main as run_pong_game, FPS
from src.models.model_loader import PongAIPlayer
from src.evaluation.statistical import friedman_test, wilcoxon_posthoc


@dataclass
class EvaluationResult:
    """Results from a single game evaluation."""
    model_name: str
    survival_time_seconds: float
    survival_time_frames: int
    final_pc_score: int
    final_ai_score: int
    total_hits: int


MODEL_CONFIGS: dict[str, tuple[str, str]] = {
    "DT_pretrained": (
        "models/dt/decision_tree_pong.pkl",
        "models/dt/decision_tree_metadata.pkl",
    ),
    "HT_pretrained": (
        "models/ht/hoeffding_tree_pong.pkl",
        "models/ht/hoeffding_tree_metadata.pkl",
    ),
    "HT_online": (
        "models/ht/hoeffding_tree_online.pkl",
        "models/ht/hoeffding_tree_metadata.pkl",
    ),
    "WF_pretrained": (
        "models/wf/weighted_forest_pong.pkl",
        "models/wf/weighted_forest_metadata.pkl",
    ),
    "WF_online": (
        "models/wf/weighted_forest_online.pkl",
        "models/wf/weighted_forest_metadata.pkl",
    ),
}


def evaluate_model(
    ai_player: PongAIPlayer,
    model_name: str,
    max_score: int = 10,
) -> EvaluationResult:
    """Evaluate an AI model against the perfect AI in a single game."""
    final_state = run_pong_game(
        right_ai=ai_player,
        right_mode=model_name,
        left_mode="pc",
        max_score=max_score,
    )

    return EvaluationResult(
        model_name=model_name,
        survival_time_seconds=final_state.frame_count / FPS,
        survival_time_frames=final_state.frame_count,
        final_pc_score=final_state.left_score,
        final_ai_score=final_state.right_score,
        total_hits=final_state.right_paddle_hits,
    )


def compare_pretrained_vs_online(
    max_score: int = 10,
    output_path: str = "models/comparison_results.csv",
) -> dict[str, dict[str, EvaluationResult]]:
    """Compare pretrained vs online-trained models for HT and WF."""
    results: dict[str, dict[str, EvaluationResult]] = {"ht": {}, "wf": {}}
    rows: list[dict[str, object]] = []

    for model_key in ["ht", "wf"]:
        if model_key == "ht":
            pretrained_path = "hoeffding_tree_pong.pkl"
            metadata_path = "hoeffding_tree_metadata.pkl"
            online_path = "hoeffding_tree_online.pkl"
        else:
            pretrained_path = "weighted_forest_pong.pkl"
            metadata_path = "weighted_forest_metadata.pkl"
            online_path = "weighted_forest_online.pkl"

        models_dir = f"models/{model_key}"

        pretrained_model = os.path.join(models_dir, pretrained_path)
        pretrained_metadata = os.path.join(models_dir, metadata_path)
        if os.path.exists(pretrained_model):
            try:
                meta = pretrained_metadata if os.path.exists(pretrained_metadata) else None
                ai = PongAIPlayer(pretrained_model, meta)
                result = evaluate_model(ai, f"{model_key}_pretrained", max_score)
                results[model_key]["pretrained"] = result
                rows.append({
                    "model": f"{model_key}_pretrained",
                    "survival_time": result.survival_time_seconds,
                    "ai_score": result.final_ai_score,
                    "pc_score": result.final_pc_score,
                    "hits": result.total_hits,
                })
            except Exception:
                pass

        online_model = os.path.join(models_dir, online_path)
        online_metadata = os.path.join(models_dir, metadata_path)
        if os.path.exists(online_model):
            try:
                meta = online_metadata if os.path.exists(online_metadata) else None
                ai = PongAIPlayer(online_model, meta)
                result = evaluate_model(ai, f"{model_key}_online", max_score)
                results[model_key]["online"] = result
                rows.append({
                    "model": f"{model_key}_online",
                    "survival_time": result.survival_time_seconds,
                    "ai_score": result.final_ai_score,
                    "pc_score": result.final_pc_score,
                    "hits": result.total_hits,
                })
            except Exception:
                pass

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    return results


def _run_game_simulations(num_games: int, max_score: int) -> pd.DataFrame:
    """Run game simulations for all available models."""
    all_results: list[dict[str, object]] = []

    for model_name, (model_path, metadata_path) in MODEL_CONFIGS.items():
        if not os.path.exists(model_path):
            continue

        try:
            ai = PongAIPlayer(model_path, metadata_path)
        except Exception:
            continue

        for game_num in range(1, num_games + 1):
            result = evaluate_model(ai, model_name, max_score)
            goal_diff = result.final_ai_score - result.final_pc_score

            all_results.append({
                "model": model_name,
                "game": game_num,
                "survival_time": result.survival_time_seconds,
                "returns": result.total_hits,
                "goal_diff": goal_diff,
                "ai_score": result.final_ai_score,
                "pc_score": result.final_pc_score,
            })

    return pd.DataFrame(all_results)


def _run_statistical_tests(
    df: pd.DataFrame,
    num_games: int,
    output_dir: str,
) -> dict[str, dict[str, object]]:
    """Run Friedman and post-hoc Wilcoxon tests for each metric."""
    metrics = ["survival_time", "returns", "goal_diff"]
    stats_results: dict[str, dict[str, object]] = {}
    stats_rows: list[dict[str, object]] = []

    for metric in metrics:
        results_dict: dict[str, list[float]] = {}
        for model in MODEL_CONFIGS.keys():
            model_data = df[df["model"] == model][metric].values
            if len(model_data) == num_games:
                results_dict[model] = model_data.tolist()

        if len(results_dict) < 3:
            continue

        try:
            friedman_stat, friedman_p = friedman_test(results_dict)
            stats_results[metric] = {
                "friedman_stat": friedman_stat,
                "friedman_p": friedman_p,
                "posthoc": None,
            }

            stats_rows.append({
                "metric": metric,
                "test": "friedman",
                "statistic": friedman_stat,
                "p_value": friedman_p,
                "significant": friedman_p < 0.05,
            })

            if friedman_p < 0.05:
                posthoc = wilcoxon_posthoc(results_dict)
                stats_results[metric]["posthoc"] = posthoc

                for pair, result in posthoc.items():
                    stats_rows.append({
                        "metric": metric,
                        "test": f"wilcoxon_{pair}",
                        "statistic": None,
                        "p_value": result["p_corrected"],
                        "significant": result["significant"],
                    })

        except Exception:
            continue

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(output_dir, "online_statistical_tests.csv"), index=False)

    return stats_results


def run_online_evaluation(
    num_games: int = 10,
    max_score: int = 5,
    output_dir: str = "models",
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Run online evaluation via game simulations."""
    df = _run_game_simulations(num_games, max_score)

    os.makedirs(output_dir, exist_ok=True)

    if not df.empty:
        df.to_csv(os.path.join(output_dir, "online_game_results.csv"), index=False)

        summary = df.groupby("model").agg({
            "survival_time": ["mean", "std"],
            "returns": ["mean", "std"],
            "goal_diff": ["mean", "std"],
        }).round(4)
        summary.columns = ["_".join(col) for col in summary.columns]
        summary.to_csv(os.path.join(output_dir, "online_summary.csv"))

    stats = _run_statistical_tests(df, num_games, output_dir)

    print(f"Results saved to {output_dir}/online_*.csv")

    return df, stats
