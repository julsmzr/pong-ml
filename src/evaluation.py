from dataclasses import dataclass
from src.game.pong import main as run_pong_game, FPS
from src.models.model_loader import PongAIPlayer


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    model_name: str
    survival_time_seconds: float
    survival_time_frames: int
    final_pc_score: int
    final_ai_score: int
    total_hits: int

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"Evaluation Results: {self.model_name.upper()}\n"
            f"{'='*50}\n"
            f"Survival Time: {self.survival_time_seconds:.2f}s ({self.survival_time_frames} frames)\n"
            f"Final Score: PC {self.final_pc_score} - {self.final_ai_score} AI\n"
            f"Total Ball Hits by AI: {self.total_hits}\n"
            f"{'='*50}\n"
        )


def evaluate_model(ai_player: PongAIPlayer, model_name: str, max_score: int = 10) -> EvaluationResult:
    """Evaluate an AI model against the Perfect Computer."""
    final_state = run_pong_game(
        right_ai=ai_player,
        right_mode=model_name,
        left_mode="pc",
        max_score=max_score
    )

    survival_time_seconds = final_state.frame_count / FPS

    return EvaluationResult(
        model_name=model_name,
        survival_time_seconds=survival_time_seconds,
        survival_time_frames=final_state.frame_count,
        final_pc_score=final_state.left_score,
        final_ai_score=final_state.right_score,
        total_hits=final_state.right_paddle_hits
    )


def evaluate_all_models(max_score: int = 10, models_dir: str = "models") -> dict[str, EvaluationResult]:
    """Evaluate all three models and return results."""
    from src.models.model_loader import (
        load_decision_tree_model,
        load_hoeffding_tree_model,
        load_weighted_forest_model
    )

    models = {
        'dt': ('Decision Tree', load_decision_tree_model),
        'ht': ('Hoeffding Tree', load_hoeffding_tree_model),
        'ct': ('Weighted Forest', load_weighted_forest_model)
    }

    results = {}

    for model_key, (model_name, loader_func) in models.items():
        print(f"\nEvaluating {model_name}...")
        try:
            ai_player = loader_func(models_dir)
            result = evaluate_model(ai_player, model_key, max_score)
            results[model_key] = result
            print(result)
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    return results


def compare_pretrained_vs_online(max_score: int = 10, models_dir: str = "models") -> dict[str, dict[str, EvaluationResult]]:
    """Compare pretrained vs online-trained models."""
    from pathlib import Path

    print("=" * 60)
    print("COMPARING PRETRAINED VS ONLINE-TRAINED MODELS")
    print("=" * 60)

    results = {'ht': {}, 'ct': {}}

    for model_key in ['ht', 'ct']:
        model_name = 'Hoeffding Tree' if model_key == 'ht' else 'Weighted Forest'
        pretrained_path = f"{'hoeffding_tree' if model_key == 'ht' else 'weighted_forest'}_pong.pkl"
        metadata_path = f"{'hoeffding_tree' if model_key == 'ht' else 'weighted_forest'}_metadata.pkl"
        online_path = f"{'hoeffding_tree' if model_key == 'ht' else 'weighted_forest'}_online.pkl"

        print(f"\n{model_name}:")

        try:
            pretrained_model = Path(models_dir) / pretrained_path
            pretrained_metadata = Path(models_dir) / metadata_path
            if pretrained_model.exists():
                ai = PongAIPlayer(pretrained_model, pretrained_metadata if pretrained_metadata.exists() else None)
                result = evaluate_model(ai, f"{model_key}_pretrained", max_score)
                results[model_key]['pretrained'] = result
                print(f"  Pretrained: {result.survival_time_seconds:.2f}s, Score: {result.final_pc_score}-{result.final_ai_score}, Hits: {result.total_hits}")
        except Exception as e:
            print(f"  Pretrained: Error - {e}")

        try:
            online_model = Path(models_dir) / online_path
            online_metadata = Path(models_dir) / metadata_path
            if online_model.exists():
                ai = PongAIPlayer(online_model, online_metadata if online_metadata.exists() else None)
                result = evaluate_model(ai, f"{model_key}_online", max_score)
                results[model_key]['online'] = result
                print(f"  Online: {result.survival_time_seconds:.2f}s, Score: {result.final_pc_score}-{result.final_ai_score}, Hits: {result.total_hits}")

                if 'pretrained' in results[model_key]:
                    improvement = ((result.survival_time_seconds - results[model_key]['pretrained'].survival_time_seconds) / results[model_key]['pretrained'].survival_time_seconds) * 100
                    print(f"  Improvement: {improvement:+.1f}%")
        except Exception as e:
            print(f"  Online: Error - {e}")

    return results


if __name__ == "__main__":
    results = evaluate_all_models(max_score=10)

    print("\n" + "="*50)
    print("SUMMARY OF ALL EVALUATIONS")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result.survival_time_seconds:.2f}s, "
              f"Score {result.final_pc_score}-{result.final_ai_score}, "
              f"Hits: {result.total_hits}")
