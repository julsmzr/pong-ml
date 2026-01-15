from src.game.pong import main as run_pong
from src.models.model_loader import load_decision_tree_model, load_hoeffding_tree_model, load_weighted_forest_model


def main(mode: str = "human") -> None:
    """Run Pong. mode: 'human', 'pc', 'dt', 'ht', 'wf'."""
    right_ai = None

    mode_registry = {
        "dt": load_decision_tree_model,
        "ht": load_hoeffding_tree_model,
        "wf": load_weighted_forest_model
    }
    
    try:
        right_ai = mode_registry[mode]()
        print(f"Loaded {mode}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    run_pong(right_ai=right_ai, right_mode=mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play Pong")
    parser.add_argument("--mode", choices=["human", "pc", "dt", "ht", "wf"], default="pc", help="Right player mode")
    args = parser.parse_args()
    main(args.mode)
