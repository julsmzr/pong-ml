from src.game.pong import main as run_pong
from src.models.model_loader import load_decision_tree_model


def main(mode: str = "human") -> None:
    """Run Pong. mode: 'human', 'pc', 'dt', 'ht', 'ct'."""
    right_ai = None

    if mode == "dt":
        try:
            right_ai = load_decision_tree_model()
            print(f"Loaded Decision Tree (acc: {right_ai.get_info().get('test_accuracy', 0):.3f})")
        except FileNotFoundError:
            print("DT model not found. Train it: python3 src/models/decision_tree/offline_train.py")
            exit(1)
    elif mode == "ht":
        print("Hoeffding Tree not yet implemented")
        exit(1)
    elif mode == "ct":
        print("Custom Tree not yet implemented")
        exit(1)

    run_pong(right_ai=right_ai, right_mode=mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play Pong")
    parser.add_argument("--mode", choices=["human", "pc", "dt", "ht", "ct"], default="pc", help="Right player mode")
    args = parser.parse_args()
    main(args.mode)
