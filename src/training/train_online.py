import os
import pickle
from pathlib import Path
import pandas as pd

from src.game.pong import main as run_pong_game, FPS
from src.training.online_trainer import HoeffdingOnlineTrainer, WeightedForestOnlineTrainer, TrainerAIWrapper


def train_hoeffding_online(
    pretrained_model_path: str = "models/hoeffding_tree_pong.pkl",
    num_episodes: int = 100,
    max_score_per_episode: int = 5,
    save_interval: int = 10,
    output_dir: str = "models"
) -> None:
    """Run online training for Hoeffding Tree."""
    print("=" * 60)
    print("HOEFFDING TREE ONLINE TRAINING")
    print("=" * 60)

    print(f"\n[1/4] Loading pretrained model from {pretrained_model_path}...")
    with open(pretrained_model_path, 'rb') as f:
        model = pickle.load(f)
    print("  Model loaded successfully")

    print(f"\n[2/4] Setting up online trainer...")
    trainer = HoeffdingOnlineTrainer(model)
    ai_player = TrainerAIWrapper(trainer)
    print("  Trainer initialized")

    print(f"\n[3/4] Running {num_episodes} training episodes...")
    metrics_log = []

    for episode in range(num_episodes):
        print(f"\n  Episode {episode + 1}/{num_episodes}")

        final_state = run_pong_game(
            right_ai=ai_player,
            right_mode='ht',
            left_mode='pc',
            max_score=max_score_per_episode,
            online_trainer=trainer,
            learning_mode=True
        )

        episode_metrics = trainer.get_metrics()
        episode_metrics['episode'] = episode + 1
        episode_metrics['survival_frames'] = final_state.frame_count
        episode_metrics['survival_seconds'] = final_state.frame_count / FPS
        episode_metrics['pc_score'] = final_state.left_score
        episode_metrics['ai_score'] = final_state.right_score
        episode_metrics['hits'] = final_state.right_paddle_hits
        metrics_log.append(episode_metrics)

        print(f"  Survival: {episode_metrics['survival_seconds']:.2f}s, Score: {episode_metrics['pc_score']}-{episode_metrics['ai_score']}, Hits: {episode_metrics['hits']}")
        print(f"  Updates: {episode_metrics['total_updates']}, Avg Reward: {episode_metrics['avg_reward']:.3f}, Accuracy: {episode_metrics['progressive_accuracy']:.3f}")

        if (episode + 1) % save_interval == 0:
            checkpoint_path = Path(output_dir) / f"hoeffding_tree_online_ep{episode + 1}.pkl"
            trainer.save_model(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

    print(f"\n[4/4] Saving final model...")
    final_model_path = Path(output_dir) / "hoeffding_tree_online.pkl"
    trainer.save_model(final_model_path)
    print(f"  Final model saved to {final_model_path}")

    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = Path(output_dir) / "hoeffding_online_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Metrics saved to {metrics_csv_path}")

    print("\n" + "=" * 60)
    print("ONLINE TRAINING COMPLETE!")
    print("=" * 60)


def train_weighted_forest_online(
    pretrained_model_path: str = "models/weighted_forest_pong.pkl",
    metadata_path: str = "models/weighted_forest_metadata.pkl",
    num_episodes: int = 100,
    max_score_per_episode: int = 5,
    save_interval: int = 10,
    output_dir: str = "models"
) -> None:
    """Run online training for Weighted Forest."""
    print("=" * 60)
    print("WEIGHTED FOREST ONLINE TRAINING")
    print("=" * 60)

    print(f"\n[1/4] Loading pretrained model from {pretrained_model_path}...")
    with open(pretrained_model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print("  Model loaded successfully")

    print(f"\n[2/4] Setting up online trainer...")
    class_mapping = metadata.get('class_mapping', {0: 'D', 1: 'I', 2: 'U'})
    trainer = WeightedForestOnlineTrainer(model, class_mapping)
    ai_player = TrainerAIWrapper(trainer)
    print("  Trainer initialized")

    print(f"\n[3/4] Running {num_episodes} training episodes...")
    metrics_log = []

    for episode in range(num_episodes):
        print(f"\n  Episode {episode + 1}/{num_episodes}")

        final_state = run_pong_game(
            right_ai=ai_player,
            right_mode='ct',
            left_mode='pc',
            max_score=max_score_per_episode,
            online_trainer=trainer,
            learning_mode=True
        )

        episode_metrics = trainer.get_metrics()
        episode_metrics['episode'] = episode + 1
        episode_metrics['survival_frames'] = final_state.frame_count
        episode_metrics['survival_seconds'] = final_state.frame_count / FPS
        episode_metrics['pc_score'] = final_state.left_score
        episode_metrics['ai_score'] = final_state.right_score
        episode_metrics['hits'] = final_state.right_paddle_hits
        metrics_log.append(episode_metrics)

        print(f"  Survival: {episode_metrics['survival_seconds']:.2f}s, Score: {episode_metrics['pc_score']}-{episode_metrics['ai_score']}, Hits: {episode_metrics['hits']}")
        print(f"  Updates: {episode_metrics['total_updates']}, Avg Reward: {episode_metrics['avg_reward']:.3f}, Accuracy: {episode_metrics['accuracy']:.3f}, Cells: {episode_metrics['num_cells']}")

        if (episode + 1) % save_interval == 0:
            checkpoint_path = Path(output_dir) / f"weighted_forest_online_ep{episode + 1}.pkl"
            trainer.save_model(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

    print(f"\n[4/4] Saving final model...")
    final_model_path = Path(output_dir) / "weighted_forest_online.pkl"
    trainer.save_model(final_model_path)
    print(f"  Final model saved to {final_model_path}")

    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = Path(output_dir) / "weighted_forest_online_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Metrics saved to {metrics_csv_path}")

    print("\n" + "=" * 60)
    print("ONLINE TRAINING COMPLETE!")
    print("=" * 60)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Online training for Pong AI models")
    parser.add_argument("--model", choices=["ht", "wf", "both"], default="both", help="Model to train: ht (Hoeffding Tree), wf (Weighted Forest), both")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-score", type=int, default=5, help="Max score per episode")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N episodes")
    args = parser.parse_args()

    if args.model in ["ht", "both"]:
        train_hoeffding_online(
            num_episodes=args.episodes,
            max_score_per_episode=args.max_score,
            save_interval=args.save_interval
        )

    if args.model in ["wf", "both"]:
        train_weighted_forest_online(
            num_episodes=args.episodes,
            max_score_per_episode=args.max_score,
            save_interval=args.save_interval
        )


if __name__ == "__main__":
    main()
