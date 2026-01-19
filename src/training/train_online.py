import os
import pickle
import pandas as pd

from src.game.pong import main as run_pong_game, FPS
from src.training.online_trainer import DecisionOnlineTrainer, HoeffdingOnlineTrainer, WeightedForestOnlineTrainer, TrainerAIWrapper


VERBOSE = False


def vprint(message: str) -> None:
    if VERBOSE:
        print(message)

def train_decision_online(
    pretrained_model_path: str = "models/dt/decision_tree_pong.pkl",
    num_episodes: int = 100,
    max_score_per_episode: int = 5,
    save_interval: int = 10,
    output_dir: str = "models/dt"
) -> None:
    """This function is mainly made for consisency reasons. No training is performed"""
    print("Running online training for Decision Tree")

    print(f"Loading pretrained model from {pretrained_model_path}...")
    with open(pretrained_model_path, 'rb') as f:
        model = pickle.load(f)
    vprint("  Model loaded successfully")

    print(f"Setting up online trainer...")
    trainer = DecisionOnlineTrainer(model)
    ai_player = TrainerAIWrapper(trainer)
    vprint("  Trainer initialized")

    print(f"Running {num_episodes} training episodes...")
    metrics_log = []

    for episode in range(num_episodes):
        vprint(f"\n  Episode {episode + 1}/{num_episodes}")

        final_state = run_pong_game(
            right_ai=ai_player,
            right_mode='dt',
            left_mode='pc',
            max_score=max_score_per_episode,
            online_trainer=trainer,
            learning_mode=False
        )

        episode_metrics = trainer.get_metrics()
        episode_metrics['episode'] = episode + 1
        episode_metrics['survival_frames'] = final_state.frame_count
        episode_metrics['survival_seconds'] = final_state.frame_count / FPS
        episode_metrics['pc_score'] = final_state.left_score
        episode_metrics['ai_score'] = final_state.right_score
        episode_metrics['hits'] = final_state.right_paddle_hits
        metrics_log.append(episode_metrics)

        vprint(f"  Survival: {episode_metrics['survival_seconds']:.2f}s, Score: {episode_metrics['pc_score']}-{episode_metrics['ai_score']}, Hits: {episode_metrics['hits']}")
        vprint(f"  Updates: {episode_metrics['total_updates']}, Avg Reward: {episode_metrics['avg_reward']:.3f}, Accuracy: {episode_metrics['progressive_accuracy']:.3f}")


    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = f"{output_dir}/decision_online_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Metrics saved to {metrics_csv_path}")

    print("Online Training Finished for Decision Tree\n")

def train_hoeffding_online(
    pretrained_model_path: str = "models/ht/hoeffding_tree_pong.pkl",
    num_episodes: int = 100,
    max_score_per_episode: int = 5,
    save_interval: int = 10,
    output_dir: str = "models/ht"
) -> None:
    """Run online training for Hoeffding Tree."""
    print("Running online training for Hoeffding Tree")

    print(f"Loading pretrained model from {pretrained_model_path}...")
    with open(pretrained_model_path, 'rb') as f:
        model = pickle.load(f)
    vprint("  Model loaded successfully")

    print(f"Setting up online trainer...")
    trainer = HoeffdingOnlineTrainer(model)
    ai_player = TrainerAIWrapper(trainer)
    vprint("  Trainer initialized")

    print(f"Running {num_episodes} training episodes...")
    metrics_log = []

    for episode in range(num_episodes):
        vprint(f"\n  Episode {episode + 1}/{num_episodes}")

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

        vprint(f"  Survival: {episode_metrics['survival_seconds']:.2f}s, Score: {episode_metrics['pc_score']}-{episode_metrics['ai_score']}, Hits: {episode_metrics['hits']}")
        vprint(f"  Updates: {episode_metrics['total_updates']}, Avg Reward: {episode_metrics['avg_reward']:.3f}, Accuracy: {episode_metrics['progressive_accuracy']:.3f}")

        if (episode + 1) % save_interval == 0:
            checkpoint_path = f"{output_dir}/hoeffding_tree_online_ep{episode + 1}.pkl"
            trainer.save_model(checkpoint_path)
            vprint(f"  Checkpoint saved to {checkpoint_path}")

    print(f"Saving final model...")
    final_model_path = f"{output_dir}/hoeffding_tree_online.pkl"
    trainer.save_model(final_model_path)
    print(f"  Final model saved to {final_model_path}")

    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = f"{output_dir}/hoeffding_online_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Metrics saved to {metrics_csv_path}")

    print("Online Training Finished for Hoeffding Tree\n")


def train_weighted_forest_online(
    pretrained_model_path: str = "models/wf/weighted_forest_pong.pkl",
    metadata_path: str = "models/wf/weighted_forest_metadata.pkl",
    num_episodes: int = 100,
    max_score_per_episode: int = 5,
    save_interval: int = 10,
    output_dir: str = "models/wf"
) -> None:
    """Run online training for Weighted Forest."""
    print("Running online training for Weighted Forest")

    print(f"Loading pretrained model from {pretrained_model_path}...")
    with open(pretrained_model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    vprint("  Model loaded successfully")

    print(f"Setting up online trainer...")
    class_mapping = metadata.get('class_mapping', {'D': 0, 'I': 1, 'U': 2})
    scaler = metadata.get('scaler', None)
    trainer = WeightedForestOnlineTrainer(model, class_mapping, scaler=scaler)
    ai_player = TrainerAIWrapper(trainer)
    vprint("  Trainer initialized")
    if scaler is not None:
        vprint("  Using min-max scaling from offline training")

    print(f"Running {num_episodes} training episodes...")
    metrics_log = []

    for episode in range(num_episodes):
        vprint(f"\n  Episode {episode + 1}/{num_episodes}")

        final_state = run_pong_game(
            right_ai=ai_player,
            right_mode='wf',
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

        vprint(f"  Survival: {episode_metrics['survival_seconds']:.2f}s, Score: {episode_metrics['pc_score']}-{episode_metrics['ai_score']}, Hits: {episode_metrics['hits']}")
        vprint(f"  Updates: {episode_metrics['total_updates']}, Avg Reward: {episode_metrics['avg_reward']:.3f}, Accuracy: {episode_metrics['accuracy']:.3f}, Cells: {episode_metrics['num_cells']}")

        if (episode + 1) % save_interval == 0:
            checkpoint_path = f"{output_dir}/weighted_forest_online_ep{episode + 1}.pkl"
            trainer.save_model(checkpoint_path)
            vprint(f"  Checkpoint saved to {checkpoint_path}")

    print(f"Saving final model...")
    final_model_path = f"{output_dir}/weighted_forest_online.pkl"
    trainer.save_model(final_model_path)
    print(f"  Final model saved to {final_model_path}")

    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = f"{output_dir}/weighted_forest_online_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Metrics saved to {metrics_csv_path}")

    print("Online Training Finished for Weighted Forest\n")
