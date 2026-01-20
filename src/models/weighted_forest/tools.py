import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

from src.data.loader import load_training_data, TARGET_COL
from src.data.preparation import min_max_scale, convert_str_to_int, undersample
from src.models.weighted_forest.clf import WeightedForest, euclidean_distance
from src.models.base_train_tools import OnlineTrainer, StateSnapshot, calculate_bool_reward, TrainerAIWrapper
from src.game.pong import main as run_pong_game, FPS

VERBOSE = False
CLASS_MAPPING = {'D': 0, 'I': 1, 'U': 2}
MODEL_DIR = "models/wf"


def vprint(message: str) -> None:
    if VERBOSE:
        print(message)

def create_weighted_forest(
    num_features: int,
    num_classes: int,
    learning_decay: float = 0.9,
    accuracy_goal: float = 0.8,
    random_state: int = 42,
    num_start_cells: int = 4,
    similarity_threshold: float = 2.0,
) -> WeightedForest:
    clf = WeightedForest(
        num_features=num_features,
        num_classes=num_classes,
        distance_function=euclidean_distance,
        learning_decay=learning_decay,
        accuracy_goal=accuracy_goal,
        random_seed=random_state,
        num_start_cells=num_start_cells,
        similarity_threshold=similarity_threshold
    )

    return clf

def save(clf, models_dir=MODEL_DIR):
    os.makedirs(models_dir, exist_ok=True)
    model_path = f"{models_dir}/weighted_forest_pong.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Model saved to: {model_path}")

def load(models_dir="models/wf"):
    model_path = f"{models_dir}/weighted_forest_pong.pkl"

    print(f"Loading pretrained model from {model_path}...")
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    vprint("  Model loaded successfully")
    return clf

class WeightedForestOnlineTrainer(OnlineTrainer):
    """Online trainer for Weighted Forest."""

    def __init__(self, model) -> None:
        super().__init__(model, 'weighted_forest', save)
        self.reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
        self.metrics['num_cells'] = len(model.cells)
        self.metrics['accuracy'] = 0.0
        self._correct_predictions = 0
        self._total_predictions = 0

    def predict(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> str:
        """Predict action using forward pass."""
        features = np.array([
            paddle_y,
            ball_x,
            ball_y,
            ball_angle,
            ball_speed,
            ball_y - (paddle_y + 50)
        ])
        pred_int = self.model.forward(features)
        return self.reverse_mapping[pred_int]

    def learn(self, prev_state: StateSnapshot, action: str, new_state: StateSnapshot) -> None:
        """Learn from experience using backward pass."""
        right_decision = calculate_bool_reward(prev_state, new_state)

        self.model.backward(right_decision)

        if right_decision:
            self._correct_predictions += 1
        self._total_predictions += 1

        self.metrics['total_updates'] += 1
        self.metrics['num_cells'] = len(self.model.cells)
        if self._total_predictions > 0:
            self.metrics['accuracy'] = self._correct_predictions / self._total_predictions

def train_weighted_forest_online(
    num_episodes: int = 100,
    max_score_per_episode: int = 5,
    save_interval: int = 10,
    output_dir: str = "models/wf"
) -> None:
    """Run online training for Weighted Forest."""
    print("Running online training for Weighted Forest")

    model = load()

    print(f"Setting up online trainer...")
    
    trainer = WeightedForestOnlineTrainer(model)
    ai_player = TrainerAIWrapper(trainer)
    vprint("  Trainer initialized")

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
            trainer.save_model()

    metrics_df = pd.DataFrame(metrics_log)
    metrics_csv_path = f"{output_dir}/weighted_forest_online_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Metrics saved to {metrics_csv_path}")

    print("Online Training Finished for Weighted Forest\n")
