import numpy as np

from src.data.loader import load_training_data, TARGET_COL
from src.data.preparation import min_max_scale, convert_str_to_int, undersample
from src.models.weighted_forest.clf import WeightedForest, euclidean_distance
from src.training.online_trainer import WeightedForestOnlineTrainer, TrainerAIWrapper
from src.game.pong import main as run_pong_game, FPS

def perform_data_prep(random_state=42):
    X, y = load_training_data(random_state=random_state)

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    X_np, feature_min, feature_max = min_max_scale(X_np)
    y_np, class_mapping = convert_str_to_int(y_np)
    X_np, y_np = undersample(X_np, y_np, random_seed=random_state)
    scaler = {'feature_min': feature_min, 'feature_max': feature_max}

    return X_np, y_np, scaler, class_mapping

def create_trainer_and_player(X, y, class_mapping, scaler, random_state=42, epochs=3, num_start_cells=4, similarity_threshold=2):
    clf = WeightedForest(
        num_features=X.shape[1],
        num_classes=len(class_mapping),
        distance_function=euclidean_distance,
        learning_decay=0.95,
        accuracy_goal=0.65,
        initializer_low=0,
        initializer_high=1,
        random_seed=random_state,
        num_start_cells=num_start_cells,
        similarity_threshold=similarity_threshold
    )

    clf.fit(X, y, epochs=epochs)

    trainer = WeightedForestOnlineTrainer(clf, class_mapping, scaler=scaler)
    ai_player = TrainerAIWrapper(trainer)

    return trainer, ai_player

def test_online(trainer, ai_player):
    final_state = run_pong_game(
        right_ai=ai_player,
        right_mode='wf',
        left_mode='pc',
        max_score=10,
        online_trainer=trainer,
        learning_mode=True
    )

    episode_metrics = trainer.get_metrics()
    episode_metrics['survival_seconds'] = final_state.frame_count / FPS
    episode_metrics['hits'] = final_state.right_paddle_hits

    return episode_metrics['survival_seconds'], episode_metrics['hits']