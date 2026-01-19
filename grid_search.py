import numpy as np

from src.data.loader import load_training_data, TARGET_COL
from src.data.preparation import min_max_scale, convert_str_to_int, undersample
from src.models.weighted_forest.clf import WeightedForest, euclidean_distance
from src.training.online_trainer import WeightedForestOnlineTrainer, TrainerAIWrapper
from src.game.pong import main as run_pong_game, FPS

def perform_grid_search(random_state=42):
    print("Loading data...")
    X, y = load_training_data(random_state=random_state)

    X_np = X.to_numpy()
    y_np = y.to_numpy()

    print("Preparing data...")
    X_np, feature_min, feature_max = min_max_scale(X_np)
    y_np, class_mapping = convert_str_to_int(y_np)
    X_np, y_np = undersample(X_np, y_np, random_seed=random_state)
    scaler = {'feature_min': feature_min, 'feature_max': feature_max}

    # Definiere die Parameter
    num_cells = np.arange(5, 11, 2)  # Werte von 1 bis 10, 2er Schritte
    similarity_threshold = np.arange(0.5, 2.6, 0.25)  # Werte von 0.5 bis 2.5, 0.25er Schritte
    epochs = np.arange(3, 6, 2)  # Werte von 1 bis 5, 2er Schritte

    # Erstelle ein Gitter für die Parameter
    grid = np.array(np.meshgrid(num_cells, similarity_threshold, epochs)).T.reshape(-1, 3)
    print(f"Number Tests: {grid.shape[0]}")

    results = np.zeros(shape=(grid.shape[0], 2))

    for idx in range(grid.shape[0]):
        print(f"Test hyperparameter: Epoch: {grid[idx, 2]}, Num Cells: {grid[idx,0]}, Similarity Distance: {grid[idx,1]}")

        clf = WeightedForest(
            num_features=X_np.shape[1],
            num_classes=len(class_mapping),
            distance_function=euclidean_distance,
            learning_decay=0.95,
            accuracy_goal=0.65,
            initializer_low=0,
            initializer_high=1,
            random_seed=random_state,
            num_start_cells=int(grid[idx,0]),
            similarity_threshold=grid[idx,1]
        )

        clf.fit(X_np, y_np, epochs=int(grid[idx,2]))

        trainer = WeightedForestOnlineTrainer(clf, class_mapping, scaler=scaler)
        ai_player = TrainerAIWrapper(trainer)

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

        results[idx] = np.array([episode_metrics['survival_seconds'], episode_metrics['hits']])

        np.save("data/hyperparameter_tuning_results.npy", results)

        print(f"Test {idx+1}: Survival: {episode_metrics['survival_seconds']:.2f}s, Hits: {episode_metrics['hits']}")

if __name__ == "__main__":
    perform_grid_search()