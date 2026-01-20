from src.models.weighted_forest.tools import create_weighted_forest, train_weighted_forest_online, save

clf = create_weighted_forest(
    num_features=6,
    num_classes=3,
    accuracy_goal=0.7,
    random_state=42,
    num_start_cells=10,
    similarity_threshold=1.5
)

##save(clf)

train_weighted_forest_online(
    num_episodes=200,
    max_score_per_episode=10,
    save_interval=2
)