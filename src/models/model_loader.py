import pickle
from pathlib import Path
import pandas as pd
import numpy as np


class PongAIPlayer:
    """Wrapper class for AI models to play Pong."""

    def __init__(self, model_path: str | Path, metadata_path: str | Path | None = None) -> None:
        """Load a trained model from pickle file."""
        self.model_path = Path(model_path)
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.metadata = None
        if metadata_path:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

    def predict(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> str:
        """Predict next action: 'U', 'D', or 'I'."""
        y_diff = ball_y - (paddle_y + 50)  # 50 is PADDLE_H/2

        # Check model type
        if hasattr(self.model, 'predict_one'):
            # River model: uses predict_one with dictionary input
            features = {
                'right_paddle_y': paddle_y,
                'ball_x': ball_x,
                'ball_y': ball_y,
                'ball_angle': ball_angle,
                'ball_speed': ball_speed,
                'y_diff': y_diff
            }

            prediction = self.model.predict_one(features)

        elif hasattr(self.model, 'forward'):
            # Weighted Forest: uses forward with numpy array and returns integer
            features = np.array([paddle_y, ball_x, ball_y, ball_angle, ball_speed, y_diff])
            pred_int = self.model.forward(features)

            # Convert integer to string label
            class_mapping = self.metadata.get('class_mapping', {0: 'D', 1: 'I', 2: 'U'})
            reverse_mapping = {v: k for k, v in class_mapping.items()}

            prediction = reverse_mapping[pred_int]

        else:
            # Sklearn model: uses predict with DataFrame input
            features = pd.DataFrame(
                [[paddle_y, ball_x, ball_y, ball_angle, ball_speed]],
                columns=['right_paddle_y', 'ball_x', 'ball_y', 'ball_angle', 'ball_speed']
            )
            features['y_diff'] = y_diff

            prediction = self.model.predict(features)[0]

        return prediction

    def get_info(self) -> dict:
        """Get model metadata information."""
        if self.metadata:
            return self.metadata
        return {'model_path': str(self.model_path)}


def load_decision_tree_model(models_dir: str = "models") -> PongAIPlayer:
    """Load the Decision Tree model."""
    models_path = Path(models_dir)
    model_file = models_path / "decision_tree_pong.pkl"
    metadata_file = models_path / "decision_tree_metadata.pkl"

    if not model_file.exists():
        raise FileNotFoundError(f"Decision Tree model not found at {model_file}. ")

    return PongAIPlayer(model_file, metadata_file if metadata_file.exists() else None)


def load_hoeffding_tree_model(models_dir: str = "models") -> PongAIPlayer:
    """Load the Hoeffding Tree model."""
    models_path = Path(models_dir)
    model_file = models_path / "hoeffding_tree_pong.pkl"
    metadata_file = models_path / "hoeffding_tree_metadata.pkl"

    if not model_file.exists():
        raise FileNotFoundError(f"Hoeffding Tree model not found at {model_file}. ")

    return PongAIPlayer(model_file, metadata_file if metadata_file.exists() else None)


def load_weighted_forest_model(models_dir: str = "models") -> PongAIPlayer:
    """Load the Weighted Forest model."""
    models_path = Path(models_dir)
    model_file = models_path / "weighted_forest_pong.pkl"
    metadata_file = models_path / "weighted_forest_metadata.pkl"

    if not model_file.exists():
        raise FileNotFoundError(f"Weighted Forest model not found at {model_file}. ")

    return PongAIPlayer(model_file, metadata_file if metadata_file.exists() else None)


if __name__ == "__main__":
    try:
        ai = load_decision_tree_model()
        print("Model loaded successfully!")
        print("\nModel info:")
        for key, value in ai.get_info().items():
            print(f"  {key}: {value}")

        print("\nTest prediction:")
        action = ai.predict(250.0, 450.0, 300.0, 0.0, 360)
        print(f"  Input: paddle_y=250, ball_x=450, ball_y=300, angle=0, speed=360")
        print(f"  Predicted action: {action}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
