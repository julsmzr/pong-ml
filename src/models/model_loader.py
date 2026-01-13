"""Model loader utilities for loading trained ML models to play Pong."""
import pickle
from pathlib import Path
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
        features = np.array([[paddle_y, ball_x, ball_y, ball_angle, ball_speed]])
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
        raise FileNotFoundError(
            f"Decision Tree model not found at {model_file}. "
            "Train it first: python3 src/models/decision_tree/offline_train.py"
        )

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
