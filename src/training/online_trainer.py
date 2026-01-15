import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
from collections import deque

from src.training.reward import StateSnapshot, calculate_reward, generate_label_from_reward
from src.data.preparation import min_max_scale_single


class TrainerAIWrapper:
    """Wrapper to make OnlineTrainer compatible with PongAIPlayer interface."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer

    def predict(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> str:
        """Delegate prediction to trainer."""
        return self.trainer.predict(paddle_y, ball_x, ball_y, ball_angle, ball_speed)


class OnlineTrainer(ABC):
    """Base class for online training wrappers."""

    def __init__(self, model, model_type: str) -> None:
        self.model = model
        self.model_type = model_type
        self.experience_buffer = deque(maxlen=10)
        self.metrics = {
            'total_updates': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0
        }

    @abstractmethod
    def predict(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> str:
        """Get action prediction."""
        pass

    @abstractmethod
    def learn(self, prev_state: StateSnapshot, action: str, new_state: StateSnapshot) -> None:
        """Update model with experience."""
        pass

    def get_metrics(self) -> dict:
        """Return current learning metrics."""
        if self.metrics['total_updates'] > 0:
            self.metrics['avg_reward'] = self.metrics['total_reward'] / self.metrics['total_updates']
        return self.metrics.copy()

    def save_model(self, path: str) -> None:
        """Save updated model."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


class HoeffdingOnlineTrainer(OnlineTrainer):
    """Online trainer for Hoeffding Tree using River."""

    def __init__(self, model) -> None:
        super().__init__(model, 'hoeffding_tree')
        self.metrics['progressive_accuracy'] = 0.0
        self._correct_predictions = 0
        self._total_predictions = 0

    def predict(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> str:
        """Predict action using River's predict_one."""
        y_diff = ball_y - (paddle_y + 50)
        features = {
            'right_paddle_y': paddle_y,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'ball_angle': ball_angle,
            'ball_speed': ball_speed,
            'y_diff': y_diff
        }
        return self.model.predict_one(features)

    def learn(self, prev_state: StateSnapshot, action: str, new_state: StateSnapshot) -> None:
        """Learn from experience using learn_one."""
        reward = calculate_reward(prev_state, action, new_state)

        label = generate_label_from_reward(new_state.paddle_y, new_state.ball_y, reward)

        y_diff = new_state.ball_y - (new_state.paddle_y + 50)
        features = {
            'right_paddle_y': new_state.paddle_y,
            'ball_x': new_state.ball_x,
            'ball_y': new_state.ball_y,
            'ball_angle': new_state.ball_angle,
            'ball_speed': new_state.ball_speed,
            'y_diff': y_diff
        }

        pred = self.model.predict_one(features)
        self.model.learn_one(features, label)

        if pred == label:
            self._correct_predictions += 1
        self._total_predictions += 1

        self.metrics['total_updates'] += 1
        self.metrics['total_reward'] += reward
        if self._total_predictions > 0:
            self.metrics['progressive_accuracy'] = self._correct_predictions / self._total_predictions


class WeightedForestOnlineTrainer(OnlineTrainer):
    """Online trainer for Weighted Forest."""

    def __init__(self, model, class_mapping: dict[str, int], scaler: dict = None) -> None:
        super().__init__(model, 'weighted_forest')
        self.class_mapping = class_mapping
        self.reverse_mapping = {v: k for k, v in class_mapping.items()}
        self.scaler = scaler 
        self.metrics['num_cells'] = len(model.cells)
        self.metrics['accuracy'] = 0.0
        self._correct_predictions = 0
        self._total_predictions = 0

    def _build_features(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> np.ndarray:
        """Build feature vector with optional scaling."""
        y_diff = ball_y - (paddle_y + 50)
        features = np.array([paddle_y, ball_x, ball_y, ball_angle, ball_speed, y_diff])

        if self.scaler is not None:
            features = min_max_scale_single(
                features,
                self.scaler['feature_min'],
                self.scaler['feature_max']
            )

        return features

    def predict(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> str:
        """Predict action using forward pass."""
        features = self._build_features(paddle_y, ball_x, ball_y, ball_angle, ball_speed)
        pred_int = self.model.forward(features)
        return self.reverse_mapping[pred_int]

    def learn(self, prev_state: StateSnapshot, action: str, new_state: StateSnapshot) -> None:
        """Learn from experience using backward pass."""
        reward = calculate_reward(prev_state, action, new_state)

        right_decision = reward > 0.0
        self.model.backward(right_decision)

        if right_decision:
            self._correct_predictions += 1
        self._total_predictions += 1

        self.metrics['total_updates'] += 1
        self.metrics['total_reward'] += reward
        self.metrics['num_cells'] = len(self.model.cells)
        if self._total_predictions > 0:
            self.metrics['accuracy'] = self._correct_predictions / self._total_predictions
