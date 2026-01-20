import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
from collections import deque

from src.training.reward import StateSnapshot, calculate_reward, generate_label_from_reward
from src.data.preparation import min_max_scale_single
import math
from dataclasses import dataclass

WIDTH = 900
HEIGHT = 600
PADDLE_W = 12
RIGHT_PADDLE_X = WIDTH - 24 - PADDLE_W


class TrainerAIWrapper:
    """Wrapper to make OnlineTrainer compatible with PongAIPlayer interface."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer

    def predict(self, paddle_y: float, ball_x: float, ball_y: float, ball_angle: float, ball_speed: float) -> str:
        """Delegate prediction to trainer."""
        return self.trainer.predict(paddle_y, ball_x, ball_y, ball_angle, ball_speed)

class OnlineTrainer(ABC):
    """Base class for online training wrappers."""

    def __init__(self, model, model_type: str, model_save) -> None:
        self.model = model
        self.model_type = model_type
        self.model_save = model_save
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

    def save_model(self) -> None:
        """Save updated model."""
        self.model_save(self.model)

@dataclass
class StateSnapshot:
    """Capture game state at a point in time."""
    paddle_y: float
    ball_x: float
    ball_y: float
    ball_angle: float
    ball_speed: float
    ball_moving_towards: bool
    paddle_hits: int

def _predict_ball_target_y(ball_x: float, ball_y: float, ball_angle: float,
                          y_min: float = 0, y_max: float = HEIGHT) -> float:
    """Predict where ball will hit the right paddle side, accounting for wall bounces."""
    vx = math.cos(ball_angle)
    vy = math.sin(ball_angle)

    if vx <= 0:
        return ball_y

    dx = RIGHT_PADDLE_X - ball_x
    if dx <= 0:
        return ball_y

    dy = (dx / vx) * vy
    target_y = ball_y + dy

    playable_height = y_max - y_min
    if playable_height <= 0:
        return ball_y

    target_y = target_y - y_min
    bounces = int(abs(target_y) // playable_height)
    remainder = target_y % playable_height if target_y >= 0 else -((-target_y) % playable_height)

    if target_y < 0:
        remainder = -target_y % playable_height
        if (int(-target_y // playable_height) % 2) == 0:
            target_y = y_min + remainder
        else:
            target_y = y_max - remainder
    else:
        remainder = target_y % playable_height
        if (bounces % 2) == 0:
            target_y = y_min + remainder
        else:
            target_y = y_max - remainder

    return max(y_min, min(y_max, target_y))

def calculate_bool_reward(prev_state: StateSnapshot, new_state: StateSnapshot, paddle_h: int = 100) -> float:
    """Calculate reward for state transition using predicted ball target."""
    new_paddle_center = new_state.paddle_y + paddle_h / 2
    prev_paddle_center = prev_state.paddle_y + paddle_h / 2

    if new_state.ball_moving_towards:
        target_y = _predict_ball_target_y(new_state.ball_x, new_state.ball_y, new_state.ball_angle)
    else:
        target_y = HEIGHT / 2
        
    prev_dist_to_target = abs(target_y - prev_paddle_center)
    new_dist_to_target = abs(target_y - new_paddle_center)

    if new_dist_to_target < 50:
        return True
    else:
        if new_dist_to_target - prev_dist_to_target < 0:
            return True
    return False