import math
from dataclasses import dataclass


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


def calculate_reward(prev_state: StateSnapshot, action: str, new_state: StateSnapshot, paddle_h: int = 100) -> float:
    """Calculate reward for state transition with immediate feedback."""
    reward = 0.0

    paddle_center = new_state.paddle_y + paddle_h / 2
    ball_to_paddle_dist = abs(new_state.ball_y - paddle_center)

    if new_state.paddle_hits > prev_state.paddle_hits:
        reward += 10.0

    if new_state.ball_moving_towards:
        proximity_factor = max(0.0, 1.0 - (ball_to_paddle_dist / 300.0))
        reward += proximity_factor * 1.0

    if action == "I":
        reward -= 0.1

    reward += 0.01

    return reward


def calculate_sparse_reward(prev_state: StateSnapshot, new_state: StateSnapshot, ball_missed: bool) -> float:
    """Calculate sparse reward (only hits and misses)."""
    if new_state.paddle_hits > prev_state.paddle_hits:
        return 1.0
    elif ball_missed:
        return -1.0
    return 0.0


def generate_label_from_reward(paddle_y: float, ball_y: float, reward: float, paddle_h: int = 100) -> str:
    """Convert reward signal to action label for supervised learning."""
    if reward > 0.5:
        paddle_center = paddle_y + paddle_h / 2
        diff = ball_y - paddle_center
        if diff < -10:
            return "U"
        elif diff > 10:
            return "D"
        return "I"

    return "I"
