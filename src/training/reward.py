import math
from dataclasses import dataclass

WIDTH = 900
HEIGHT = 600
PADDLE_W = 12
RIGHT_PADDLE_X = WIDTH - 24 - PADDLE_W


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


def predict_ball_target_y(ball_x: float, ball_y: float, ball_angle: float,
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


def calculate_reward(prev_state: StateSnapshot, action: str, new_state: StateSnapshot, paddle_h: int = 100) -> float:
    """Calculate reward for state transition using predicted ball target."""
    reward = 0.0

    paddle_center = new_state.paddle_y + paddle_h / 2

    if new_state.ball_moving_towards:
        target_y = predict_ball_target_y(
            new_state.ball_x, new_state.ball_y, new_state.ball_angle
        )
        dist_to_target = abs(target_y - paddle_center)
    else:
        dist_to_target = abs(new_state.ball_y - paddle_center)

    if new_state.paddle_hits > prev_state.paddle_hits:
        reward += 10.0

    if new_state.ball_moving_towards:
        proximity_factor = max(0.0, 1.0 - (dist_to_target / 300.0))
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


def generate_label_from_target(paddle_y: float, ball_x: float, ball_y: float,
                                ball_angle: float, ball_moving_towards: bool,
                                paddle_h: int = 100, threshold: float = 10.0) -> str:
    """Generate action label based on predicted ball target position."""
    paddle_center = paddle_y + paddle_h / 2

    if ball_moving_towards:
        target_y = predict_ball_target_y(ball_x, ball_y, ball_angle)
    else:
        target_y = ball_y

    diff = target_y - paddle_center

    if diff < -threshold:
        return "U"
    elif diff > threshold:
        return "D"
    return "I"
