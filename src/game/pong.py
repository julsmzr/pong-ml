from __future__ import annotations

# Hide welcome message from PyGame
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import math
from dataclasses import dataclass
import pygame
import csv
from pathlib import Path


WIDTH, HEIGHT = 900, 600
FPS = 60

PADDLE_W, PADDLE_H = 12, 100
PADDLE_SPEED = 420  # px/s

BALL_SIZE = 12
BALL_SPEED = 360 
ball_speed_multiplier = 1.0 

CENTER_LINE_GAP = 18

BG_COLOR = (16, 18, 22)
FG_COLOR = (245, 246, 248)

# Data collection setup
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
run_number = 0
frame_id = 0

def get_player_input(keys) -> tuple[str, str]:
    left_input = "I"
    if keys[pygame.K_q]:
        left_input = "U"
    elif keys[pygame.K_a]:
        left_input = "D"
        
    right_input = "I" 
    if keys[pygame.K_UP]:
        right_input = "U"
    elif keys[pygame.K_DOWN]:
        right_input = "D"
        
    return left_input, right_input

def write_frame_data(csv_writer, frame_id, left_input, right_input):
    csv_writer.writerow([
        frame_id,
        left_input,
        right_input,
        state.left_paddle_y,
        state.right_paddle_y,
        state.ball_pos.x,
        state.ball_pos.y,
        state.ball_angle,
        BALL_SPEED
    ])

@dataclass
class GameState:
    left_paddle_y: float
    right_paddle_y: float
    ball_pos: pygame.Vector2
    ball_vel: pygame.Vector2
    ball_angle: float  # radians, computed from velocity
    left_score: int
    right_score: int

    def update_angle(self) -> None:
        # atan2(y, x) -> radians; keep in (-pi, pi]
        self.ball_angle = math.atan2(self.ball_vel.y, self.ball_vel.x)

state = GameState(
    left_paddle_y=(HEIGHT - PADDLE_H) / 2,
    right_paddle_y=(HEIGHT - PADDLE_H) / 2,
    ball_pos=pygame.Vector2(WIDTH / 2, HEIGHT / 2),
    ball_vel=pygame.Vector2(BALL_SPEED, 0),  # starts to the right
    ball_angle=0.0,
    left_score=0,
    right_score=0,
)


def _reset_ball(direction: int = 1) -> None:
    """Center the ball and launch horizontally. direction: +1 to right, -1 to left."""
    global ball_speed_multiplier
    ball_speed_multiplier = 1.0
    state.ball_pos.update(WIDTH / 2, HEIGHT / 2)
    state.ball_vel.update(BALL_SPEED * direction, 0)
    state.update_angle()



def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# Run Loop
def main() -> None:
    global ball_speed_multiplier

    pygame.init()
    pygame.display.set_caption("Pong (Q/A vs ↑/↓)")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 56)

    # Data collection setup
    run_number = 0
    while (DATA_DIR / f"run_{run_number:04d}.csv").exists():
        run_number += 1
        
    csv_file = open(DATA_DIR / f"run_{run_number:04d}.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "left_input", "right_input", "left_paddle_y", "right_paddle_y", 
                        "ball_x", "ball_y", "ball_angle", "ball_speed"])
    frame_id = 0

    _reset_ball(direction=1)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since last frame

        # Events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

         # Input
        keys = pygame.key.get_pressed()
        left_input, right_input = get_player_input(keys)
        
        # Use the input for movement
        if left_input == "U":
            state.left_paddle_y -= PADDLE_SPEED * dt
        elif left_input == "D":
            state.left_paddle_y += PADDLE_SPEED * dt
            
        if right_input == "U":
            state.right_paddle_y -= PADDLE_SPEED * dt
        elif right_input == "D":
            state.right_paddle_y += PADDLE_SPEED * dt

        state.left_paddle_y = _clamp(state.left_paddle_y, 0, HEIGHT - PADDLE_H)
        state.right_paddle_y = _clamp(state.right_paddle_y, 0, HEIGHT - PADDLE_H)

        # Move Ball
        state.ball_pos += state.ball_vel * dt

        # Top/bottom walls
        if state.ball_pos.y <= 0:
            state.ball_pos.y = 0
            state.ball_vel.y *= -1
        elif state.ball_pos.y + BALL_SIZE >= HEIGHT:
            state.ball_pos.y = HEIGHT - BALL_SIZE
            state.ball_vel.y *= -1

        # Paddles rects
        left_rect = pygame.Rect(24, int(state.left_paddle_y), PADDLE_W, PADDLE_H)
        right_rect = pygame.Rect(WIDTH - 24 - PADDLE_W, int(state.right_paddle_y), PADDLE_W, PADDLE_H)
        ball_rect = pygame.Rect(int(state.ball_pos.x), int(state.ball_pos.y), BALL_SIZE, BALL_SIZE)

        # Left paddle collision
        if ball_rect.colliderect(left_rect) and state.ball_vel.x < 0:
            hit_rel = (state.ball_pos.y + BALL_SIZE / 2) - (left_rect.centery)
            norm = _clamp(hit_rel / (PADDLE_H / 2), -1.0, 1.0)  # -1 top, +1 bottom
            angle = norm * (math.pi / 3)  # spread up to ±60°

            # Rebuild velocity at current speed, heading to the right
            state.ball_vel.from_polar((BALL_SPEED * ball_speed_multiplier, math.degrees(angle)))
            state.ball_vel.x = abs(state.ball_vel.x)  # ensure right

            # Nudge out of the paddle to avoid sticking
            state.ball_pos.x = left_rect.right
            ball_speed_multiplier *= 1.1

        # Right paddle collision
        if ball_rect.colliderect(right_rect) and state.ball_vel.x > 0:
            hit_rel = (state.ball_pos.y + BALL_SIZE / 2) - (right_rect.centery)
            norm = _clamp(hit_rel / (PADDLE_H / 2), -1.0, 1.0)
            angle = norm * (math.pi / 3)

            state.ball_vel.from_polar((BALL_SPEED * ball_speed_multiplier, math.degrees(angle)))
            state.ball_vel.x = -abs(state.ball_vel.x)  # ensure left

            state.ball_pos.x = right_rect.left - BALL_SIZE
            ball_speed_multiplier *= 1.1 

        # Scoring
        if state.ball_pos.x + BALL_SIZE < 0:
            state.right_score += 1
            _reset_ball(direction=1)
        elif state.ball_pos.x > WIDTH:
            state.left_score += 1
            _reset_ball(direction=-1)

        if state.ball_vel.length_squared() != 0:
            state.ball_vel.scale_to_length(BALL_SPEED * ball_speed_multiplier)

        state.update_angle()

        # Render
        screen.fill(BG_COLOR)

        # Draw Center line
        dash_h = 12
        y = 0
        cx = WIDTH // 2
        while y < HEIGHT:
            pygame.draw.rect(screen, FG_COLOR, (cx - 2, y, 4, dash_h))
            y += dash_h + CENTER_LINE_GAP

        # Draw Paddles & ball
        pygame.draw.rect(screen, FG_COLOR, left_rect, border_radius=3)
        pygame.draw.rect(screen, FG_COLOR, right_rect, border_radius=3)
        pygame.draw.rect(screen, FG_COLOR, ball_rect, border_radius=6)

        # Draw Scores
        ls = font.render(str(state.left_score), True, FG_COLOR)
        rs = font.render(str(state.right_score), True, FG_COLOR)
        screen.blit(ls, (WIDTH * 0.25 - ls.get_width() / 2, 24))
        screen.blit(rs, (WIDTH * 0.75 - rs.get_width() / 2, 24))

        # Write frame data
        write_frame_data(csv_writer, frame_id, left_input, right_input)
        frame_id += 1

        pygame.display.flip()

    csv_file.close()
    pygame.quit()


if __name__ == "__main__":
    main()
