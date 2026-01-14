import glob
import os

import pandas as pd

DATA_DIR = "data"

FEATURE_COLS = ["right_paddle_y", "ball_x", "ball_y", "ball_angle", "ball_speed"]
TARGET_COL = "right_input"


def load_csvs_by_type() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all CSVs and separate by game type."""
    hvh_files = sorted(glob.glob(os.path.join(DATA_DIR, "hvhuman_*.csv")))
    hvpc_files = sorted(glob.glob(os.path.join(DATA_DIR, "hvpc_*.csv")))

    hvh_df = pd.concat([pd.read_csv(f) for f in hvh_files], ignore_index=True) if hvh_files else pd.DataFrame()
    hvpc_df = pd.concat([pd.read_csv(f) for f in hvpc_files], ignore_index=True) if hvpc_files else pd.DataFrame()

    return hvh_df, hvpc_df


def load_balanced(random_state: int = 42) -> pd.DataFrame:
    """Load and merge data with 50/50 balance between hvhuman and hvpc."""
    hvh_df, hvpc_df = load_csvs_by_type()

    if hvh_df.empty or hvpc_df.empty:
        raise ValueError("Need both hvhuman and hvpc data files")

    # Balance by undersampling the larger set
    min_size = min(len(hvh_df), len(hvpc_df))
    hvh_sampled = hvh_df.sample(n=min_size, random_state=random_state)
    hvpc_sampled = hvpc_df.sample(n=min_size, random_state=random_state)

    merged = pd.concat([hvh_sampled, hvpc_sampled], ignore_index=True)
    return merged.sample(frac=1, random_state=random_state).reset_index(drop=True)  # shuffle


def get_features_target(df: pd.DataFrame, add_derived_features: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()

    if add_derived_features:
        X['y_diff'] = X['ball_y'] - (X['right_paddle_y'] + 50)  # 50 is PADDLE_H/2

    y = df[TARGET_COL]
    return X, y


def load_training_data(random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    df = load_balanced(random_state=random_state)
    return get_features_target(df)


if __name__ == "__main__":
    # Statistics
    hvh, hvpc = load_csvs_by_type()
    print(f"HvH samples: {len(hvh)}")
    print(f"HvPC samples: {len(hvpc)}")

    X, y = load_training_data()
    print(f"\nBalanced training set: {len(X)} samples")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")
