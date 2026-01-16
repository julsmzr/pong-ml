import glob
import os

import pandas as pd

DATA_DIR = "data"

FEATURE_COLS = ["right_paddle_y", "ball_x", "ball_y", "ball_angle", "ball_speed"]
TARGET_COL = "right_input"


def load_csvs_by_type() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all CSVs and separate by game type."""
    hvh_files = sorted(glob.glob(f"{DATA_DIR}/humanvhuman/RUN_*.csv"))
    hvpc_files = sorted(glob.glob(f"{DATA_DIR}/humanvpc/RUN_*.csv"))

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


def load_training_data_class_balanced(random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data with balanced classes (equal D, I, U samples)."""
    df = load_balanced(random_state=random_state)

    # Balance by class: undersample majority class
    df_D = df[df[TARGET_COL] == 'D']
    df_I = df[df[TARGET_COL] == 'I']
    df_U = df[df[TARGET_COL] == 'U']

    min_count = min(len(df_D), len(df_I), len(df_U))

    df_D_sampled = df_D.sample(n=min_count, random_state=random_state)
    df_I_sampled = df_I.sample(n=min_count, random_state=random_state)
    df_U_sampled = df_U.sample(n=min_count, random_state=random_state)

    df_balanced = pd.concat([df_D_sampled, df_I_sampled, df_U_sampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return get_features_target(df_balanced)
