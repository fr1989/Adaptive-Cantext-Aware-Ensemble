import numpy as np
import pandas as pd
from .data import clean_text, load_and_prepare
from .context import ContextDetectorSK
from .learners import BaseLearners
from .agent import QLearningAgent, DQN

def add_context(df: pd.DataFrame, detector: ContextDetectorSK) -> pd.DataFrame:
    """
    به هر ردیف df وکتور زمینه اضافه می‌کند.
    detector بایستی از ContextDetectorSK باشد و قبلاً fit شده باشد.
    """
    cvs = []
    for _, row in df.iterrows():
        cvs.append(detector(row.clean_text, row.date, row.user_id))
    arr = np.vstack(cvs)
    cols = [f"context_{i}" for i in range(arr.shape[1])]
    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(arr, columns=cols)],
        axis=1
    )

__all__ = [
    "load_and_prepare",
    "ContextDetectorSK",
    "BaseLearners",
    "QLearningAgent",
    "add_context"
]
