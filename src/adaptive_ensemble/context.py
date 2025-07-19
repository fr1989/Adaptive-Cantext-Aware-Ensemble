import numpy as np
import pandas as pd

# اگر detector در ماژول دیگریست، مسیر صحیح import را اصلاح کن
from .data import detector  

def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    به هر ردیف از df وکتور زمینه می‌افزاید و
    DataFrame جدید با ستون‌های context_i بازمی‌گرداند.
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
