import pandas as pd
import numpy as np
from adaptive_ensemble.context import add_context

def dummy_detector(text, date, user_id):
    # برای تست: همیشه یک وکتور ثابت برمی‌گرداند
    return np.array([1.0, 2.0, 3.0])

def test_add_context(monkeypatch):
    df = pd.DataFrame({
        'clean_text': ['a', 'b'],
        'date': ['2025-01-01','2025-01-02'],
        'user_id': [1, 2]
    })
    # جایگزین کردن تابع اصلی detector با نسخه تستی
    monkeypatch.setattr('adaptive_ensemble.context.detector', dummy_detector)
    out = add_context(df)
    # باید سه ستون context_0,1,2 اضافه شده باشد
    assert all(col in out.columns for col in ['context_0','context_1','context_2'])
    # مقادیر درست هستند
    assert (out[['context_0','context_1','context_2']].values == np.array([[1,2,3],[1,2,3]])).all()
