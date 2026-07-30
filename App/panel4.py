"""Panel 4: System Health.

build_alerts()  -> nhật ký alert
compute_drift() -> độ trôi mean/std vs baseline

Alert lấy từ score model IF-feature đã chấm sẵn trên tập test
(scores_IF_feat.parquet) + ngưỡng b_thr_pw (chọn trên val). 
Vì parquet kèm nhãn thật `y` nên mỗi alert biết được đúng (TP) hay báo động giả (FP).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from common import load_data

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "Modeling" / "Artifacts"
IF_CFG = ROOT / "Modeling" / "ML" / "IsolationForest" / "if_per_kpi_config.json"

sys.path.insert(0, str(ROOT / "Modeling" / "Code"))   # dùng lại cách chia của lúc train
from eval_protocol import time_split_per_kpi           # noqa: E402


@st.cache_data(show_spinner=False)
def thresholds() -> dict:
    """Ngưỡng b_thr_pw của từng KPI (route feature, chọn max-F1 trên val)."""
    return {r["kpi"]: float(r["b_thr_pw"])
            for r in json.load(open(IF_CFG, encoding="utf-8"))}


@st.cache_data(show_spinner="Đang dựng nhật ký alert…")
def build_alerts() -> pd.DataFrame:
    """Mọi điểm test có score >= ngưỡng của KPI đó = một alert.

    Trả DataFrame cột: kpi, timestamp, score, thr, margin (score-thr),
    y (nhãn thật), outcome ('đúng (TP)' | 'báo động giả (FP)'), sắp theo thời gian.
    """
    s = pd.read_parquet(ART / "scores_IF_feat.parquet")
    s["thr"] = s["kpi"].map(thresholds())
    a = s[s["score"] >= s["thr"]].copy()
    a["margin"] = a["score"] - a["thr"]
    a["dt"] = pd.to_datetime(a["timestamp"], unit="s")
    a["outcome"] = a["y"].map({1: "đúng (TP)", 0: "báo động giả (FP)"})
    return a.sort_values("dt").reset_index(drop=True)


@st.cache_data(show_spinner="Đang tính drift…")
def compute_drift(min_n: int = 100) -> pd.DataFrame:
    """Độ drift của từng KPI: baseline (train 60%) so với hiện tại (test 20%).

    drift_mean = (mean_test − mean_train) / std_train  — nền dịch mấy độ lệch chuẩn
    std_ratio  = std_test / std_train                   — dao động gấp mấy lần

    Drift thuần thống kê trên giá trị thô, không cần model — nên phủ được cả KPI
    không có model (khác alert log chỉ có 17-18 KPI). Bỏ KPI có < min_n điểm mỗi phần.
    Sắp theo |drift_mean| giảm dần (trôi mạnh nhất lên đầu).
    """
    df = load_data("train")
    rows = []
    for k8, g in df.groupby("k8"):
        g = time_split_per_kpi(g, train_frac=0.6, val_frac=0.2, kpi_col="k8")
        base = g.loc[g["split"] == "train", "value"]
        now = g.loc[g["split"] == "test", "value"]
        if len(base) < min_n or len(now) < min_n:
            continue
        mb, sb = float(base.mean()), float(base.std())
        mn, sn = float(now.mean()), float(now.std())
        rows.append(dict(
            kpi=k8, mean_base=mb, mean_now=mn, std_base=sb, std_now=sn,
            drift_mean=(mn - mb) / sb if sb else 0.0,
            std_ratio=sn / sb if sb else 0.0))
    d = pd.DataFrame(rows)
    d["abs_drift"] = d["drift_mean"].abs()
    return d.sort_values("abs_drift", ascending=False).reset_index(drop=True)
