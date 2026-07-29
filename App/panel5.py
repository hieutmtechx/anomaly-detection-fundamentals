"""Panel 5 data layer — Root Cause.

build_incidents(gap_min)  -> gom alert gần nhau thành "sự cố"
   (top-3 root cause + SHAP)

Một "sự cố" = cụm alert (của một hoặc nhiều KPI) cách nhau ≤ gap_min phút.
Alert cách nhau xa hơn gap_min ⇒ mở sự cố mới. Không phân biệt KPI: một sự cố
diện rộng làm nhiều KPI cùng kêu trong cùng khoảng thời gian.
"""
import numpy as np
import pandas as pd
import streamlit as st

from panel4 import build_alerts


@st.cache_data(show_spinner=False)
def build_incidents(gap_min: int = 30) -> pd.DataFrame:
    """Gom alert (mọi KPI, xếp theo thời gian) thành sự cố theo khoảng lặng gap_min.

    Trả DataFrame mỗi dòng một sự cố: start, end, duration_min, n_alerts, n_kpis,
    kpis (chuỗi liệt kê), n_tp, n_fp. Sắp theo thời gian bắt đầu (timeline).
    """
    a = build_alerts().sort_values("timestamp").reset_index(drop=True)
    ts = a["timestamp"].values

    # điểm cắt = chỗ hai alert liên tiếp cách nhau > gap_min phút
    brk = np.where(np.diff(ts) > gap_min * 60)[0]
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, len(ts) - 1]

    rows = []
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        seg = a.iloc[s:e + 1]
        rows.append(dict(
            incident=i,
            start=seg["dt"].iloc[0],
            end=seg["dt"].iloc[-1],
            duration_min=(ts[e] - ts[s]) / 60,
            n_alerts=len(seg),
            n_kpis=seg["kpi"].nunique(),
            kpis=", ".join(sorted(seg["kpi"].unique())),
            n_tp=int((seg["y"] == 1).sum()),
            n_fp=int((seg["y"] == 0).sum()),
            _s=s, _e=e,          # chỉ số dòng đầu/cuối trong bảng alert (dùng cho root_cause)
        ))
    return pd.DataFrame(rows)


def _norm01(s: pd.Series) -> pd.Series:
    """Chuẩn hoá về [0,1] trong nội bộ sự cố. Nếu mọi giá trị bằng nhau -> 1.0."""
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(1.0, index=s.index)


@st.cache_data(show_spinner=False)
def root_cause(gap_min: int, incident_id: int) -> pd.DataFrame:
    """Xếp hạng các KPI trong một sự cố theo mức nghi là nguyên nhân gốc.

    Gộp 3 dấu hiệu, mỗi cái chuẩn hoá về [0,1] trong sự cố:
      earliness   — kêu càng sớm càng cao
      severity    — peak_margin: vượt ngưỡng càng mạnh càng cao
      reliability — tỷ lệ alert đúng (TP/tổng): lọc KPI toàn báo động giả
    root_score = trung bình 3 dấu hiệu (trọng số bằng nhau). Sắp giảm dần.

    Đây không phải quan hệ nhân quả thật: dữ liệu ẩn danh không cho
    biết service nào gọi service nào. "Gốc" = ứng viên khả nghi nhất, không phải
    kết luận. Trọng số bằng nhau là lựa chọn mặc định, có thể chỉnh.
    """
    a = build_alerts().sort_values("timestamp").reset_index(drop=True)
    inc = build_incidents(gap_min)
    row = inc[inc["incident"] == incident_id].iloc[0]
    seg = a.iloc[int(row["_s"]):int(row["_e"]) + 1]      # đúng các alert của sự cố

    g = seg.groupby("kpi").agg(
        first=("dt", "min"),
        first_ts=("timestamp", "min"),
        peak_margin=("margin", "max"),
        n_alerts=("margin", "size"),
        n_tp=("y", "sum"),
    )
    g["n_fp"] = g["n_alerts"] - g["n_tp"]
    g["rank_time"] = g["first"].rank(method="min").astype(int)   # 1 = kêu đầu tiên

    # 3 dấu hiệu chuẩn hoá [0,1]
    g["earliness"] = 1 - _norm01(g["first_ts"])          # kêu sớm -> gần 1
    g["severity"] = _norm01(g["peak_margin"])            # mạnh   -> gần 1
    g["reliability"] = g["n_tp"] / g["n_alerts"]         # tỷ lệ đúng (đã sẵn 0..1)
    g["root_score"] = g[["earliness", "severity", "reliability"]].mean(axis=1)

    return (g.drop(columns="first_ts")
            .sort_values("root_score", ascending=False).reset_index())
