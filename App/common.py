"""
Key xuyên suốt app là `k8` = 8 ký tự đầu của KPI ID (26 KPI đều
phân biệt được ở 8 ký tự). Full hash chỉ dùng khi đọc file gốc.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import streamlit as st

# Đường dẫn data — App/ nằm cạnh Data/
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "Data" / "train.csv"
TEST_CSV = ROOT / "Data" / "test.csv"

sys.path.insert(0, str(ROOT / "Modeling" / "Code"))
from preprocess import mask_long_gaps, reindex_regular_grid   # noqa: E402

# 16 KPI có lưới thời gian 60s khớp nhau (cohort A + B, 2017), dùng làm
# tập mặc định cho correlation heatmap. Xem memory kpi-time-cohorts.
COHORT_A = ["71595dd7", "7c189dd3", "8bef9af9", "8c892e55",
            "a40b1df8", "affb01ca", "cff6d3c0", "e0770391"]
COHORT_B = ["02e99bd4", "09513ae3", "18fbb1d5", "1c35dbf5",
            "9bd90500", "a5bf5d65", "c58bfcba", "da403e4e"]
ALIGNED_16 = COHORT_A + COHORT_B

# Cặp (cùng service, 2 replica) — anomaly trùng nhau rất cao.
MIRROR_PAIRS = [("8c892e55", "e0770391"), ("7c189dd3", "a40b1df8")]


@st.cache_data(show_spinner="Đang nạp dữ liệu…")
def load_data(source: str = "train") -> pd.DataFrame:
    """Đọc CSV, thêm cột `k8` (id ngắn) và `dt` (datetime). Cache 1 lần."""
    path = TRAIN_CSV if source == "train" else TEST_CSV

    df = pd.read_csv(path)
    df = df.rename(columns={"KPI ID": "kpi"})
    if "label" not in df.columns:      # test.csv không có nhãn -> NaN, tránh crash
        df["label"] = np.nan
    df["k8"] = df["kpi"].str[:8]
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


@st.cache_data(show_spinner=False)
def kpi_meta(source: str = "train") -> pd.DataFrame:
    """Bảng metadata mỗi KPI: bước lấy mẫu, khoảng thời gian, tỷ lệ anomaly, scale."""
    df = load_data(source)
    rows = []
    for k8, g in df.groupby("k8"):
        ts = np.sort(g["timestamp"].values)
        step = int(np.median(np.diff(ts))) if len(ts) > 1 else 0
        rows.append(dict(
            k8=k8, n=len(g), step_s=step,
            start=g["dt"].min(), end=g["dt"].max(),
            days=round((ts[-1] - ts[0]) / 86400, 1),
            anom_rate=float(g["label"].mean()),
            vmin=float(g["value"].min()), vmax=float(g["value"].max()),
        ))
    meta = pd.DataFrame(rows)
    meta["year"] = meta["start"].dt.year
    return meta.sort_values(["step_s", "start", "k8"]).reset_index(drop=True)


@st.cache_data(show_spinner="Đang dựng lưới thời gian đều…")
def grid_series(k8: str, source: str = "train") -> pd.DataFrame:
    """Đưa 1 KPI về lưới thời gian đều — y hệt bước preprocess lúc train.

    Dữ liệu thô thiếu ~2% điểm và CSV không lưu dòng trống, nên cắt thẳng N dòng
    thô sẽ được một cửa sổ trải HƠN N bước thời gian. Lúc train mọi feature được
    tính trên lưới đều (gap ngắn ≤5 điểm nội suy, gap dài đánh dấu `masked`), nên
    lúc serve phải cắt trên đúng lưới đó thì feature mới khớp.

    Trả về DataFrame có: timestamp, value_filled (NaN ở gap dài), masked, label, dt.
    """
    df = load_data(source)
    g = df[df["k8"] == k8][["timestamp", "value", "label", "kpi"]].sort_values("timestamp")
    g = reindex_regular_grid(g, kpi_col="kpi")
    g = mask_long_gaps(g, max_gap_points=5)
    g["dt"] = pd.to_datetime(g["timestamp"], unit="s")
    return g.reset_index(drop=True)


def kpi_label(row: pd.Series) -> str:
    """Nhãn hiển thị trong selectbox — nói rõ cohort qua thời gian + bước mẫu."""
    return (f"{row.k8} · {row.start:%Y-%m-%d}→{row.end:%m-%d} · "
            f"{row.step_s}s · anom {row.anom_rate * 100:.2f}%")


def downsample_keep_anomalies(g: pd.DataFrame, max_points: int = 12000) -> pd.DataFrame:
    """Giảm điểm để vẽ nhanh nhưng giữ mọi điểm label==1 (và điểm liền kề).

    Điểm normal lấy theo stride đều; điểm anomaly luôn được giữ — nếu không,
    downsample sẽ xóa mất chính thứ panel cần cho thấy.
    """
    if len(g) <= max_points:
        return g
    g = g.sort_values("timestamp")
    is_anom = g["label"].values == 1
    keep = is_anom.copy()
    # giữ thêm điểm hai bên anomaly để đường không bị đứt đoạn
    keep[:-1] |= is_anom[1:]
    keep[1:] |= is_anom[:-1]
    stride = max(1, len(g) // max_points)
    keep[::stride] = True
    return g[keep]


@st.cache_data(show_spinner=False)
def corr_matrix(k8_list: tuple[str, ...], differenced: bool, source: str = "train"):
    """Ma trận tương quan trên các KPI đã chọn.

    Trả về (corr_df, n_overlap_rows). Pivot theo timestamp → dropna (chỉ giữ
    hàng mọi KPI cùng có mặt) → tùy chọn diff() → corr. differenced=True loại
    tương quan giả do cùng chu kỳ ngày.
    """
    df = load_data(source)
    sub = df[df["k8"].isin(k8_list)]
    piv = sub.pivot_table(index="timestamp", columns="k8", values="value").dropna()
    if len(piv) < 3:
        return None, len(piv)
    if differenced:
        piv = piv.diff().dropna()
    return piv.corr(), len(piv)

@st.cache_data(show_spinner=False)
def acf_profile(k8: str, source: str = "train", max_days: int = 8):
    """ACF trên lưới thời gian đều (resample + nội suy ngắn để bù gap).

    Gap rất nhiều ở một số KPI (~2.900), nếu tính ACF trên chuỗi thô (theo vị trí)
    thì lag không còn tương ứng thời gian thật -> phải resample trước.
    Trả về dict: acf, step_s, per_day, dom_lag, dom_hours.
    """
    df = load_data(source)
    g = df[df["k8"] == k8].sort_values("timestamp")
    ts = g["timestamp"].values
    step = int(np.median(np.diff(ts))) if len(ts) > 1 else 60
    s = g.set_index("dt")["value"].resample(f"{step}s").mean().interpolate(limit=10)
    v = s.values
    v = v[~np.isnan(v)]
    per_day = max(1, int(round(86400 / step)))

    n = len(v)
    x = v - v.mean()
    f = np.fft.rfft(x, 2 * n)                 # ACF qua FFT (nhanh, chuỗi ~150k điểm)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac /= ac[0]
    ac = ac[:min(n - 1, per_day * max_days)]

    lo, hi = per_day // 4, min(per_day * 2, len(ac))   # bỏ lag nhỏ khi dò đỉnh
    dom = int(np.argmax(ac[lo:hi]) + lo) if hi > lo else per_day
    return dict(acf=ac, step_s=step, per_day=per_day, dom_lag=dom,
                dom_hours=dom * step / 3600)


@st.cache_data(show_spinner=False)
def hour_weekday(k8: str, exclude_anom: bool, source: str = "train"):
    """Ma trận 7×24: giá trị trung bình theo (thứ trong tuần × giờ trong ngày)."""
    df = load_data(source)
    g = df[df["k8"] == k8]
    if exclude_anom:
        g = g[g["label"] != 1]          # NaN (test) != 1 -> True -> giữ nguyên
    return (g.assign(hour=g["dt"].dt.hour, dow=g["dt"].dt.dayofweek)
             .pivot_table(index="dow", columns="hour", values="value", aggfunc="mean")
             .reindex(index=range(7), columns=range(24)))