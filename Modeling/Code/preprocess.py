import numpy as np
import pandas as pd

def infer_step_s(timestamps):
    """Sampling interval phổ biến (giây) = mode của hiệu các timestamp liên tiếp."""
    d = np.diff(np.sort(np.asarray(timestamps)))
    return int(pd.Series(d).mode().iloc[0])


def reindex_regular_grid(g, step_s=None, time_col="timestamp",
                         value_col="value", label_col="label", kpi_col="kpi"):
    """Đưa 1 KPI về grid đều [min, max] bước step_s. Điểm thiếu -> value NaN, filled=True.
    label của điểm chèn = 0 (mặc định coi là bình thường), filled đánh dấu để loại nếu cần."""
    g = g.sort_values(time_col).copy()
    if step_s is None:
        step_s = infer_step_s(g[time_col].values)
    full = np.arange(int(g[time_col].min()), int(g[time_col].max()) + step_s, step_s)
    out = pd.DataFrame({time_col: full})
    out = out.merge(g, on=time_col, how="left")
    out["filled"] = out[value_col].isna()
    if kpi_col in g.columns:
        out[kpi_col] = g[kpi_col].iloc[0]
    if label_col in out.columns:
        out[label_col] = out[label_col].fillna(0).astype(int)
    out.attrs["step_s"] = step_s
    return out


def mask_long_gaps(g, max_gap_points=5, value_col="value", interpolate=True):
    """Phân loại các đoạn 'filled' liên tiếp theo độ dài:
       - <= max_gap_points  -> gap NGẮN: nội suy tuyến tính (nếu interpolate=True)
       - >  max_gap_points  -> gap DÀI : masked=True, KHÔNG nội suy (giữ NaN)
    Thêm 'segment': id đoạn liên tục, tăng mỗi khi qua một gap dài (để không bắc cầu)."""
    g = g.copy().reset_index(drop=True)
    filled = g["filled"].values
    masked = np.zeros(len(g), dtype=bool)

    # gom các run 'filled' liên tiếp
    i = 0; n = len(g)
    while i < n:
        if filled[i]:
            j = i
            while j < n and filled[j]:
                j += 1
            if (j - i) > max_gap_points:          # gap dài
                masked[i:j] = True
            i = j
        else:
            i += 1
    g["masked"] = masked

    # value_filled: nội suy gap ngắn, gap dài (masked) giữ NaN
    vf = g[value_col].astype(float).copy()
    if interpolate:
        vf = vf.interpolate(method="linear", limit_direction="both")
    vf[masked] = np.nan
    g["value_filled"] = vf

    # segment id: tăng sau mỗi vùng masked (gap dài)
    seg = np.zeros(len(g), dtype=int)
    cur = 0; prev_masked = False
    for k in range(len(g)):
        if masked[k]:
            prev_masked = True
        else:
            if prev_masked:
                cur += 1
            prev_masked = False
        seg[k] = cur
    g["segment"] = seg
    return g


def normalize_per_kpi(g, method="robust", value_col="value_filled",
                      out_col="value_norm", fit_mask=None, eps=1e-9):
    """Chuẩn hoá theo KPI. center/scale FIT chỉ trên fit_mask (mặc định split=='train',
    bỏ filled & masked). method: 'zscore' | 'robust'(median/MAD) | 'minmax'."""
    g = g.copy()
    x = g[value_col].astype(float)
    if fit_mask is None:
        fit_mask = np.ones(len(g), dtype=bool)
        if "split" in g.columns:
            fit_mask &= (g["split"].values == "train")
        for c in ("filled", "masked"):
            if c in g.columns:
                fit_mask &= ~g[c].values
    xf = x[fit_mask].dropna()

    if method == "zscore":
        center, scale = xf.mean(), xf.std()
    elif method == "robust":
        center = xf.median()
        scale = np.median(np.abs(xf - center)) * 1.4826
    elif method == "minmax":
        center, scale = xf.min(), (xf.max() - xf.min())
    else:
        raise ValueError(f"method không hợp lệ: {method}")
    scale = scale if scale and scale > eps else eps

    g[out_col] = (x - center) / scale
    g.attrs["norm"] = dict(method=method, center=float(center), scale=float(scale))
    return g


def preprocess_all(df, step_map=None, max_gap_points=5, norm_method="robust",
                   kpi_col="kpi", interpolate=True):
    """Chạy reindex -> mask_long_gaps -> normalize cho từng KPI rồi gộp lại.
    LƯU Ý: cột 'split' phải có sẵn (gọi time_split_per_kpi trước) để normalize fit đúng trên train.
    step_map: dict {kpi: step_s} nếu muốn cố định; None -> tự suy từ data."""
    parts = []
    for kpi, g in df.groupby(kpi_col):
        step = None if step_map is None else step_map.get(kpi)
        g2 = reindex_regular_grid(g, step_s=step, kpi_col=kpi_col)
        if "split" in g.columns:
            g2 = g2.sort_values("timestamp")
            g2["split"] = g2["split"].ffill().bfill()
        g2 = mask_long_gaps(g2, max_gap_points=max_gap_points, interpolate=interpolate)
        g2 = normalize_per_kpi(g2, method=norm_method)
        parts.append(g2)
    return pd.concat(parts, ignore_index=True)
