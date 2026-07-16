"""Feature Engineering & Feature Selection cho anomaly detection (univariate, per-KPI).

Hai hàm chính, mỗi hàm một trách nhiệm tách bạch:
- build_features(p):   biến chuỗi thời gian -> 27 feature
- select_features(F, p): lọc feature dựa trên nhãn train (variance -> correlation -> MI).

Quy ước đầu vào `p`: DataFrame đã qua preprocess_all, có sẵn các cột
`value_filled`, `timestamp`, `masked`, `split`, `label`, `segment`.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def build_features(p, period=None):
    """Tạo 27 feature thô từ `value_filled`. Deterministic, không dùng nhãn.

    `period` = độ dài cửa sổ trend (số hàng ~ 1 ngày). Nếu None sẽ tự suy từ
    sampling step: 1440 cho KPI 60s, 288 cho KPI 300s. Truyền tay để override.

    Nhóm feature:
      - lag_{1,5,10,60}                 : giá trị quá khứ trực tiếp
      - rmean/rstd/rmin/rmax_{5,15,60}  : thống kê cửa sổ trượt
      - dev_{5,15,60} = x - rmean       : độ lệch so với baseline cục bộ
      - diff_1, pct_1                   : tốc độ thay đổi
      - deseason_resid                  : phần dư sau khi bỏ trend + seasonal
      - tod_sin/cos, dow_sin/cos        : mã hoá thời gian tuần hoàn
    """
    x = p.value_filled.astype(float)
    ts = p.timestamp.values
    masked = p.masked.values
    tr_rows = (p.split.values == "train")

    # period (số hàng ~ 1 ngày) suy từ sampling step nếu không truyền vào
    if period is None:
        step = int(pd.Series(np.diff(ts)).mode().iloc[0])   # 60s hoặc 300s
        period = 86400 // step                              # -> 1440 (60s) | 288 (300s)

    F = pd.DataFrame(index=p.index)
    F["value"] = x

    # 1) Lag — giá trị tại t-L
    for L in [1, 5, 10, 60]:
        F[f"lag_{L}"] = x.shift(L)

    # 2) Rolling stats + độ lệch baseline
    for w in [5, 15, 60]:
        r = x.rolling(w)
        F[f"rmean_{w}"] = r.mean() 
        F[f"rstd_{w}"] = r.std()
        F[f"rmin_{w}"]  = r.min()
        F[f"rmax_{w}"] = r.max()
        F[f"dev_{w}"]   = x - r.mean()

    # 3) Rate of change
    F["diff_1"] = x.diff(1)
    F["pct_1"]  = (x.diff(1) / (x.shift(1).abs() + 1e-6)).clip(-50, 50)

    #    trend = rolling median TRAILING (past-only, center=False)
    #    seasonal profile = median theo phút-trong-ngày
    xi = x.interpolate(limit_direction="both").bfill().ffill()
    trend = xi.rolling(period, center=False, min_periods=min(200, period // 2)).median()
    mod = ((ts % 86400) // 60).astype(int)                       # phút-trong-ngày 0..1439
    prof = pd.Series((xi - trend).values[tr_rows]).groupby(mod[tr_rows]).median()
    seasonal = pd.Series(mod).map(prof)
    resid = (xi.values - trend.values - seasonal.values)
    resid[masked] = np.nan                                       # gap dài -> không tin cậy
    F["deseason_resid"] = resid

    # 5) Time encoding (tuần hoàn)
    tod = (ts % 86400) / 86400.0
    dow = ((ts // 86400) % 7) / 7.0
    F["tod_sin"] = np.sin(2*np.pi*tod); F["tod_cos"] = np.cos(2*np.pi*tod)
    F["dow_sin"] = np.sin(2*np.pi*dow); F["dow_cos"] = np.cos(2*np.pi*dow)
    return F


def select_features(F, p, corr_thr=0.95, mi_thr=1e-4, min_samples=50):
    """Lọc feature 3 bước:

      B1 Degenerate  : loại cột suy biến (std == 0 hoặc toàn NaN). KHÔNG lọc theo
                       variance/CV — feature tốt cho anomaly thường gần hằng số rồi
                       bùng nổ hiếm, lọc phương sai sẽ loại nhầm; để B3 (MI) lo.
      B2 Correlation : với mỗi cặp |corr| > corr_thr, giữ feature có MI cao hơn.
      B3 MI ranking  : giữ feature có mutual information với nhãn > mi_thr.

    MI tính riêng từng feature trên đúng các hàng train non-NaN của nó (univariate),
    không đòi hỏi mọi feature cùng non-NaN → tận dụng tối đa dữ liệu quanh gap.

    Trả về: list tên feature giữ lại, xếp theo MI giảm dần.
    """
    lab = p.label.values.astype(int)
    tr = (p.split.values == "train")

    # B1 — chỉ loại cột hằng số / toàn NaN (std NaN -> "> 0" là False -> loại)
    std_tr = F[tr].std()
    keep = [c for c in F.columns if std_tr[c] > 0]

    # MI per-feature: mỗi feature dùng riêng các hàng train non-NaN của nó
    mi = {}
    for c in keep:
        m = tr & F[c].notna().values
        mi[c] = (mutual_info_classif(F.loc[m, [c]], lab[m],
                                     discrete_features=False, random_state=0)[0]
                 if m.sum() >= min_samples and lab[m].sum() > 0 else 0.0)
    mi = pd.Series(mi)

    # B2 — bỏ feature dư thừa theo cặp tương quan cao, giữ cái MI lớn hơn
    corr = F.loc[tr, keep].corr().abs()
    drop = set()
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            if a in drop or b in drop:
                continue
            if corr.loc[a, b] > corr_thr:
                drop.add(a if mi[a] < mi[b] else b)

    # B3 — xếp hạng MI, giữ ngưỡng
    final = mi[[c for c in keep if c not in drop]].sort_values(ascending=False)
    return final[final > mi_thr].index.tolist()
