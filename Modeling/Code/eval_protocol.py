import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def time_split_per_kpi(df, train_frac=0.6, val_frac=0.2, kpi_col="kpi", time_col="timestamp"):
    parts = []
    for _, g in df.groupby(kpi_col):
        g = g.sort_values(time_col).copy()
        n = len(g); i_tr = int(n * train_frac); i_va = int(n * (train_frac + val_frac))
        s = np.empty(n, dtype=object)
        s[:i_tr] = "train"; s[i_tr:i_va] = "val"; s[i_va:] = "test"
        g["split"] = s
        parts.append(g)
    return pd.concat(parts)


def prf(y_true, y_pred, beta=1.0):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, fbeta=f)


def _segments(y_true):
    y = np.asarray(y_true); idx = np.where(y == 1)[0]
    if len(idx) == 0:
        return []
    brk = np.where(np.diff(idx) > 1)[0]
    return list(zip(np.r_[idx[0], idx[brk + 1]].tolist(), np.r_[idx[brk], idx[-1]].tolist()))


def point_adjust(y_true, y_pred):
    """>=1 điểm pred=1 trong segment ground-truth -> cả segment thành TP."""
    y_pred = np.asarray(y_pred).copy()
    for s, e in _segments(y_true):
        if y_pred[s:e + 1].any():
            y_pred[s:e + 1] = 1
    return y_pred


def time_to_detect(y_true, y_pred, step_s=60):
    y_pred = np.asarray(y_pred); delays = []; n_det = 0; n_seg = 0
    for s, e in _segments(y_true):
        n_seg += 1
        hits = np.where(y_pred[s:e + 1] == 1)[0]
        if len(hits):
            n_det += 1; delays.append(hits[0] * step_s)
    return dict(ttd_sec_mean=float(np.mean(delays)) if delays else None,
                detected_segments=n_det, total_segments=n_seg)


def best_f1_prcurve(y_true, scores, beta=1.0):
    """Quét MỌI threshold qua precision_recall_curve (không xấp xỉ grid).
    Chỉ dùng cho point-wise: point-adjust làm prediction không đơn điệu theo threshold."""
    p, r, th = precision_recall_curve(y_true, scores)
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r + 1e-12)
    f_use = f[:-1] if len(th) else f   # precision_recall_curve trả len(th)+1 điểm; bỏ điểm cuối (recall=0)
    if len(f_use) == 0:
        return dict(threshold=float("inf"), precision=0.0, recall=0.0, fbeta=0.0)
    i = int(np.argmax(f_use))
    return dict(threshold=float(th[i]), precision=float(p[i]), recall=float(r[i]),
                fbeta=float(f_use[i]))


def best_f1_threshold(y_true, scores, adjust=False, beta=1.0, n_grid=150):
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    cand = np.unique(np.quantile(scores, np.linspace(0, 1, n_grid)))
    best = None
    for t in cand:
        yp = (scores >= t).astype(int)
        if adjust:
            yp = point_adjust(y_true, yp)
        m = prf(y_true, yp, beta)
        if best is None or m["fbeta"] > best["fbeta"]:
            best = dict(threshold=float(t), **m)
    return best


def evaluate_protocol(y_val, s_val, y_test, s_test, beta=1.0, step_s=60):
    """Gọi RIÊNG cho từng KPI rồi mới gộp — segment không được bắc qua ranh giới KPI."""
    out = {}
    for mode, adj in [("PW", False), ("PA", True)]:
        # PW: best-F1 chính xác qua PR curve sklearn; PA: grid sweep + point-adjust
        bt = best_f1_threshold(y_val, s_val, adjust=True, beta=beta) if adj \
            else best_f1_prcurve(y_val, s_val, beta=beta)
        thr = bt["threshold"]
        yp = (np.asarray(s_test) >= thr).astype(int)
        if adj:
            yp = point_adjust(y_test, yp)
        m = prf(y_test, yp, beta); m["threshold"] = thr; m["val_fbeta"] = bt["fbeta"]
        out[mode] = m
    out["AP_pw"] = float(average_precision_score(y_test, s_test))
    out["TTD"] = time_to_detect(y_test, (np.asarray(s_test) >= out["PW"]["threshold"]).astype(int), step_s)
    return out


def plot_pr_curve(curves, title=None, mark_best_f1=True, per_kpi=False, ax=None):
    """Vẽ PR curve point-wise cho 1 hoặc nhiều model.

    per_kpi=False (micro/gộp): curves = {ten_model: (y_true, scores)} — gộp test mọi KPI,
        quét MỘT threshold toàn cục. Legend ghi AP gộp. Đánh dấu best-F1.
    per_kpi=True (macro): curves = {ten_model: [(y1,s1), (y2,s2), ...]} — 1 phần tử / KPI.
        Vẽ mờ từng KPI theo màu model, legend ghi MACRO AP = trung bình AP per-KPI
        (nhất quán với cách tune threshold riêng từng KPI).
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    base_rates = []

    if per_kpi:
        for name, kpi_list in curves.items():
            aps = []
            color = None
            for y_true, scores in kpi_list:
                y_true = np.asarray(y_true); scores = np.asarray(scores)
                if y_true.sum() == 0:
                    continue
                p, r, _ = precision_recall_curve(y_true, scores)
                line, = ax.plot(r, p, lw=0.9, alpha=0.45,
                                color=color)          # cùng màu cho mọi KPI của 1 model
                color = line.get_color()
                aps.append(average_precision_score(y_true, scores))
                base_rates.append(y_true.mean())
            macro = float(np.mean(aps)) if aps else 0.0
            ax.plot([], [], color=color, lw=2,
                    label=f"{name} (macro AP={macro:.3f}, n={len(aps)} KPI)")
        ttl = title or "PR Curve per-KPI (macro)"
    else:
        for name, (y_true, scores) in curves.items():
            y_true = np.asarray(y_true); scores = np.asarray(scores)
            p, r, th = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            line, = ax.plot(r, p, lw=1.5, label=f"{name} (AP={ap:.3f})")
            base_rates.append(y_true.mean())
            if mark_best_f1 and len(th):
                f = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
                i = int(np.argmax(f))
                ax.scatter([r[i]], [p[i]], color=line.get_color(), s=45, zorder=5,
                           edgecolor="black", linewidth=0.6)
                ax.annotate(f"F1={f[i]:.2f}", (r[i], p[i]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8)
        ttl = title or "PR Curve (point-wise, gộp/micro)"

    base = float(np.mean(base_rates)) if base_rates else 0.0
    ax.axhline(base, ls="--", color="grey", lw=0.8, label=f"random (P={base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(ttl); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    return ax
