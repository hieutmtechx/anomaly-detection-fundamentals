"""Panel 4 — System Health.
Hai khối: (A) nhật ký alert, (B) chỉ báo drift.
"""

import plotly.graph_objects as go
import streamlit as st

from common import load_data
from panel4 import build_alerts, compute_drift

TP_COLOR, FP_COLOR = "#4c78a8", "#e45756"   # đúng (TP) xanh · báo động giả (FP) đỏ

st.set_page_config(page_title="System Health", layout="wide")
st.title("Panel 4 · System Health")
st.caption("kiểm tra hệ thống trên tập test — nhật ký alert, độ trôi (drift) ")

alerts = build_alerts()

# ───────────────────────── A. Alert log ─────────────────────────
st.subheader("A · Nhật ký alert")

kpi_opts = sorted(alerts["kpi"].unique())
tmin, tmax = alerts["dt"].min().to_pydatetime(), alerts["dt"].max().to_pydatetime()

f1, f2, f3 = st.columns([2, 2, 1])
sel_kpi = f1.multiselect("KPI", kpi_opts, default=kpi_opts,
                         help="Để trống = tất cả KPI")
sel_outcome = f2.radio("Kết quả", ["Tất cả", "Chỉ đúng (TP)", "Chỉ báo động giả (FP)"],
                       horizontal=True)
sort_by = f3.radio("Sắp theo", ["Vượt ngưỡng nhiều nhất", "Mới nhất"])

rng = st.slider("Khoảng thời gian", min_value=tmin, max_value=tmax,
                value=(tmin, tmax), format="YYYY-MM-DD")

# --- áp bộ lọc ---
a = alerts
if sel_kpi:
    a = a[a["kpi"].isin(sel_kpi)]
a = a[(a["dt"] >= rng[0]) & (a["dt"] <= rng[1])]
if sel_outcome == "Chỉ đúng (TP)":
    a = a[a["y"] == 1]
elif sel_outcome == "Chỉ báo động giả (FP)":
    a = a[a["y"] == 0]

a = a.sort_values("margin" if sort_by.startswith("Vượt") else "dt",
                  ascending=False).reset_index(drop=True)

# --- bảng ---
st.caption(f"Hiển thị **{len(a):,}** / {len(alerts):,} alert.")
show = a.copy()
show["dt"] = show["dt"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(
    show[["dt", "kpi", "score", "thr", "margin", "outcome"]].rename(columns={
        "dt": "Thời điểm", "kpi": "KPI", "score": "Score",
        "thr": "Ngưỡng", "margin": "Vượt ngưỡng", "outcome": "Kết quả"}),
    width="stretch", height=380, hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn(format="%.4f"),
        "Ngưỡng": st.column_config.NumberColumn(format="%.4f"),
        "Vượt ngưỡng": st.column_config.NumberColumn(format="%.4f"),
    })

# --- 3 metric tóm tắt (theo bộ lọc) ---
if len(a):
    tp = int((a["y"] == 1).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng alert", f"{len(a):,}")
    m2.metric("Tỷ lệ đúng (TP)", f"{tp / len(a) * 100:.0f}%",
              help="Bao nhiêu % alert thật sự là anomaly (còn lại là báo động giả)")
    m3.metric("Số KPI dính alert", f"{a['kpi'].nunique()}")

    # --- Biểu đồ 1: alert theo từng KPI, tách TP/FP ---
    st.markdown("**Số alert theo KPI** (xanh = đúng · đỏ = báo động giả)")
    by_kpi = (a.assign(tp=a["y"] == 1)
              .groupby("kpi")["tp"].agg(tp="sum", tong="size"))
    by_kpi["fp"] = by_kpi["tong"] - by_kpi["tp"]
    by_kpi = by_kpi.sort_values("tong")
    fig1 = go.Figure()
    fig1.add_bar(y=by_kpi.index, x=by_kpi["tp"], orientation="h",
                 name="đúng (TP)", marker_color=TP_COLOR,
                 hovertemplate="%{y}: %{x} đúng<extra></extra>")
    fig1.add_bar(y=by_kpi.index, x=by_kpi["fp"], orientation="h",
                 name="báo động giả (FP)", marker_color=FP_COLOR,
                 hovertemplate="%{y}: %{x} giả<extra></extra>")
    fig1.update_layout(barmode="stack", height=max(240, 26 * len(by_kpi)),
                       margin=dict(l=0, r=0, t=6, b=0),
                       legend=dict(orientation="h", y=1.05),
                       xaxis_title="số alert")
    st.plotly_chart(fig1, width="stretch")

    # --- Biểu đồ 2: alert theo thời gian (số alert mỗi ngày) ---
    st.markdown("**Số alert theo ngày**")
    per_day = (a.set_index("dt").assign(tp=a["y"].values == 1)
               .resample("D")["tp"].agg(tp="sum", tong="size"))
    per_day = per_day[per_day["tong"] > 0]
    per_day["fp"] = per_day["tong"] - per_day["tp"]
    fig2 = go.Figure()
    fig2.add_bar(x=per_day.index, y=per_day["tp"], name="đúng (TP)",
                 marker_color=TP_COLOR)
    fig2.add_bar(x=per_day.index, y=per_day["fp"], name="báo động giả (FP)",
                 marker_color=FP_COLOR)
    fig2.update_layout(barmode="stack", height=280,
                       margin=dict(l=0, r=0, t=6, b=0),
                       legend=dict(orientation="h", y=1.12),
                       xaxis_title="ngày", yaxis_title="số alert")
    st.plotly_chart(fig2, width="stretch")
    st.caption("Các cụm alert dồn vào vài giai đoạn, mỗi KPI chỉ hoạt động trong một khoảng thời gian riêng, không trải đều cả năm.")
else:
    st.info("Không có alert nào khớp bộ lọc hiện tại.")

# ───────────────────────── B. Drift indicator ─────────────────────────
st.divider()
st.subheader("B · Chỉ báo drift")
st.caption("Mức nền (mean) và độ dao động (std) của mỗi KPI ở **hiện tại** (test) "
           "đã trôi bao xa so với **baseline** (train). `drift` = số độ lệch chuẩn "
           "nền đã dịch; |drift|>2 = trôi mạnh, có thể cần train lại.")

drift = compute_drift()


def _band(v):
    """Màu nền theo mức drift: ổn định / vừa / mạnh."""
    x = abs(v)
    if x > 2:
        return "background-color:#e45756;color:white"     # mạnh
    if x > 1:
        return "background-color:#f2cf5b"                  # vừa
    return "background-color:#a6d5a1"                      # ổn định


show_d = drift.copy()
show_d["hướng"] = show_d["drift_mean"].map(lambda v: "▲" if v > 0 else ("▼" if v < 0 else "–"))
show_d = show_d[["kpi", "mean_base", "mean_now", "hướng", "drift_mean",
                 "std_base", "std_now", "std_ratio"]]
show_d.columns = ["KPI", "mean nền", "mean hiện tại", "↕", "drift",
                  "std nền", "std hiện tại", "std tỉ lệ"]
st.dataframe(
    show_d.style
    .format({c: "{:.3g}" for c in ["mean nền", "mean hiện tại", "std nền", "std hiện tại"]}
            | {"drift": "{:+.2f}", "std tỉ lệ": "{:.2f}"})
    .map(_band, subset=["drift"]),
    width="stretch", height=380, hide_index=True)

n_strong = int((drift["abs_drift"] > 2).sum())
n_mod = int(((drift["abs_drift"] > 1) & (drift["abs_drift"] <= 2)).sum())
st.caption(f"🟩 ổn định (|drift|≤1): {len(drift) - n_mod - n_strong}  ·  "
           f"🟨 vừa (1–2): {n_mod}  ·  🟥 mạnh (>2): {n_strong}. "
           + ("Bộ dữ liệu này không có KPI trôi mạnh"
              if n_strong == 0 else ""))

# --- Chọn 1 KPI: vẽ baseline band vs chuỗi thật ---
st.markdown("####")          # khoảng trống dọc lớn giữa bảng và khối chi tiết
st.markdown("##### Xem chi tiết một KPI")

with st.container(border=True):
    sel_d = st.selectbox("Chọn KPI", drift["kpi"].tolist(), key="drift_kpi")
    row = drift[drift["kpi"] == sel_d].iloc[0]
    gser = load_data("train")
    gser = gser[gser["k8"] == sel_d].sort_values("timestamp")
    gp = gser.iloc[:: max(1, len(gser) // 4000)]     # giảm điểm để vẽ nhanh
    mb, sb, mn = row["mean_base"], row["std_base"], row["mean_now"]

    st.markdown(f"nền train (xanh) = **{mb:.3g}** · nền hiện tại (đỏ) = **{mn:.3g}** · "
                f"drift **{row['drift_mean']:+.2f}**")

    figd = go.Figure()
    # dải baseline mean ± std (vùng "bình thường" hồi train)
    figd.add_hrect(y0=mb - sb, y1=mb + sb, line_width=0,
                   fillcolor="#4c78a8", opacity=0.12,
                   annotation_text="baseline mean±std", annotation_position="top left")
    figd.add_hline(y=mb, line=dict(color="#4c78a8", width=1.5, dash="dash"))
    figd.add_hline(y=mn, line=dict(color="#e45756", width=1.5, dash="dot"))
    figd.add_trace(go.Scattergl(x=gp["dt"], y=gp["value"], mode="lines",
                                line=dict(width=1, color="#888"), name="value"))
    figd.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                       showlegend=False)
    st.plotly_chart(figd, width="stretch")
    st.caption("Vùng xanh = khoảng bình thường hồi train (mean±std). Đường đỏ chấm = "
               "mức nền hiện tại. Đỏ lệch khỏi vùng xanh ⇒ mức nền đã trôi.")