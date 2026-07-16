"""Panel 1 — Data Explorer.

Sáu khối: (A) chuỗi thời gian, (B) phân bố nhãn, (C) tương quan giữa KPI,
(D) phân bố giá trị, (E) đoạn anomaly, (F) chu kỳ mùa vụ.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


from common import (ALIGNED_16, MIRROR_PAIRS, acf_profile, corr_matrix,
                    downsample_keep_anomalies, hour_weekday, kpi_label,
                    kpi_meta, load_data)

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Panel 1 · Data Explorer")

source = st.sidebar.radio("Dữ liệu", ["train", "test"], horizontal=True)
meta = kpi_meta(source)
df = load_data(source)
has_labels = df["label"].notna().any()
if not has_labels:
    st.sidebar.caption("`test.csv`")


options = meta["k8"].tolist()
labels = {r.k8: kpi_label(r) for r in meta.itertuples()}
sel = st.sidebar.selectbox("Chọn KPI", options, format_func=lambda k: labels[k])
row = meta[meta["k8"] == sel].iloc[0]

# ───────────────────────── A. Time series ─────────────────────────
st.subheader(f"A · Chuỗi thời gian — `{sel}`")

tmin, tmax = row["start"].to_pydatetime(), row["end"].to_pydatetime()
rng = st.slider("Khoảng thời gian", min_value=tmin, max_value=tmax,
                value=(tmin, tmax), format="YYYY-MM-DD")
show_anom = st.checkbox("Điểm anomaly", value=True) if has_labels else False

g = df[df["k8"] == sel]
g = g[(g["dt"] >= rng[0]) & (g["dt"] <= rng[1])]
gp = downsample_keep_anomalies(g)

fig = go.Figure()
fig.add_trace(go.Scattergl(x=gp["dt"], y=gp["value"], mode="lines",
                           name="value", line=dict(width=1, color="#4c78a8")))
if show_anom:
    a = gp[gp["label"] == 1]
    fig.add_trace(go.Scattergl(x=a["dt"], y=a["value"], mode="markers",
                               name="anomaly",
                               marker=dict(color="#e45756", size=5)))
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width="stretch")

if has_labels:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số điểm trong cửa sổ", f"{len(g):,}")
    c2.metric("Số điểm anomaly", f"{int(g['label'].sum()):,}")
    c3.metric("Tỷ lệ anomaly", f"{g['label'].mean() * 100:.2f}%")
else:
    st.metric("Số điểm trong cửa sổ", f"{len(g):,}")

if len(gp) < len(g):
    st.caption(f"Đã downsample {len(g):,} → {len(gp):,} điểm để vẽ nhanh "
               f"(giữ nguyên toàn bộ điểm anomaly).")

# ───────────────────────── B. Label distribution ─────────────────────────
st.subheader("B · Phân bố tỷ lệ anomaly theo KPI")
if not has_labels:
    st.info("Tập test không có nhãn nên bỏ qua phân bố anomaly.")
else:
    d = meta.sort_values("anom_rate")
    colors = ["#e45756" if k == sel else "#9ecae1" for k in d["k8"]]
    fig2 = go.Figure(go.Bar(x=d["anom_rate"] * 100, y=d["k8"], orientation="h",
                        marker_color=colors,
                        hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig2.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                   xaxis_title="Tỷ lệ anomaly (%) — thang log", xaxis_type="log")
    st.plotly_chart(fig2, width="stretch")
    st.caption(
    f"Tỷ lệ anomaly chênh nhau tới {d['anom_rate'].max() / d['anom_rate'].min():.0f} lần "
    f"giữa các KPI ({d['anom_rate'].min() * 100:.2f}%–{d['anom_rate'].max() * 100:.2f}%). "
    "Dữ liệu mất cân bằng mạnh ⇒ dùng F1/PR-AUC, không dùng accuracy; và tune "
    "ngưỡng riêng từng KPI thay vì một ngưỡng chung. KPI đang chọn được tô đỏ."
    )

# ───────────────────────── C. Correlation heatmap ─────────────────────────
st.subheader("C · Correlation heatmap giữa các KPI")

available = [k for k in ALIGNED_16 if k in set(meta["k8"])]
picked = st.multiselect(
    "KPI đưa vào heatmap (mặc định = 16 KPI có lưới 60s khớp nhau)",
    options=meta["k8"].tolist(), default=available)
differenced = st.toggle("Dùng chuỗi sai phân diff() (khử tương quan giả)",
                        value=True)

if len(picked) < 2:
    st.info("Chọn ít nhất 2 KPI.")
else:
    corr, n_rows = corr_matrix(tuple(sorted(picked)), differenced, source)
    if corr is None:
        st.warning(f"Các KPI này chỉ chia sẻ {n_rows} timestamp chung — "
                   "không đủ để tính tương quan. Chọn các KPI cùng cohort.")
    else:
        fig3 = px.imshow(corr, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
        fig3.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, width="stretch")

        v = corr.values[np.triu_indices_from(corr, 1)]
        st.caption(
            f"{n_rows:,} hàng có đầy đủ mọi KPI đã chọn · "
            f"|r| trung bình = **{np.abs(v).mean():.3f}**. "
            + ("Đang xem **sai phân**: tương quan còn lại là thật (biến động cùng nhau). "
               if differenced else
               "Đang xem **giá trị thô**: phần lớn tương quan cao là **giả** — chỉ do "
               "mọi KPI cùng chu kỳ ngày. Bật diff() để thấy sự khác biệt. ")
        )
        mp = [f"`{a}`↔`{b}`" for a, b in MIRROR_PAIRS
              if a in picked and b in picked]
        if mp and differenced:
            st.caption("Cặp còn tương quan mạnh sau sai phân: " + ", ".join(mp)
                       + " — nhiều khả năng cùng một service (2 replica), anomaly "
                       "trùng nhau tới ~83%. Đây là bằng chứng cho system-level anomaly.")

# ───────────────────────── D. Value distribution ─────────────────────────
st.subheader(f"D · Phân bố giá trị — `{sel}` ({source})")

vals = df[df["k8"] == sel]                       # toàn bộ KPI, không giới hạn cửa sổ
c_log, c_clip = st.columns(2)
logy = c_log.toggle("Trục đếm log (y)", value=True, key="dist_logy")
clip = c_clip.toggle("Phóng to vùng chính (cắt đuôi p1–p99)", value=False,
                     key="dist_clip")

vp = vals
if clip:
    lo, hi = vals["value"].quantile(0.01), vals["value"].quantile(0.99)
    vp = vals[(vals["value"] >= lo) & (vals["value"] <= hi)]

fig4 = go.Figure()
if has_labels:
    fig4.add_trace(go.Histogram(x=vp[vp["label"] == 0]["value"], name="normal",
                                histnorm="probability density", nbinsx=80,
                                marker_color="#4c78a8", opacity=0.75))
    fig4.add_trace(go.Histogram(x=vp[vp["label"] == 1]["value"], name="anomaly",
                                histnorm="probability density", nbinsx=80,
                                marker_color="#e45756", opacity=0.75))
    fig4.update_layout(barmode="overlay")
    ytitle = "mật độ"
else:
    fig4.add_trace(go.Histogram(x=vp["value"], name="value",
                                marker_color="#4c78a8", nbinsx=80))
    ytitle = "số điểm"
fig4.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                   legend=dict(orientation="h", y=1.1),
                   yaxis_type="log" if logy else "linear",
                   xaxis_title="value", yaxis_title=ytitle)
st.plotly_chart(fig4, width="stretch")

s = vals["value"]
cols = st.columns(5)
cols[0].metric("mean", f"{s.mean():.3g}")
cols[1].metric("std", f"{s.std():.3g}")
cols[2].metric("median", f"{s.median():.3g}")
cols[3].metric("min", f"{s.min():.3g}")
cols[4].metric("max", f"{s.max():.3g}")

if has_labels:
    st.caption("Mật độ **normal** (xanh) vs **anomaly** (đỏ). Đỏ dồn ở **đuôi** ⇒ "
               "point anomaly (giá trị cực trị, dễ bắt bằng ngưỡng); đỏ **chồng** lên "
               "vùng normal ⇒ contextual anomaly (ngưỡng tĩnh sẽ bỏ sót). Bật 'cắt "
               "đuôi' để phóng to vùng chính — lưu ý có thể che mất anomaly ở đuôi.")


# ───────────────────────── E. Anomaly segments (episodes) ─────────────────────────
st.subheader(f"E · Đoạn anomaly (episodes) — `{sel}`")

if not has_labels:
    st.info("Tập test không có nhãn — không có đoạn anomaly để hiển thị.")
else:
    gs = df[df["k8"] == sel].sort_values("timestamp").reset_index(drop=True)
    idx = np.where(gs["label"].values == 1)[0]
    if len(idx) == 0:
        st.info("KPI này không có điểm anomaly nào trong tập train.")
    else:
        # gom điểm label==1 liền kề thành đoạn — tách khi CÓ điểm normal chen giữa
        # HOẶC cách nhau quá xa về thời gian (lỗ dữ liệu), tránh thổi phồng "thời lượng"
        step = int(np.median(np.diff(gs["timestamp"].values)))
        MAX_GAP = 2 * step            # cách >2 bước mẫu ⇒ coi là lỗ dữ liệu ⇒ đoạn mới
        ts_a = gs["timestamp"].values[idx]
        brk = np.where((np.diff(idx) > 1) | (np.diff(ts_a) > MAX_GAP))[0]
        starts = np.r_[idx[0], idx[brk + 1]]
        ends = np.r_[idx[brk], idx[-1]]

        seg = pd.DataFrame({
            "bắt đầu": gs["dt"].iloc[starts].values,
            "kết thúc": gs["dt"].iloc[ends].values,
            "số điểm": ends - starts + 1,
        })
        seg["thời lượng"] = seg["kết thúc"] - seg["bắt đầu"]
        seg = seg.sort_values("số điểm", ascending=False).reset_index(drop=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số đoạn", len(seg))
        c2.metric("Tổng điểm anomaly", int(seg["số điểm"].sum()))
        c3.metric("Độ dài trung vị (điểm)", int(seg["số điểm"].median()))
        c4.metric("Đoạn dài nhất (điểm)", int(seg["số điểm"].max()))

        fig5 = go.Figure(go.Histogram(x=seg["số điểm"], nbinsx=40,
                                      marker_color="#e45756"))
        fig5.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="độ dài đoạn (số điểm liên tiếp)",
                           yaxis_title="số đoạn", yaxis_type="log")
        st.plotly_chart(fig5, width="stretch")
        st.caption("Mỗi đoạn = một cụm điểm anomaly liền kề (một 'sự cố'). Đoạn dài "
                   "⇒ collective anomaly (kéo dài); đoạn 1–vài điểm ⇒ gần point "
                   "anomaly. Bảng dưới sắp theo độ dài giảm dần — mỗi dòng là một "
                   "ứng viên cho incident timeline ở Panel 5.")

        show = seg.copy()
        show["bắt đầu"] = show["bắt đầu"].dt.strftime("%Y-%m-%d %H:%M")
        show["kết thúc"] = show["kết thúc"].dt.strftime("%Y-%m-%d %H:%M")
        show["thời lượng"] = show["thời lượng"].astype(str).str.replace("0 days ", "", regex=False)
        st.dataframe(show[["bắt đầu", "kết thúc", "thời lượng", "số điểm"]],
                     width="stretch", height=320)
        
# ───────────────────────── F. Seasonality (chu kỳ / mùa vụ) ─────────────────────────
st.subheader(f"F · Chu kỳ / mùa vụ — `{sel}`")

sea = acf_profile(sel, source)
ac, step, per_day = sea["acf"], sea["step_s"], sea["per_day"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Chu kỳ trội", f"{sea['dom_hours']:.1f}h",
          help=f"{sea['dom_lag']} điểm × {step}s — tự dò từ đỉnh ACF")
c2.metric("ACF @ 1 ngày", f"{ac[per_day]:+.3f}" if per_day < len(ac) else "—")
c3.metric("ACF @ 1 tuần", f"{ac[per_day * 7]:+.3f}" if per_day * 7 < len(ac) else "—")
c4.metric("Bước lấy mẫu", f"{step}s")

# --- F1. ACF theo độ trễ ---
lags_h = np.arange(len(ac)) * step / 3600
fig6 = go.Figure(go.Scattergl(x=lags_h, y=ac, mode="lines",
                              line=dict(width=1, color="#4c78a8"), name="ACF"))
fig6.add_hline(y=0, line=dict(color="grey", width=0.8))
for d in range(1, int(lags_h[-1] // 24) + 1):
    fig6.add_vline(x=d * 24, line=dict(color="#e45756", width=0.8, dash="dot"))
fig6.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                   xaxis_title="độ trễ (giờ) — vạch đỏ = mốc mỗi 24h",
                   yaxis_title="autocorrelation")
st.plotly_chart(fig6, width="stretch")
st.caption(f"Đỉnh ACF lặp lại đều tại các mốc 24h ⇒ chuỗi có **chu kỳ ngày**. "
           f"Chu kỳ dò được (**{sea['dom_lag']} điểm**) chính là tham số `period` mà "
           f"STL/ARIMA cần.")

# --- F2. Heatmap giờ × thứ ---
excl = st.toggle("Chỉ dùng điểm normal (loại anomaly)", value=True,
                 key="sea_excl") if has_labels else False
hw = hour_weekday(sel, excl, source)
fig7 = px.imshow(hw, color_continuous_scale="Viridis", aspect="auto",
                 labels=dict(x="giờ trong ngày", y="thứ", color="value TB"))
fig7.update_yaxes(tickmode="array", tickvals=list(range(7)),
                  ticktext=["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5",
                            "Thứ 6", "Thứ 7", "CN"])
fig7.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig7, width="stretch")
st.caption("Giá trị trung bình theo **giờ trong ngày × thứ trong tuần**. Màu biến "
           "thiên mạnh theo cột ⇒ **nhịp ngày** (biên độ ~2–3 lần độ lệch chuẩn). "
           "So các hàng với nhau để kiểm tra **nhịp tuần**: nếu 7 hàng gần như giống "
           "nhau thì không có mẫu hình tuần — lưu ý ACF@1 tuần cao không chứng minh "
           "nhịp tuần, vì 1 tuần = 7 chu kỳ ngày.")