"""Panel 5 — Root Cause.

Khối: (A) incident timeline. (B) top-3 root cause + SHAP
"""
import plotly.graph_objects as go
import streamlit as st

from modellab import shap_importance
from panel5 import build_incidents, root_cause


@st.cache_data(show_spinner=False)
def _shap(kpi: str):
    """SHAP feature importance của KPI."""
    try:
        return shap_importance(kpi)
    except Exception:
        return None

st.set_page_config(page_title="Root Cause", layout="wide")
st.title("Panel 5 · Root Cause")
st.caption("Gom alert thành sự cố, rồi truy metric nào là nguyên nhân gốc.")

# ───────────────────────── A. Incident timeline ─────────────────────────
st.subheader("A · Dòng thời gian sự cố")

c1, c2 = st.columns([2, 3])
gap = c1.slider("Khoảng lặng tách sự cố (phút)", 5, 120, 30, step=5,
                help="Hai alert cách nhau xa hơn số này sẽ coi là hai sự cố khác nhau.")

inc = build_incidents(gap)                       # tính trước để lấy số KPI lớn nhất thật
mx = max(int(inc["n_kpis"].max()), 2)            # thanh trượt chạy tới đúng max (theo GAP)
min_kpi = c2.select_slider("Chỉ hiện sự cố có ≥ số KPI", options=list(range(1, mx + 1)),
                           value=1,
                           help="Lọc bớt sự cố lẻ tẻ, tập trung vào sự cố diện rộng.")

inc_f = inc[inc["n_kpis"] >= min_kpi].copy()

m1, m2, m3 = st.columns(3)
m1.metric("Tổng sự cố", f"{len(inc):,}")
m2.metric(f"Sự cố có ≥{min_kpi} KPI", f"{len(inc_f):,}")
m3.metric("Sự cố nhiều KPI nhất", f"{int(inc['n_kpis'].max())} KPI")

# --- Biểu đồ timeline: mỗi sự cố một chấm (x=thời gian, y=số alert, cỡ=số KPI) ---
st.markdown("**Tổng quan** — mỗi chấm một sự cố · cao = nhiều alert · to = nhiều KPI")
fig = go.Figure(go.Scatter(
    x=inc_f["start"], y=inc_f["n_alerts"], mode="markers",
    marker=dict(size=inc_f["n_kpis"] * 4 + 4, color=inc_f["n_kpis"],
                colorscale="YlOrRd", showscale=True,
                colorbar=dict(title="số KPI"), line=dict(width=0.5, color="#555")),
    customdata=inc_f[["incident", "n_kpis", "duration_min"]].values,
    hovertemplate="Sự cố #%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M}<br>"
                  "%{y} alert · %{customdata[1]} KPI · %{customdata[2]:.0f} phút<extra></extra>"))
fig.update_layout(height=320, margin=dict(l=0, r=0, t=6, b=0),
                  yaxis=dict(title="số alert", type="log"), xaxis_title="thời gian bắt đầu")
st.plotly_chart(fig, width="stretch")

# --- Bảng timeline ---
def _dur(m):
    h, mm = divmod(int(m), 60)
    return f"{h}h{mm:02d}" if h else f"{mm} phút"


inc_sorted = inc_f.sort_values("n_alerts", ascending=False)   # dùng chung cho bảng + selectbox
show = inc_sorted.copy()
show["Bắt đầu"] = show["start"].dt.strftime("%Y-%m-%d %H:%M")
show["Kéo dài"] = show["duration_min"].map(_dur)
show["Đúng/Giả"] = show["n_tp"].astype(str) + " / " + show["n_fp"].astype(str)
tbl = show[["incident", "Bắt đầu", "Kéo dài", "n_alerts", "n_kpis", "Đúng/Giả", "kpis"]]
tbl.columns = ["#", "Bắt đầu", "Kéo dài", "Alert", "KPI", "Đúng/Giả", "Các KPI"]
st.caption(f"{len(inc_f):,} sự cố (sắp theo số alert giảm dần).")
st.dataframe(tbl, width="stretch", height=340, hide_index=True)

# --- Chọn một sự cố (chuẩn bị cho phần truy nguyên nhân) ---
st.markdown("##### Chọn một sự cố để xem chi tiết")
# nhãn tính sẵn 1 lần (tránh lookup lặp trong format_func mỗi lần render)
labels = {r.incident: f"#{r.incident} · {r.start:%Y-%m-%d %H:%M} · "
                      f"{r.n_alerts} alert · {r.n_kpis} KPI"
          for r in inc_sorted.itertuples()}
sel = st.selectbox("Sự cố", inc_sorted["incident"].tolist(),
                   format_func=lambda i: labels[i])
row = inc[inc["incident"] == sel].iloc[0]
st.caption(f"Đã chọn **sự cố #{sel}** — {row['start']:%Y-%m-%d %H:%M} → {row['end']:%H:%M}, "
           f"{row['n_alerts']} alert trên {row['n_kpis']} KPI.")

# ───────────────────────── B. Root cause ─────────────────────────
st.divider()
st.subheader("B · Nguyên nhân gốc")

rc = root_cause(gap, sel)

if len(rc) < 2:
    st.info(f"Sự cố #{sel} chỉ có 1 KPI (`{rc['kpi'].iloc[0]}`) — nguyên nhân gốc "
            "chính là nó, không cần so sánh.")
else:
    st.caption("`root_score` = trung bình 3 dấu hiệu (chuẩn hoá 0–1): **kêu sớm** + "
               "**vượt ngưỡng mạnh** + **tỷ lệ alert đúng**. Cao nhất = nghi là gốc nhất.")

    # --- Top-3 thẻ ---
    top3 = rc.head(3)
    medals = ["Top 1", "Top 2", "Top 3"]
    for col, medal, (_, r) in zip(st.columns(3), medals, top3.iterrows()):
        col.metric(f"{medal} {r['kpi']}", f"{r['root_score']:.2f}")
        col.caption(f"sớm {r['earliness']:.2f} · mạnh {r['severity']:.2f} · "
                    f"tin {r['reliability']:.2f}  ·  kêu thứ {int(r['rank_time'])}")

    st.markdown("**Vì sao** — mỗi thanh = `root_score`, chia theo 3 dấu hiệu đóng góp")
    d = rc.sort_values("root_score")
    fig = go.Figure()
    for name, col in [("earliness", "#4c78a8"), ("severity", "#f58518"),
                      ("reliability", "#54a24b")]:
        fig.add_bar(y=d["kpi"], x=d[name] / 3, orientation="h", name=name,
                    marker_color=col)
    fig.update_layout(barmode="stack", height=max(200, 34 * len(d)),
                      margin=dict(l=0, r=0, t=6, b=0),
                      legend=dict(orientation="h", y=1.06),
                      xaxis_title="đóng góp vào root_score")
    st.plotly_chart(fig, width="stretch")

    # --- SHAP cho KPI gốc (#1) ---
    root_kpi = rc["kpi"].iloc[0]
    st.markdown(f"**`{root_kpi}` bất thường vì feature nào — SHAP**")
    with st.spinner("Đang tính SHAP…"):
        imp = _shap(root_kpi)
    if imp is None:
        st.warning(f"Không tính được SHAP cho `{root_kpi}` (thiếu model IF-feature "
                   "hoặc chưa cài `shap`).")
    else:
        fs = imp.head(10)[::-1]
        figs = go.Figure(go.Bar(x=fs.values, y=fs.index, orientation="h",
                                marker_color="#54a24b"))
        figs.update_layout(height=320, margin=dict(l=0, r=0, t=6, b=0),
                           xaxis_title="|SHAP| trung bình (mức đẩy điểm bất thường)")
        st.plotly_chart(figs, width="stretch")

    r0 = rc.iloc[0]
    first_txt = ("kêu đầu tiên" if int(r0["rank_time"]) == 1
                 else f"kêu thứ {int(r0['rank_time'])}")
    others = [k for k in rc["kpi"].tolist()[1:]]
    st.markdown("**Giải thích**")
    st.info(
        f"Sự cố #{sel} có {row['n_kpis']} KPI cùng lỗi trong {_dur(row['duration_min'])}. "
        f"Nghi ngờ `{root_kpi}` là nguyên nhân gốc: nó {first_txt} trong sự cố, "
        f"{r0['reliability'] * 100:.0f}% cảnh báo là thật ({int(r0['n_tp'])}/{int(r0['n_alerts'])}). "
        f"Các KPI còn lại ({', '.join(f'`{k}`' for k in others[:4])}"
        f"{'…' if len(others) > 4 else ''}) nhiều khả năng bị ảnh hưởng theo.\n\n"
        "Đây là phỏng đoán từ dấu hiệu (thời gian + mức độ + độ tin cậy), "
        "không phải quan hệ nhân quả đã chứng minh, dữ liệu ẩn danh không có thông "
        "tin phụ thuộc thật giữa các service.")
