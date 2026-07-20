"""Panel 2 — Model Lab.

(1) Bảng so sánh model  (2) Trực quan detection trên cửa sổ thời gian  (3) SHAP.
Chọn model bằng AP_val (val), báo cáo AP_test.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from common import load_data
from modellab import (MODEL_COLORS, best_by_val, load_scores, macro_table,
                      model_table, shap_importance)

st.set_page_config(page_title="Model Lab", layout="wide")
st.title("Panel 2 · Model Lab")

t = model_table()
kpis = sorted(t["kpi"].unique())
sel = st.sidebar.selectbox("Chọn KPI (khối per-KPI, B, C)", kpis)
bv = best_by_val(t)

# ───────────────────────── A. Bảng so sánh ─────────────────────────
st.subheader("A · So sánh model")

macro = macro_table()
show = macro[["AP", "AP_val", "ROC", "P", "R", "F1", "ttd", "n_kpi", "interpret"]].copy()
show.columns = ["AP test", "AP val", "ROC", "Precision", "Recall", "F1", "TTD(s)", "n KPI", "Diễn giải"]
st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[:6]} | {"TTD(s)": "{:.0f}"},
                               na_rep="—"), width="stretch")

fig = go.Figure(go.Bar(
    x=macro["AP"], y=macro.index, orientation="h",
    marker_color=[MODEL_COLORS[m] for m in macro.index],
    text=[f"{v:.3f}" for v in macro["AP"]], textposition="outside",
    hovertemplate="%{y}: AP=%{x:.3f}<extra></extra>"))
fig.update_layout(height=230, margin=dict(l=0, r=0, t=6, b=0),
                  xaxis_title="Macro AP (test) — trung bình 18 KPI", xaxis_range=[0, 0.55],
                  yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("Xếp hạng theo AP test. Chọn model dùng AP val — "
           "ROC bị thổi trên dữ liệu lệch nên chỉ là phụ.")

st.markdown(f"Chi tiết KPI `{sel}` — best theo val: "
            f"`{bv.loc[sel, 'model']}` (AP val={bv.loc[sel, 'AP_val']:.3f}, "
            f"AP test={bv.loc[sel, 'AP']:.3f})")
sub = (t[t["kpi"] == sel].set_index("model")
       [["AP", "AP_val", "ROC", "P", "R", "F1", "ttd", "param"]]
       .reindex([m for m in ["IF", "ARIMA", "STL", "RL"] if m in t.model.values]))
sub.columns = ["AP test", "AP val", "ROC", "Precision", "Recall", "F1", "TTD(s)", "Tham số"]
st.dataframe(sub.style.format({c: "{:.3f}" for c in sub.columns[:6]} | {"TTD(s)": "{:.0f}"},
                              na_rep="—").highlight_max(subset=["AP test"], color="#2b6a2f"),
             width="stretch")
gap = abs(bv.loc[sel, "AP_val"] - bv.loc[sel, "AP"])
if gap > 0.2:
    st.warning(f" AP val và AP test lệch {gap:.2f} ở KPI này"
               ", val không đại diện cho test.")

# ───────────────────────── B. Detection ─────────────────────────
st.subheader(f"B · Kết quả phát hiện — `{sel}`")

avail = [m for m in ["IF", "ARIMA", "STL", "RL"] if m in t.model.values]
default_m = bv.loc[sel, "model"] if bv.loc[sel, "model"] in avail else avail[0]
mdl = st.radio("Model", avail, index=avail.index(default_m), horizontal=True)

sc = load_scores(mdl)
sc = sc[sc["kpi"] == sel]
if len(sc) == 0:
    st.info("Model này không có score cho KPI đã chọn.")
else:
    raw = load_data("train")
    raw = raw[raw["k8"] == sel][["timestamp", "value", "dt"]]
    d = sc.merge(raw, on="timestamp", how="left").sort_values("timestamp")

    tmin, tmax = d["dt"].min().to_pydatetime(), d["dt"].max().to_pydatetime()
    rng = st.slider("Cửa sổ thời gian", tmin, tmax, (tmin, tmax), format="YYYY-MM-DD")
    w = d[(d["dt"] >= rng[0]) & (d["dt"] <= rng[1])]

    thr_row = t[(t["kpi"] == sel) & (t["model"] == mdl)]["thr"]
    thr = float(thr_row.iloc[0]) if len(thr_row) and np.isfinite(thr_row.iloc[0]) else None
    col = MODEL_COLORS[mdl]

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.5, 0.5])
    fig2.add_trace(go.Scattergl(x=w["dt"], y=w["value"], mode="lines", name="value",
                                line=dict(width=1, color="#4c78a8")), row=1, col=1)
    a = w[w["y"] == 1]
    fig2.add_trace(go.Scattergl(x=a["dt"], y=a["value"], mode="markers", name="anomaly thật",
                                marker=dict(color="#e45756", size=5)), row=1, col=1)
    fig2.add_trace(go.Scattergl(x=w["dt"], y=w["score"], mode="lines", name=f"score {mdl}",
                                line=dict(width=1, color=col)), row=2, col=1)
    if thr is not None:
        fig2.add_hline(y=thr, line=dict(color="grey", width=1, dash="dash"), row=2, col=1)
        fl = w[w["score"] >= thr]
        fig2.add_trace(go.Scattergl(x=fl["dt"], y=fl["score"], mode="markers", name="vượt ngưỡng",
                                    marker=dict(color=col, size=5, line=dict(width=0.5, color="black"))),
                       row=2, col=1)
    fig2.update_yaxes(title_text="value", row=1, col=1)
    fig2.update_yaxes(title_text=f"score {mdl}", row=2, col=1)
    fig2.update_layout(height=520, margin=dict(l=0, r=0, t=6, b=0),
                       legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig2, width="stretch")
    cap = "Trên: chuỗi gốc + anomaly màu đỏ. Dưới: score model + đường ngưỡng (trên val) + điểm vượt ngưỡng."
    if thr is None:
        cap += f" · {mdl} không lưu ngưỡng nên bỏ đường ngưỡng."
    st.caption(cap)

# ───────────────────────── C. SHAP ─────────────────────────
st.subheader(f"C · SHAP feature importance — `{sel}`")

if bv.loc[sel, "model"] != "IF":
    st.info(f"SHAP chỉ khả dụng cho IF-feature. Model tốt nhất của KPI này là "
            f"`{bv.loc[sel, 'model']}` → không vẽ SHAP (ARIMA/STL/RL không có feature để giải thích).")
else:
    try:
        imp = shap_importance(sel)
    except ImportError:
        imp = "no_shap"
    if imp is None:
        st.warning("Không tìm thấy model IF-feature (`iffeat_*.joblib`) cho KPI này.")
    elif isinstance(imp, str):
        st.info("Chưa cài `shap`. Chạy: `C:\\Projects\\.venv\\Scripts\\python.exe -m pip install shap` "
                "rồi tải lại trang.")
    else:
        fig3 = go.Figure(go.Bar(
            x=imp.values[::-1], y=imp.index[::-1], orientation="h",
            marker_color="#54a24b",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig3.update_layout(height=380, margin=dict(l=0, r=0, t=6, b=0),
                           xaxis_title="|SHAP| trung bình (độ quan trọng)")
        st.plotly_chart(fig3, width="stretch")
        st.caption(f"Feature nào đẩy điểm bất thường lên cao nhất cho IF-feature ở KPI `{sel}`. "
                   "Tính trên mẫu ≤2000 hàng test.")
