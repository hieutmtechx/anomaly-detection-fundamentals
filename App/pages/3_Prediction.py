"""Panel 3 — Live Prediction.

Chọn KPI → lấy cửa sổ giá trị thật hoặc tự dán → gọi FastAPI /predict →
hiển thị nhãn normal/anomaly, score, confidence.

Cửa sổ dài bao nhiêu là do API quyết định `min_pts` của từng KPI, không cố định:
KPI dùng feature `deseason_resid` cần đủ một chu kỳ ngày (1440 điểm với bước 60s)
vì trend là trung vị trượt trên cửa sổ đó; KPI không dùng thì 65 điểm là đủ.
Cửa sổ được cắt trên LƯỚI THỜI GIAN ĐỀU — xem `common.grid_series`.
"""
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

from common import grid_series

API = "http://localhost:8000"

st.set_page_config(page_title="Live Prediction", layout="wide")
st.title("Panel 3 · Live Prediction")

# ───────── 0. Kiểm tra API + lấy danh sách KPI kèm yêu cầu dữ liệu ─────────
try:
    meta_api = requests.get(f"{API}/kpis", timeout=5).json()
    kpis, info = meta_api["kpis"], meta_api.get("info", {})
except Exception:
    st.error("Không kết nối được API. Hãy bật server ở terminal khác:\n\n"
             "`uvicorn api:app --app-dir Serving --reload --port 8000`\n\nrồi tải lại trang.")
    st.stop()

kpi = st.sidebar.selectbox(
    "Chọn KPI", kpis,
    format_func=lambda k: f"{k} · {info.get(k, {}).get('min_pts', '?')} điểm/lần")
src = st.sidebar.radio("Nguồn dữ liệu vào", ["Điểm thật (chọn trên chuỗi)", "Tự dán giá trị"])

need = int(info.get(kpi, {}).get("min_pts", 201))       # số điểm API yêu cầu
step_s = int(info.get(kpi, {}).get("step_s", 60))
deseason = info.get(kpi, {}).get("needs_deseason", False)

st.sidebar.caption(
    f"KPI này cần **{need:,} điểm**/lần chấm "
    f"(≈ {need * step_s / 3600:.1f} giờ lịch sử)."
)


def _detail_msg(resp) -> str:
    """Rút thông báo lỗi đọc được từ response API.

    Lỗi của `infer.py` (thiếu điểm, feature NaN) trả `detail` là chuỗi. Lỗi kiểm
    kiểu của Pydantic trả `detail` là list dict {loc, msg, ...} — in thẳng ra rất
    khó đọc, nên gộp thành "trường: thông điệp".
    """
    try:
        d = resp.json().get("detail")
    except Exception:
        return resp.text[:200]
    if isinstance(d, str):
        return d
    if isinstance(d, list):
        parts = []
        for e in d:
            loc = ".".join(str(x) for x in e.get("loc", []) if x != "body")
            parts.append(f"{loc}: {e.get('msg', '')}" if loc else e.get("msg", ""))
        return " · ".join(parts)
    return str(d)


@st.cache_data(show_spinner=False)
def valid_ends(k8: str, win: int) -> np.ndarray:
    """Các vị trí kết thúc mà cửa sổ `win` điểm không dính gap dài (không có NaN)."""
    g = grid_series(k8)
    bad = g["value_filled"].isna().values.astype(int)
    if len(bad) < win:
        return np.array([], dtype=int)
    c = np.convolve(bad, np.ones(win, dtype=int), mode="valid")
    return np.nonzero(c == 0)[0] + win - 1


g = grid_series(kpi)
values = ts_end = win = None

# ───────── 1. Chuẩn bị cửa sổ giá trị ─────────
if src == "Điểm thật (chọn trên chuỗi)":
    ends = valid_ends(kpi, need)
    if len(ends) == 0:
        st.error(f"Không có cửa sổ {need:,} điểm nào liền mạch cho KPI này "
                 "(dữ liệu thủng quá nhiều).")
        st.stop()

    lab = g["label"].values
    key = f"pos_{kpi}"
    if key not in st.session_state:
        st.session_state[key] = int(ends[len(ends) // 2])

    c1, c2 = st.columns(2)
    e_anom = ends[lab[ends] == 1]
    if c1.button(f"Nhảy tới điểm anomaly ({len(e_anom):,} chỗ)",
                 disabled=len(e_anom) == 0, width="stretch"):
        st.session_state[key] = int(np.random.choice(e_anom))
    e_norm = ends[lab[ends] == 0]
    if c2.button(f"Nhảy tới điểm normal ({len(e_norm):,} chỗ)",
                 disabled=len(e_norm) == 0, width="stretch"):
        st.session_state[key] = int(np.random.choice(e_norm))

    pos = st.slider("Vị trí điểm cần chấm (điểm cuối cửa sổ)",
                    int(need - 1), int(len(g) - 1), int(st.session_state[key]))
    st.session_state[key] = pos

    win = g.iloc[pos - need + 1: pos + 1]
    if win["value_filled"].isna().any():
        n_nan = int(win["value_filled"].isna().sum())
        st.warning(
            f"Cửa sổ này dính **gap dài** ({n_nan:,}/{need:,} điểm mất, không nội suy "
            "được) nên không chấm được. Dùng nút ở trên để nhảy tới vị trí liền mạch.")
        st.stop()

    values = win["value_filled"].tolist()
    ts_end = int(win["timestamp"].iloc[-1])
    truth = "anomaly" if int(win["label"].iloc[-1]) == 1 else "normal"
    st.caption(f"Cửa sổ **{need:,} điểm** ({win['dt'].iloc[0]:%Y-%m-%d %H:%M} → "
               f"**{win['dt'].iloc[-1]:%Y-%m-%d %H:%M}**) · ")
else:
    st.info(f"KPI `{kpi}` cần đúng **{need:,} giá trị** liên tiếp, cách nhau {step_s}s.")
    txt = st.text_area("Dán giá trị (phân tách bằng phẩy / khoảng trắng / xuống dòng)",
                       height=120, placeholder="1.2, 1.3, 1.1, ...")
    if txt.strip():
        try:
            values = [float(x) for x in txt.replace(",", " ").split()]
            ts_end = int(g["timestamp"].iloc[-1])     # mượn mốc thời gian cuối của chuỗi
            if len(values) < need:
                st.warning(f"Mới có **{len(values):,}** giá trị, cần ≥ **{need:,}**.")
                values = None
            else:
                st.caption(f"Đã nhận **{len(values):,}** giá trị.")
        except ValueError:
            st.warning("Có giá trị không phải số — kiểm tra lại.")
            values = None

# ───────── 2. Vẽ cửa sổ (khi chọn điểm thật) ─────────
if win is not None:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=win["dt"], y=win["value_filled"], mode="lines",
                               line=dict(width=1, color="#4c78a8"), name="value"))
    fig.add_trace(go.Scattergl(x=[win["dt"].iloc[-1]], y=[win["value_filled"].iloc[-1]],
                               mode="markers", name="điểm chấm",
                               marker=dict(color="#e45756", size=12, symbol="circle-open",
                                           line=dict(width=3))))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

# ───────── 3. Gọi /predict + hiển thị ─────────
if values is not None:
    try:
        r = requests.post(f"{API}/predict",
                          json={"kpi_id": kpi, "values": values, "ts_end": ts_end}, timeout=30)
    except Exception as e:
        st.error(f"Lỗi gọi API: {e}"); st.stop()

    if r.status_code != 200:
        st.warning(f"API trả {r.status_code}: {_detail_msg(r)}")
    else:
        d = r.json()
        anom = d["label"] == "anomaly"
        col = "#e45756" if anom else "#54a24b"
        st.markdown(f"<div style='padding:14px;border-radius:8px;background:{col};color:white;"
                    f"font-size:24px;font-weight:700;text-align:center'>"
                    f"{'⚠️ ANOMALY' if anom else '✓ NORMAL'}</div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Anomaly score", f"{d['score']:.4f}",
                  delta=f"{d['score'] - d['threshold']:+.4f} vs ngưỡng")
        m2.metric("Ngưỡng", f"{d['threshold']:.4f}")
        m3.metric("Confidence", f"{d['confidence'] * 100:.1f}%" if d["confidence"] is not None else "—")

        lo, hi = min(d["score"], d["threshold"]), max(d["score"], d["threshold"])
        pad = (hi - lo) or abs(hi) or 0.1
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=d["score"], number={"valueformat": ".4f"},
            gauge={"axis": {"range": [lo - pad, hi + pad]}, "bar": {"color": col},
                   "threshold": {"line": {"color": "black", "width": 4},
                                 "thickness": 0.9, "value": d["threshold"]}}))
        gauge.update_layout(height=240, margin=dict(l=20, r=20, t=10, b=0))
        st.plotly_chart(gauge, width="stretch")
        st.caption(f"Model IF-feature dùng {d['features_used']} feature trên "
                   f"{d['n_points_used']:,} điểm. Vạch đen = ngưỡng; score vượt ngưỡng ⇒ "
                   "anomaly. Confidence = phân vị score so lịch sử.")
