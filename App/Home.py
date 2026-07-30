"""Chạy: streamlit run App/Home.py"""
import streamlit as st

from common import kpi_meta

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.title("Anomaly Detection Dashboard")
st.caption("Dashboard trên dữ liệu KPI time series (AIOps)")

meta = kpi_meta("train")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số KPI", meta["k8"].nunique())
c2.metric("Tổng điểm dữ liệu", f"{int(meta['n'].sum()):,}")
c3.metric("Tỷ lệ anomaly TB", f"{meta['anom_rate'].mean() * 100:.2f}%")
c4.metric("Số cohort thời gian", meta.groupby(['year', 'step_s']).ngroups)

st.markdown(
    """
    ### Các panel
    - **1 · Data Explorer** — khám phá chuỗi thời gian, phân bố nhãn, tương quan giữa các KPI.
    - **2 · Model Lab** — thử nghiệm các mô hình. 
    - **3 · Live Prediction** — chọn KPI, gọi FastAPI /predict, hiển thị nhãn normal/anomaly, score.    
    - **4 · Model Monitoring** — nhật ký alert và chỉ số drift.
    - **5 · Root Cause Analysis** — phân tích nguyên nhân gốc.
    """
)

with st.expander("Về cấu trúc dữ liệu"):
    st.markdown(
        "26 KPI không cùng khoảng thời gian — chúng thuộc nhiều cohort rời "
        "nhau (một nhóm ở 2016 lấy mẫu 300s, phần còn lại ở 2017 lấy mẫu 60s). "
        "Vì vậy correlation heatmap chỉ tính trên nhóm 16 KPI có lưới 60s khớp nhau."
    )
