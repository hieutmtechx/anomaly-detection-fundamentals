"""Panel 2 data layer — đọc + chuẩn hóa kết quả các model, đọc score, SHAP.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "Modeling" / "Artifacts"

MODELS = ["IF", "ARIMA", "STL", "RL"]
MODEL_COLORS = {"IF": "#54a24b", "ARIMA": "#4c78a8", "STL": "#f58518", "RL": "#9aa0a6"}
INTERPRET = {"IF": "Thấp", "ARIMA": "TB", "STL": "TB", "RL": "Cao"}

_CFG = {
    "RL":    (ROOT/"Modeling"/"Stats"/"RL"/"rl_per_kpi_config.json",
              dict(AP="ap_test", AP_val="ap_val", ROC="roc", P="prec", R="rec",
                   F1="f1_pw", ttd="ttd_sec", thr=None, param="window")),
    "STL":   (ROOT/"Modeling"/"Stats"/"STL"/"stl_per_kpi_config.json",
              dict(AP="AP", AP_val="AP_val", ROC="ROC", P="PW_P", R="PW_R",
                   F1="PW_F1", ttd="ttd_s", thr="thr_pw", param="period")),
    "ARIMA": (ROOT/"Modeling"/"Stats"/"ARIMA"/"arima_per_kpi_config.json",
              dict(AP="AP", AP_val="AP_val", ROC="ROC", P="PW_P", R="PW_R",
                   F1="PW_F1", ttd="ttd_s", thr="thr_pw", param="order")),
    "IF":    (ROOT/"Modeling"/"ML"/"IsolationForest"/"if_per_kpi_config.json",
              dict(AP="b_AP", AP_val="b_AP_val", ROC="b_ROC", P="b_PW_P", R="b_PW_R",
                   F1="b_PW_F1", ttd=None, thr="b_thr_pw", param="b_n_feat")),
}
_SCORE = {"RL": "scores_RL", "STL": "scores_STL", "ARIMA": "scores_ARIMA", "IF": "scores_IF_feat"}
_METRICS = ["AP", "AP_val", "ROC", "P", "R", "F1", "ttd", "thr"]


@st.cache_data(show_spinner=False)
def model_table() -> pd.DataFrame:
    """Bảng dài: 1 dòng / (model, kpi) theo schema chung. Bỏ model chưa có config."""
    frames = []
    for m in MODELS:
        path, mp = _CFG[m]
        if not path.exists():
            continue
        d = pd.DataFrame(json.load(open(path, encoding="utf-8")))
        row = pd.DataFrame({"model": m, "kpi": d["kpi"].astype(str),
                            "n_anom": d["n_anom_test"]})
        for k, col in mp.items():
            if k == "param":
                row["param"] = d[col].astype(str) if col in d.columns else ""
            else:
                row[k] = d[col] if (col and col in d.columns) else np.nan
        row["interpret"] = INTERPRET[m]
        frames.append(row)
    t = pd.concat(frames, ignore_index=True)
    t["gap"] = (t["AP_val"] - t["AP"]).abs()
    return t


@st.cache_data(show_spinner=False)
def macro_table() -> pd.DataFrame:
    """Trung bình mỗi metric trên các KPI, 1 dòng / model, sắp theo AP giảm dần."""
    t = model_table()
    g = (t.groupby("model")[_METRICS].mean()
         .reindex([m for m in MODELS if m in t.model.unique()]))
    g["n_kpi"] = t.groupby("model")["kpi"].nunique()
    g["interpret"] = pd.Series(INTERPRET)
    return g.sort_values("AP", ascending=False)


@st.cache_data(show_spinner=False)
def load_scores(model: str) -> pd.DataFrame:
    return pd.read_parquet(ART / f"{_SCORE[model]}.parquet")


def best_by_val(t: pd.DataFrame) -> pd.DataFrame:
    """Mỗi KPI: model có AP_val cao nhất."""
    idx = t.dropna(subset=["AP_val"]).groupby("kpi")["AP_val"].idxmax()
    return t.loc[idx].set_index("kpi")


def shap_importance(kpi: str, top: int = 12):
    """|SHAP| trung bình theo feature cho model IF-feature của KPI."""
    import joblib
    import shap
    f = ART / f"iffeat_{kpi}.joblib"
    if not f.exists():
        return None
    d = joblib.load(f)
    sv = shap.TreeExplainer(d["model"]).shap_values(d["X_test"], check_additivity=False)
    imp = np.abs(sv).mean(0)
    return pd.Series(imp, index=d["features"]).sort_values(ascending=False).head(top)
