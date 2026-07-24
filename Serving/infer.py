"""Inference dùng chung cho Panel 3 / FastAPI.
    available_kpis()               -> list KPI có serve bundle
    load_bundle(kpi)               -> bundle (hoặc None)
    predict_one(values, ts_end, b) -> dict score/label/confidence
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "Modeling" / "Artifacts"
sys.path.insert(0, str(ROOT / "FE_FS" / "Code"))
from features import build_features


def available_kpis():
    return sorted(p.stem.replace("serve_", "") for p in ART.glob("serve_*.joblib"))


def load_bundle(kpi):
    f = ART / f"serve_{kpi}.joblib"
    return joblib.load(f) if f.exists() else None


def predict_one(values, ts_end, bundle):
    values = np.asarray(values, dtype=float)
    n, step = len(values), bundle["step_s"]
    if n < bundle["min_pts"]:
        return {"error": f"Cần ≥{bundle['min_pts']} điểm, nhận {n}."}

    # dựng lại trục thời gian đều, kết thúc tại ts_end 
    ts = int(ts_end) - (n - 1 - np.arange(n)) * step
    p = pd.DataFrame({"value_filled": values, "timestamp": ts,
                      "masked": False, "split": "train"})

    # profile mùa vụ cất từ lúc build vì một cửa sổ ~1 ngày chỉ có
    # 1 mẫu / mốc phút nên nếu để build_features tự tính thì deseason_resid sụp về 0.
    F = build_features(p, period=bundle.get("period"),
                       season_prof=bundle.get("season_prof"))
    row = F[bundle["features"]].iloc[-1]
    if row.isna().any():
        return {"error": "Thiếu lịch sử hoặc dữ liệu không đều.",
                "nan_features": list(row.index[row.isna()])}

    score = float(-bundle["model"].decision_function(row.values.reshape(1, -1))[0])
    thr = float(bundle["thr"])
    ref = bundle.get("score_ref")
    conf = float((np.asarray(ref) < score).mean()) if ref is not None else None
    return {"kpi": bundle["kpi"], "score": score, "threshold": thr,
            "label": "anomaly" if score >= thr else "normal",
            "confidence": conf, "n_points_used": n,
            "features_used": len(bundle["features"])}