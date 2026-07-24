"""FastAPI phục vụ dự đoán bất thường predict_one.
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import available_kpis, load_bundle, predict_one

app = FastAPI(title="Anomaly Detection API", version="0.1")


BUNDLES = {k: load_bundle(k) for k in available_kpis()}


class PredictRequest(BaseModel):
    kpi_id: str
    values: list[float]
    ts_end: int


@app.get("/kpis")
def list_kpis():
    """Danh sách KPI đang phục vụ được + yêu cầu dữ liệu của từng KPI.

    `min_pts` khác nhau giữa các KPI: KPI dùng `deseason_resid` cần đủ một chu kỳ
    ngày (1440 điểm với bước 60s) vì trend là trung vị trượt cửa sổ đó; KPI không
    dùng thì 65 điểm là đủ. Client phải đọc số này để biết cần gửi bao nhiêu điểm.
    """
    return {
        "kpis": list(BUNDLES),
        "info": {k: {"min_pts": int(b["min_pts"]), "step_s": int(b["step_s"]),
                     "n_features": len(b["features"]),
                     "needs_deseason": "deseason_resid" in b["features"]}
                 for k, b in BUNDLES.items()},
    }


@app.post("/predict")
def predict(req: PredictRequest):
    bundle = BUNDLES.get(req.kpi_id)
    if bundle is None:
        raise HTTPException(404, f"KPI '{req.kpi_id}' chưa có serve bundle. "
                                 f"Đang có: {list(BUNDLES)}")
    result = predict_one(req.values, req.ts_end, bundle)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result