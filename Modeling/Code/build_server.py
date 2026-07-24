"""Gom serve bundle cho Panel 3 (Live Prediction) — TẤT CẢ KPI có model IF.
Mỗi KPI -> serve_<kpi>.joblib

Tái tạo ĐÚNG pipeline lúc train (time_split -> preprocess_all -> build_features)
để lấy ra hai thứ mà một cửa sổ ngắn lúc serve không tự tính được:

  • season_prof — profile mùa vụ theo phút-trong-ngày. Lúc train nó được lấy
    trung vị trên hàng chục nghìn điểm (mỗi mốc phút ~90 mẫu). Một cửa sổ 1 ngày
    chỉ có 1 mẫu / mốc phút ⇒ trung vị của 1 số chính là nó ⇒ deseason_resid sụp
    về 0.

  • min_pts — số điểm tối thiểu client phải gửi để feature của điểm cuối khớp 
    với lúc train. KPI dùng deseason_resid cần đủ `period` điểm (1 ngày)
    vì trend là trung vị trượt cửa sổ `period`; KPI không dùng thì 65 là đủ
    (lookback dài nhất là lag_60 / rmean_60).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "Modeling" / "Artifacts"
CFG_PATH = ROOT / "Modeling" / "ML" / "IsolationForest" / "if_per_kpi_config.json"

sys.path.insert(0, str(ROOT / "FE_FS" / "Code"))
sys.path.insert(0, str(ROOT / "Modeling" / "Code"))
from eval_protocol import time_split_per_kpi          # noqa: E402
from features import build_features                   # noqa: E402
from preprocess import preprocess_all                 # noqa: E402

BASE_MIN_PTS = 65


def lookback_from_names(features):
    """Lookback dài nhất suy từ hậu tố số trong tên feature (lag_60 -> 60)."""
    lb = 0
    for c in features:
        tail = c.rsplit("_", 1)[-1]
        if tail.isdigit():
            lb = max(lb, int(tail))
    return lb


def main():
    cfg = {r["kpi"]: r for r in json.load(open(CFG_PATH, encoding="utf-8"))}
    df = pd.read_csv(ROOT / "Data" / "train.csv")
    df.columns = ["timestamp", "value", "label", "kpi"]
    df["k8"] = df["kpi"].str[:8]
    scores = pd.read_parquet(ART / "scores_IF_feat.parquet")

    kpis = sorted(p.stem.replace("iffeat_", "") for p in ART.glob("iffeat_*.joblib"))
    print(f"Tìm thấy {len(kpis)} model IF\n")

    ok, skipped = 0, []
    for kpi in kpis:
        art = ART / f"iffeat_{kpi}.joblib"
        if kpi not in cfg:
            skipped.append((kpi, "không có trong if_per_kpi_config.json")); continue
        d = joblib.load(art)
        feats = list(d["features"])

        gk = df[df.k8 == kpi].drop(columns="k8").copy()   # bỏ k8: khớp đúng input lúc train
        gk = time_split_per_kpi(gk, train_frac=0.6, val_frac=0.2)
        p = preprocess_all(gk, max_gap_points=5, norm_method="robust") \
            .sort_values("timestamp").reset_index(drop=True)
        F = build_features(p)
        prof, period = F.attrs["season_prof"], F.attrs["period"]

        # --- số điểm tối thiểu client phải gửi ---
        needs_deseason = "deseason_resid" in feats
        min_pts = max(BASE_MIN_PTS, lookback_from_names(feats) + 5)
        if needs_deseason:
            min_pts = max(min_pts, period)     # trend cần đầy cửa sổ `period`

        step = int(pd.Series(np.diff(np.sort(gk.timestamp.values))).mode().iloc[0])
        sc = scores.loc[scores.kpi == kpi, "score"].values

        bundle = dict(
            kpi=kpi,
            model=d["model"],
            features=feats,
            thr=float(cfg[kpi]["b_thr_pw"]),
            step_s=step,
            min_pts=int(min_pts),
            period=int(period),
            season_prof=prof,                  # ĐÓNG BĂNG — serve tra bảng, không tự tính
            score_ref=np.quantile(sc, np.linspace(0, 1, 101)) if len(sc) else None,
        )
        out = ART / f"serve_{kpi}.joblib"
        joblib.dump(bundle, out)
        ok += 1
        print(f"[OK] {kpi}: {len(feats):>2} feat · thr={bundle['thr']:>8.4f} · "
              f"step={step:>3}s · period={period:>4} · min_pts={min_pts:>4}"
              f"{'  (cần deseason)' if needs_deseason else ''}")

    print(f"\nXong: {ok} bundle -> {ART}")
    for kpi, why in skipped:
        print(f"[BỎ QUA] {kpi}: {why}")


if __name__ == "__main__":
    main()
