"""
FastAPI backend for KZ Real Estate Price Estimator.
Serves Jinja2 HTML with Yandex Maps and handles NN model predictions.

Architecture (flat modules at project root):
  feature_pipeline.py  – assembles all features (user inputs + REGION_GRID +
                          segment_code + stat + OSM distances)
  nn_inference.py      – loads nn_model/ artifacts and predicts price in KZT

The model receives only the features listed in nn_model/feature_list.json
(currently 13; upgrades to 47 after running scripts/save_artifacts.py).
"""
import io
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

BASE_DIR   = Path(__file__).resolve().parent
YANDEX_KEY = os.getenv("YANDEX_MAPS_API_KEY", "")

# ── Global resources (loaded once at startup) ─────────────────────────────────
_pipeline = None
_nn       = None


# ── Scoring helpers (UI-only, do not affect the model) ────────────────────────
def _score_from_distance(d_km, ideal: float, max_km: float) -> int:
    """Convert distance in km to a 0–95 closeness score (smooth exponential
    decay). Caps at 95 so even a doorstep amenity never claims a "perfect"
    score, and decays smoothly so urban variability is reflected (no plateau
    at 100). Calibration: score(0)=95, score(max_km)≈25, score(ideal)≈85."""
    if d_km is None:
        return 50
    try:
        d = float(d_km)
    except Exception:
        return 50
    if d <= 0:
        return 95
    import math
    # k so that 95·e^(−k·max_km) ≈ 25  →  k = ln(95/25)/max_km
    span = max(0.2, float(max_km))
    k = math.log(95.0 / 25.0) / span
    score = 95.0 * math.exp(-k * d)
    return int(round(max(15, min(95, score))))


def _build_proximity(distances: dict) -> dict:
    return {
        "pharmacy":     {"label": "Аптека",        "km": distances.get("dist_to_pharmacy_km"),
                         "score": _score_from_distance(distances.get("dist_to_pharmacy_km"),     0.5, 2.5)},
        "hospital":     {"label": "Больница",      "km": distances.get("dist_to_hospital_km"),
                         "score": _score_from_distance(distances.get("dist_to_hospital_km"),     1.0, 5.0)},
        "kindergarten": {"label": "Детский сад",   "km": distances.get("dist_to_kindergarten_km"),
                         "score": _score_from_distance(distances.get("dist_to_kindergarten_km"), 0.5, 2.0)},
        "main_road":    {"label": "Главная улица", "km": distances.get("dist_to_main_road_km"),
                         "score": _score_from_distance(distances.get("dist_to_main_road_km"),    0.3, 2.0)},
    }


def _build_livability(prox: dict, condition: int, year: int,
                      ceiling: float, total_floors: int) -> dict:
    # Infrastructure: weighted blend (hospital matters more than pharmacy)
    infra = round(prox["pharmacy"]["score"]    * 0.30 +
                  prox["hospital"]["score"]    * 0.40 +
                  prox["kindergarten"]["score"] * 0.30)

    # Transport: closeness to a main road
    transport = prox["main_road"]["score"]

    # Condition combines renovation grade, age and ceiling height
    cond_grade = max(1, min(5, int(condition)))
    cond_part  = (cond_grade - 1) / 4 * 100         # 0–100 from grade
    age        = max(0, 2026 - int(year))
    age_part   = max(0, 100 - min(age, 60) * 1.4)   # newer is better
    ceil_part  = max(0, min(100, (float(ceiling) - 2.4) / 0.8 * 100))
    cond_score = round(cond_part * 0.55 + age_part * 0.30 + ceil_part * 0.15)
    cond_score = max(35, min(100, cond_score))

    # Overall = arithmetic average of the three displayed sub-scores so the
    # "Комфорт" badge equals what the user sees in the legend.
    overall_raw = round((int(infra) + int(transport) + int(cond_score)) / 3)

    # ── Adaptive lower bound based on local urbanity ─────────────────────
    # Rationale: the 50–95 floor was calibrated for built-up areas. In
    # remote regions with no construction nearby (steppe, fields, villages
    # with no kindergarten/hospital), the four POI distances are all large
    # and clipping at 50 over-states comfort. Use the best of the four
    # POI scores as an "urbanity" proxy:
    #   urbanity ≥ 60  → built-up, floor = 50
    #   urbanity ≤ 30  → rural / no construction, floor = 10
    #   else           → linear interpolation
    urbanity = max(
        int(prox["pharmacy"]["score"]),
        int(prox["hospital"]["score"]),
        int(prox["kindergarten"]["score"]),
        int(prox["main_road"]["score"]),
    )
    if urbanity >= 60:
        floor_score = 50
    elif urbanity <= 30:
        floor_score = 10
    else:
        # Linear: u=30→10, u=60→50  ⇒  floor = 10 + (u-30) * (40/30)
        floor_score = int(round(10 + (urbanity - 30) * (40.0 / 30.0)))

    overall = max(floor_score, min(95, overall_raw))

    if urbanity <= 30:
        narrative = ("Удалённая локация: рядом нет крупной инфраструктуры. "
                     "Подходит для тех, кто ценит уединение и природу.")
    elif overall >= 85:
        narrative = "Отличный выбор: высокий комфорт проживания, развитая инфраструктура и удобная транспортная доступность."
    elif overall >= 70:
        narrative = "Хороший выбор: сбалансированная локация с комфортными условиями для повседневной жизни."
    else:
        narrative = "Достойный вариант: подходит для размеренной жизни, есть потенциал для улучшения комфорта."
    return {
        "infrastructure": int(infra),
        "transport":      int(transport),
        "condition":      int(cond_score),
        "overall":        int(overall),
        "narrative":      narrative,
    }


def _build_drivers(payload: dict, livability: dict) -> list:
    """Return 6 price drivers with integer percentage weights summing to exactly 100."""
    cond        = int(payload.get("CONDITION", 3))
    area        = float(payload.get("TOTAL_AREA", 60))
    year        = int(payload.get("YEAR", 2010))
    floor       = int(payload.get("FLOOR", 1))
    total_floor = max(int(payload.get("TOTAL_FLOORS", 1)), 1)
    material    = int(payload.get("MATERIAL", 3))

    floor_ratio = floor / total_floor
    raw = [
        ("Площадь",        max(15.0, min(34.0, 24.0 + (area - 60) * 0.18))),
        ("Локация",        max(10.0, min(28.0, 18.0 + (livability["infrastructure"] - 60) * 0.18))),
        ("Состояние",      max(8.0,  min(20.0, 6.0  + cond * 2.4))),
        ("Год постройки",  max(5.0,  min(15.0, 14.0 - max(0, 2026 - year) * 0.18))),
        ("Этажность",      max(5.0,  min(14.0, 6.0  + (1 - abs(floor_ratio - 0.5)) * 12))),
        ("Материал",       max(5.0,  min(13.0, 5.0  + material * 1.6))),
    ]
    total = sum(v for _, v in raw)
    # Largest-remainder rounding so integer percents sum to exactly 100.
    scaled = [(n, v / total * 100) for n, v in raw]
    floors = [(n, int(p), p - int(p)) for n, p in scaled]
    deficit = 100 - sum(f for _, f, _ in floors)
    order = sorted(range(len(floors)), key=lambda i: floors[i][2], reverse=True)
    pct = [f for _, f, _ in floors]
    for k in range(deficit):
        pct[order[k % len(order)]] += 1
    return [{"name": n, "value": int(p)} for (n, _, _), p in zip(floors, pct)]


def _confidence_band(price: float) -> dict:
    """90% confidence band: ±10% around the predicted nominal price."""
    half = price * 0.10
    return {
        "level": 90,
        "lower": int(round(price - half)),
        "upper": int(round(price + half)),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _nn

    from feature_pipeline import FeaturePipeline
    from nn_inference import NNInference

    print("Loading FeaturePipeline …")
    _pipeline = FeaturePipeline()

    print("Loading NNInference …")
    _nn = NNInference()

    # ── Startup validation (fail-fast on broken deployment) ───────────────
    n_feat   = len(_nn.feature_list)
    n_meta   = int(_nn.metadata.get("n_features", n_feat))
    n_weights = _nn.nn_model.net[0].in_features
    assert n_feat == n_meta == n_weights, (
        f"Feature count mismatch: feature_list={n_feat}, "
        f"metadata.n_features={n_meta}, NN.input_dim={n_weights}"
    )
    assert _nn.lgb_model is not None, \
        "lgb_model.txt missing — ensemble cannot run LGB+NN blending"
    assert _nn.ridge_meta is not None, \
        "ridge_meta.joblib missing — ensemble weighting unavailable"
    assert _pipeline.feature_list == _nn.feature_list, \
        "Pipeline and NN disagree on feature_list"
    print(f"✅ Startup validation OK — {n_feat} features, LGB+NN+Ridge loaded")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="KZ House Price Estimator", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Request schema ────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    ROOMS:        int   = Field(..., ge=1,    le=20,   description="Number of rooms")
    LONGITUDE:    float = Field(..., ge=46.0, le=87.0, description="Property longitude (WGS-84)")
    LATITUDE:     float = Field(..., ge=40.0, le=55.0, description="Property latitude (WGS-84)")
    TOTAL_AREA:   float = Field(..., gt=0,    le=1000, description="Total area m²")
    FLOOR:        int   = Field(..., ge=1,    le=100,  description="Floor number")
    TOTAL_FLOORS: int   = Field(..., ge=1,    le=100,  description="Total floors in building")
    FURNITURE:    int   = Field(..., ge=1,    le=3,    description="1=No furniture  2=Partial  3=Full")
    CONDITION:    int   = Field(..., ge=1,    le=5,    description="1=Rough/Open plan  2=Needs renovation  3=Neat/Average  4=Good  5=Fresh renovation")
    CEILING:      float = Field(..., ge=1.5,  le=10.0, description="Ceiling height in metres")
    MATERIAL:     int   = Field(..., ge=1,    le=4,    description="1=Other  2=Panel  3=Monolith  4=Brick")
    YEAR:         int   = Field(..., ge=1900, le=2030, description="Year built")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"yandex_key": YANDEX_KEY},
    )


@app.post("/predict")
async def predict(data: PredictionInput):
    if _pipeline is None or _nn is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — retry in a moment.")

    try:
        user_input = data.model_dump()
        lat = data.LATITUDE
        lon = data.LONGITUDE

        # Assemble model-ready feature matrix
        features_df = _pipeline.assemble(user_input)

        # Extract derived codes for display
        region_grid_code = int(features_df["REGION"].iloc[0]) \
            if "REGION" in features_df.columns else -1
        segment_code_val = int(features_df["segment_code"].iloc[0]) \
            if "segment_code" in features_df.columns else -1

        # NN prediction → price per sqm in 2025Q4 KZT
        price_per_sqm = float(_nn.predict_kzt(features_df)[0])
        # Per-region calibration (multiplicative post-hoc correction,
        # learnt from 2026 Jan-Feb leakage-free window).
        alpha = _pipeline.get_region_alpha(lat, lon)
        price_per_sqm *= alpha
        price_kzt     = price_per_sqm * data.TOTAL_AREA

        # Display-only info (region name, distances, stat summary for UI)
        display = _pipeline.get_display_info(lat, lon)

        # ── Scoring data for the right-column widgets ────────────────────────
        proximity  = _build_proximity(display.get("distances", {}) or {})
        livability = _build_livability(proximity,
                                       int(user_input["CONDITION"]),
                                       int(user_input["YEAR"]),
                                       float(user_input["CEILING"]),
                                       int(user_input["TOTAL_FLOORS"]))
        drivers    = _build_drivers(user_input, livability)
        conf_band  = _confidence_band(price_kzt)

        return {
            "success":       True,
            "price_kzt":     round(price_kzt, 0),
            "price_per_sqm": round(price_per_sqm, 0),
            "confidence":    conf_band,
            "livability":    livability,
            "proximity":     proximity,
            "drivers":       drivers,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _nn is not None}


@app.get("/geocode")
async def geocode(q: str):
    """Server-side proxy to Nominatim. Uses curl.exe (built-in on Win10+)
    with SSPI auth so it works behind corporate NTLM proxies. Covers all
    of Kazakhstan via Nominatim's global OSM index."""
    import urllib.parse, json, subprocess, shutil
    q = (q or "").strip()
    if len(q) < 2:
        return []
    url = ("https://nominatim.openstreetmap.org/search?format=json"
           "&addressdetails=1&limit=8&countrycodes=kz"
           "&accept-language=ru,en&q=" + urllib.parse.quote(q))
    ua = "kz-real-estate-advisor/1.0 (contact: local-dev)"
    curl = shutil.which("curl") or shutil.which("curl.exe")
    last_err = None
    if curl:
        for args in (
            # 1) Use system proxy with current Windows user creds (NTLM/Negotiate)
            [curl, "-s", "-S", "--max-time", "12", "--ssl-no-revoke",
             "--proxy-anyauth", "--proxy-user", ":", "-A", ua, url],
            # 2) Fallback: no proxy
            [curl, "-s", "-S", "--max-time", "12", "--ssl-no-revoke",
             "--noproxy", "*", "-A", ua, url],
        ):
            try:
                out = subprocess.run(args, capture_output=True, timeout=15)
                if out.returncode == 0 and out.stdout:
                    try:
                        return json.loads(out.stdout.decode("utf-8", "replace"))
                    except Exception as e:
                        last_err = f"parse: {e}"
                        continue
                last_err = (out.stderr or b"").decode("utf-8", "replace") or f"rc={out.returncode}"
            except Exception as e:
                last_err = str(e)
    else:
        last_err = "curl.exe not found"
    raise HTTPException(status_code=502, detail=f"Geocoder error: {last_err}")


# ── Batch predict ─────────────────────────────────────────────────────────────
REQUIRED_COLS = ["ROOMS", "LATITUDE", "LONGITUDE", "TOTAL_AREA", "FLOOR",
                 "TOTAL_FLOORS", "FURNITURE", "CONDITION", "CEILING", "MATERIAL", "YEAR"]


@app.post("/batch")
async def batch_predict(file: UploadFile = File(...)):
    if _pipeline is None or _nn is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        contents = await file.read()
        if file.filename and file.filename.lower().endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="replace")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse file: {exc}") from exc

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    df = df.head(1000)  # safety cap
    results: list[dict[str, Any]] = []
    n_fail = 0
    for idx, row in df.iterrows():
        rec: dict[str, Any] = {c: (None if pd.isna(row[c]) else
                                    int(row[c]) if isinstance(row[c], (int,)) else
                                    float(row[c]) if hasattr(row[c], '__float__') else row[c])
                               for c in REQUIRED_COLS}
        try:
            user_input = {c: row[c] for c in REQUIRED_COLS}
            features_df = _pipeline.assemble(user_input)
            price_per_sqm = float(_nn.predict_kzt(features_df)[0])
            alpha = _pipeline.get_region_alpha(
                float(row["LATITUDE"]), float(row["LONGITUDE"]))
            price_per_sqm *= alpha
            price_kzt      = price_per_sqm * float(row["TOTAL_AREA"])
            rec["pred_price_per_sqm"] = round(price_per_sqm, 0)
            rec["pred_price_kzt"]     = round(price_kzt, 0)
            rec["region_alpha"]       = round(alpha, 4)
            rec["error"]              = None
        except Exception as exc:
            rec["pred_price_per_sqm"] = None
            rec["pred_price_kzt"]     = None
            rec["region_alpha"]       = None
            rec["error"]              = f"{type(exc).__name__}: {exc}"[:200]
            n_fail += 1
            print(f"[batch] row {idx}: {rec['error']}", flush=True)
        results.append(rec)
    if n_fail:
        print(f"[batch] {n_fail}/{len(df)} rows failed", flush=True)
    return jsonable_encoder(results)


@app.post("/batch/download/xlsx")
async def batch_download_xlsx(rows: list[dict[str, Any]]):
    df  = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=predictions.xlsx"},
    )


@app.get("/template/csv")
async def template_csv():
    # Header + one annotated sample row so users know the expected encoding
    lines = [
        ",".join(REQUIRED_COLS),
        "2,43.2567,76.9286,65.0,5,12,3,5,2.7,3,2015",
        "# FURNITURE: 1=No furniture  2=Partial  3=Full",
        "# CONDITION: 1=Rough/Open plan  2=Needs renovation  3=Neat/Average  4=Good  5=Fresh renovation",
        "# MATERIAL:  1=Other  2=Panel  3=Monolith  4=Brick",
    ]
    return StreamingResponse(
        io.StringIO("\n".join(lines)),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=template.csv"},
    )


@app.get("/template/xlsx")
async def template_xlsx():
    sample = {
        "ROOMS": [2], "LATITUDE": [43.2567], "LONGITUDE": [76.9286],
        "TOTAL_AREA": [65.0], "FLOOR": [5], "TOTAL_FLOORS": [12],
        "FURNITURE": [3], "CONDITION": [5], "CEILING": [2.7],
        "MATERIAL": [3], "YEAR": [2015],
    }
    notes = {
        "ROOMS": [""], "LATITUDE": [""], "LONGITUDE": [""],
        "TOTAL_AREA": [""], "FLOOR": [""], "TOTAL_FLOORS": [""],
        "FURNITURE": ["1=No furniture  2=Partial  3=Full"],
        "CONDITION": ["1=Rough  2=Needs reno  3=Neat  4=Good  5=Fresh reno"],
        "CEILING": [""],
        "MATERIAL": ["1=Other  2=Panel  3=Monolith  4=Brick"],
        "YEAR": [""],
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(sample).to_excel(writer, sheet_name="Data", index=False)
        pd.DataFrame(notes).to_excel(writer, sheet_name="Notes", index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=template.xlsx"},
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
