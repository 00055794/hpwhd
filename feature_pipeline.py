"""
Full Feature Pipeline
=====================
Assembles ALL 55 features from 11 user inputs.

Feature groups (55 total):
  User inputs  (9 active in model): LONGITUDE, LATITUDE, YEAR, CONDITION,
               TOTAL_FLOORS, FURNITURE, CEILING, MATERIAL, "TOTAL AREA"
               (ROOMS and FLOOR collected from UI but not in feature list)
  Geographic   (2): REGION, segment_code
  City         (2): is_almaty, city_target_enc
  OSM dist.    (4): dist_to_pharmacy_km, dist_to_hospital_km,
                    dist_to_kindergarten_km, dist_to_main_road_km
  Stat/Macro  (26): srednmes_zarplata, index_real_zarplaty, CPI indices,
                    construction volumes, population stats, etc.
  Panel        (4): building_last_known_price, months_since_building_last_listed,
                    building_listing_count_historical, building_price_volatility
  Price index  (4): price_index_current, index_momentum_3m,
                    index_momentum_yoy, index_vs_5yr_trend
  BFE          (1): building_fixed_effect (shrinkage-regularised building premium)
  New v2       (3): floor_ratio, building_age, is_old_panel

Only feature_list.json columns are passed to the model (in exact order).
"""
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from osm_distances import OSMDistances
from region_grid import RegionGrid
from stat_loader import StatLoader

DATA_DIR  = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "nn_model"

# 11 fields collected from the web form
USER_FEATURES = [
    "ROOMS", "LONGITUDE", "LATITUDE", "TOTAL_AREA",
    "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION",
    "CEILING", "MATERIAL", "YEAR",
]

# City detection uses the same KNN city encoder as training:
# training assigned is_almaty/is_astana from CITY column string matching,
# and the city encoder KNN predicts the nearest CITY from coordinates.
# This correctly handles suburbs (Kaskelen→is_almaty=0, Kosshy→is_astana=0)
# that would be incorrectly captured by coordinate bounding boxes.


class FeaturePipeline:
    """
    Loads all lookup resources once at startup.
    Call assemble(input_dict) → pd.DataFrame with model-ready features.
    """

    def __init__(self):
        self.region_grid   = RegionGrid()
        self.stat_loader   = StatLoader()
        self.osm_distances = OSMDistances()

        # KNN city encoder: lat/lon → nearest city name → is_almaty / is_astana
        #                                                   → CITY_int  (v2 new)
        # Matches training notebook cell 173: is_almaty from CITY string matching
        from sklearn.neighbors import KNeighborsClassifier as _KNN
        import numpy as _np_city
        city_enc_path = DATA_DIR / "city_encoder.json"
        if city_enc_path.exists():
            with open(city_enc_path, "r", encoding="utf-8") as f:
                _city_data = json.load(f)
            _centroids   = _city_data["centroids"]
            _city_coords = _np_city.deg2rad(
                [[c["LATITUDE"], c["LONGITUDE"]] for c in _centroids]
            )
            _city_names    = [c["CITY"] for c in _centroids]
            _city_to_int   = _city_data.get("city_to_int", {})
            self._city_knn = _KNN(n_neighbors=1, metric="haversine")
            self._city_knn.fit(_city_coords, list(range(len(_city_names))))
            self._city_knn_names   = _city_names
            self._city_to_int: dict = _city_to_int
            print(f"FeaturePipeline: city KNN loaded ({len(_city_names)} cities)")
        else:
            self._city_knn        = None
            self._city_knn_names  = []
            self._city_to_int     = {}
            print("FeaturePipeline: city_encoder.json not found — city flags will be 0")

        # City target encoding: city_name → median real log-price from training set
        # (v2 new — captures city price level for all 476 cities)
        city_tenc_path = DATA_DIR / "city_target_enc.json"
        if city_tenc_path.exists():
            with open(city_tenc_path, "r", encoding="utf-8") as f:
                _tenc = json.load(f)
            self._global_city_median: float = float(_tenc.pop("_global_median", 13.046))
            self._city_target_enc: dict = {k: float(v) for k, v in _tenc.items()}
            print(f"FeaturePipeline: city_target_enc loaded ({len(self._city_target_enc)} cities)")
        else:
            self._city_target_enc     = {}
            self._global_city_median  = 13.046
            print("FeaturePipeline: city_target_enc.json not found — using global median")

        # Segments GeoJSON for spatial join
        seg_path = DATA_DIR / "segments_fine_heuristic_polygons.geojson"
        self.segments_gdf = gpd.read_file(seg_path)

        # Segment encoder — prefer saved map (matches training encoder exactly)
        seg_map_path = DATA_DIR / "segment_code_map.json"
        if seg_map_path.exists():
            with open(seg_map_path, "r", encoding="utf-8") as f:
                self._segment_encoder = {k: int(v) for k, v in json.load(f).items()}
            print(f"FeaturePipeline: segment_code_map loaded "
                  f"({len(self._segment_encoder)} segments)")
        else:
            seg_ids = sorted(self.segments_gdf["segment_id"].dropna().unique())
            self._segment_encoder = {sid: code for code, sid in enumerate(seg_ids)}
            print(f"FeaturePipeline: segment encoder built on-the-fly "
                  f"({len(self._segment_encoder)} segments)")

        # Feature list required by the model (exact order)
        with open(MODEL_DIR / "feature_list.json", "r", encoding="utf-8") as f:
            self.feature_list: list = json.load(f)
        print(f"FeaturePipeline: model expects {len(self.feature_list)} features")

        # Stat feature medians — used as fillna fallback (instead of 0.0)
        meta_path = MODEL_DIR / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._stat_medians: dict = meta.get("stat_feature_medians", {})
            print(f"FeaturePipeline: stat medians loaded: {len(self._stat_medians)} features")
        else:
            self._stat_medians = {}
            print("FeaturePipeline: metadata.json not found — using 0 for missing stat features")

        # BMN price index for price_index_current and index_momentum_3m
        # CRITICAL FIX (2026-04): training de-nominated PRICE_ln by the row's
        # listing-quarter index (base 2025Q4=1.0). So BOTH the feature value and
        # the back-transform multiplier must be the listing-quarter index — NOT
        # the last key in the JSON (which may be a future-quarter FORECAST).
        pi_path = MODEL_DIR / "price_index.json"
        if pi_path.exists():
            with open(pi_path, "r", encoding="utf-8") as f:
                self._price_index: dict = json.load(f)
#changed part in 07.2026:
            ##self._pi_keys = sorted(self._price_index.keys())
            ### Default "current" quarter = today (for single /predict with no CREATED_AT)
            ##today = datetime.now()
            ##self._default_quarter = f"{today.year}Q{((today.month - 1) // 3) + 1}"
            ##pi_block = self._price_index_block(self._default_quarter)

            #price_index.json contains SYNTHETIC values past 2026Q2:
            #2026Q3 == 2026Q2**2 / 2026Q1 (bit-exact one-step momentum carry)
            #2026Q4 == 2026Q3 (flat carry > index_momentum_3m == 0.0)
            #2012Q1..Q3 == the same scalar (backward fill, before the data start)
            #Serving a fabricated deflator shifts every prediction. Clamp to the last 
            #quarter backed by observed data; this self-heals when the file is refreshed. 
            SYNTHETIC = set(self._price_index.pop("_synthetic", []))
            self._pi_keys = sorted(k for k in self._price_index if k not in SYNTHETIC)
            today = datetime.now()
            today_q = f"{today.year}Q{((today.month - 1) // 3) +1}"
            self._default_quarter = min(today_q, self._pi_keys[-1]) # lex == chrono for YYYYQN
            if self._default_quarter != today_q:
                print(f"FeaturePipeline: WARNING wall-clock {today_q} is past the last observed "
                      f"index quarter {self._pi_keys[-1]}, clamping. Refresh price_index.json "
                      f"and re-fit region_calibration.json.")
            pi_block = self._price_index_block(self._default_quarter)
#changed part in 07.2026.            

            self._price_index_current = pi_block["price_index_current"]
            self._index_momentum_3m   = pi_block["index_momentum_3m"]
            self._index_momentum_yoy  = pi_block["index_momentum_yoy"]
            self._index_vs_5yr_trend  = pi_block["index_vs_5yr_trend"]
            print(f"FeaturePipeline: price_index loaded "
                  f"({len(self._pi_keys)} quarters, {self._pi_keys[0]}..{self._pi_keys[-1]}), "
                  f"default Q={self._default_quarter}  "
                  f"pi={self._price_index_current:.4f}  "
                  f"3m={self._index_momentum_3m:+.4f}  "
                  f"yoy={self._index_momentum_yoy:+.4f}  "
                  f"vs5y={self._index_vs_5yr_trend:+.4f}")
        else:
            self._price_index = {}
            self._pi_keys     = []
            self._default_quarter     = ""
            self._price_index_current = 1.0
            self._index_momentum_3m   = 0.0
            self._index_momentum_yoy  = 0.0
            self._index_vs_5yr_trend  = 0.0
            print("FeaturePipeline: price_index.json not found — using defaults (1.0)")

        # Region-specific building panel feature medians
        # building_last_known_price uses region-level median (national fallback = 359_366)
        # building_price_volatility uses region-level std capped at 300_000
        blp_path = MODEL_DIR / "region_blp_medians.json"
        _national_blp = 359_366.0
        _national_vol = 13_668.0
        if blp_path.exists():
            with open(blp_path, "r", encoding="utf-8") as f:
                _blp_data = json.load(f)
            _national_blp = float(_blp_data.get("_national", {}).get("blp_median", 359_366.0))
            # Reverse-map region code → region name for O(1) lookup
            enc_path = DATA_DIR / "region_grid_encoder.json"
            with open(enc_path, "r", encoding="utf-8") as f:
                _enc = json.load(f)
            _code_to_name = {v: k for k, v in _enc.items()}
            self._blp_by_code: dict = {
                code: float(_blp_data.get(name, {}).get("blp_median", _national_blp))
                for code, name in _code_to_name.items()
            }
            self._vol_by_code: dict = {
                code: float(min(_blp_data.get(name, {}).get("blp_std", _national_vol), 300_000.0))
                for code, name in _code_to_name.items()
            }
            print(f"FeaturePipeline: region BLP medians loaded "
                  f"({len(self._blp_by_code)} regions, national={_national_blp:,.0f})")
        else:
            self._blp_by_code = {}
            self._vol_by_code = {}
            print("FeaturePipeline: region_blp_medians.json not found, using national defaults")
        self._national_blp = _national_blp
        self._national_vol = _national_vol

        # Segment-level BLP medians (more granular than region; 2031 segments)
        seg_blp_path = MODEL_DIR / "segment_blp_medians.json"
        if seg_blp_path.exists():
            with open(seg_blp_path, "r", encoding="utf-8") as f:
                _seg_blp_data = json.load(f)
            # Keys are string segment codes in the JSON file
            self._blp_by_seg: dict = {
                int(k): float(v["blp_median"])
                for k, v in _seg_blp_data.items()
            }
            self._vol_by_seg: dict = {
                int(k): float(v["blp_std"])
                for k, v in _seg_blp_data.items()
            }
            print(f"FeaturePipeline: segment BLP medians loaded "
                  f"({len(self._blp_by_seg)} segments)")
        else:
            self._blp_by_seg = {}
            self._vol_by_seg = {}
            print("FeaturePipeline: segment_blp_medians.json not found")

        # Per-building BLP lookup (407 000 buildings from 14-year panel)
        # Key: "{lat:.4f}_{lon:.4f}_{year}_{total_floors}_{material}"
        # Value: last_real_price, last_date, appreciation_rate_annualized,
        #        volatility, count
        blp_lookup_path = MODEL_DIR / "building_blp_lookup.json"
        if blp_lookup_path.exists():
            with open(blp_lookup_path, "r", encoding="utf-8") as f:
                self._building_blp: dict = json.load(f)
            print(f"FeaturePipeline: building BLP lookup loaded "
                  f"({len(self._building_blp):,} buildings)")
        else:
            self._building_blp = {}
            print("FeaturePipeline: building_blp_lookup.json not found, "
                  "falling back to region medians")

        # Building fixed-effect lookup (shrinkage-regularised building premium)
        # Key: same fingerprint as BLP; Value: shrunk log-price premium (±0.73 range)
        bfe_lookup_path = MODEL_DIR / "building_fe_lookup.json"
        if bfe_lookup_path.exists():
            with open(bfe_lookup_path, "r", encoding="utf-8") as f:
                self._building_fe: dict = json.load(f)
            print(f"FeaturePipeline: building FE lookup loaded "
                  f"({len(self._building_fe):,} buildings)")
        else:
            self._building_fe = {}
            print("FeaturePipeline: building_fe_lookup.json not found, BFE will be 0.0")

        # ── Per-region calibration (multiplicative post-prediction correction) ──
        # Computed from a leakage-free recent window; alpha = median(actual/pred).
        # Applied in main.py / batch path as: pred_final = pred_base * alpha[REGION]
        cal_path = DATA_DIR / "region_calibration.json"
        if cal_path.exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                _cal = json.load(f)
            self._region_alpha: dict = {
                name: float(v.get("alpha", 1.0))
                for name, v in _cal.get("regions", {}).items()
            }
            self._calibration_meta = {
                "window":        _cal.get("calibration_window", ""),
                "method":        _cal.get("method", ""),
                "n_regions":     len(self._region_alpha),
                "shrink_below":  _cal.get("shrink_below_n"),
            }
            non_unit = sum(1 for a in self._region_alpha.values() if abs(a - 1.0) > 1e-6)
            print(f"FeaturePipeline: region calibration loaded "
                  f"({len(self._region_alpha)} regions, {non_unit} non-unit alphas, "
                  f"window={self._calibration_meta['window']})")
        else:
            self._region_alpha = {}
            self._calibration_meta = {}
            print("FeaturePipeline: region_calibration.json not found "
                  "(no regional calibration will be applied)")

    # ── Region calibration lookup ─────────────────────────────────────────────
    def get_region_alpha(self, lat: float, lon: float) -> float:
        """Return the multiplicative price-calibration factor for (lat, lon).
        Returns 1.0 if the region is unknown or no calibration has been loaded."""
        if not self._region_alpha:
            return 1.0
        region_name = self.region_grid.get_region_name(lat, lon)
        if region_name == "Unknown":
            return 1.0
        return float(self._region_alpha.get(region_name, 1.0))

    # ── Price-index helpers ───────────────────────────────────────────────────
    @staticmethod
    def _date_to_quarter(dt) -> str:
        return f"{dt.year}Q{((dt.month - 1) // 3) + 1}"

    def _price_index_block(self, quarter: str) -> dict:
        """Compute price_index_current + 3m/yoy/5yr-trend for a given listing quarter.
        Clamps to available range if quarter is before/after the JSON."""
        if not self._pi_keys:
            return dict(price_index_current=1.0, index_momentum_3m=0.0,
                        index_momentum_yoy=0.0, index_vs_5yr_trend=0.0)
        if quarter in self._price_index:
            idx = self._pi_keys.index(quarter)
        elif quarter < self._pi_keys[0]:
            idx = 0
        elif quarter > self._pi_keys[-1]:
            idx = len(self._pi_keys) - 1
        else:
            idx = max(i for i, k in enumerate(self._pi_keys) if k <= quarter)
        cur  = float(self._price_index[self._pi_keys[idx]])
        prev = float(self._price_index[self._pi_keys[max(0, idx - 1)]])
        lag4 = float(self._price_index[self._pi_keys[max(0, idx - 4)]])
        start = max(0, idx - 19)
        ma20 = float(np.mean([float(self._price_index[self._pi_keys[j]])
                              for j in range(start, idx + 1)]))
        return dict(
            price_index_current=cur,
            index_momentum_3m=cur / prev - 1.0,
            index_momentum_yoy=cur / lag4 - 1.0,
            index_vs_5yr_trend=cur / ma20 - 1.0,
        )

    # ── Segment code lookup ───────────────────────────────────────────────────
    def _get_segment_code(self, lat: float, lon: float) -> int:
        point  = Point(lon, lat)
        pt_gdf = gpd.GeoDataFrame([{"geometry": point}], crs="EPSG:4326")
        joined = gpd.sjoin(pt_gdf,
                           self.segments_gdf[["segment_id", "geometry"]],
                           how="left", predicate="within")

        if len(joined) == 0 or pd.isna(joined["segment_id"].iloc[0]):
            dists  = self.segments_gdf.geometry.distance(point)
            seg_id = self.segments_gdf.loc[dists.idxmin(), "segment_id"]
        else:
            seg_id = joined["segment_id"].iloc[0]

        return self._segment_encoder.get(seg_id, -1)

    # ── City lookup ───────────────────────────────────────────────────────────
    def _city_features(self, lat: float, lon: float) -> dict:
        """Returns city-related features: is_almaty, city_target_enc.
        KNN finds nearest city centroid → city name → all derived features.
        is_astana and is_shymkent removed — zero LGB importance, covered by city_target_enc."""
        import numpy as _np_city
        if self._city_knn is not None:
            idx       = self._city_knn.predict(_np_city.deg2rad([[lat, lon]]))[0]
            city_name = self._city_knn_names[idx]
            city_lower = city_name.lower()
            is_almaty   = int('almaty'    in city_lower or 'алматы'     in city_lower)
            city_tenc   = self._city_target_enc.get(city_name,
                              self._global_city_median)
        else:
            is_almaty   = 0
            city_tenc   = self._global_city_median
        return {
            "is_almaty":      is_almaty,
            "city_target_enc": city_tenc,
        }

    # ── Main assembly ─────────────────────────────────────────────────────────
    def assemble(self, user_input: dict, listing_quarter: str | None = None) -> pd.DataFrame:
        """
        Build a pd.DataFrame with all 50+ features. Returns only the
        columns in feature_list.json in the exact training order.

        listing_quarter: optional "YYYYQN" (e.g. "2026Q1"). If omitted,
          uses the quarter of today's date. CRITICAL for correct back-transform.
        """
        lat = float(user_input["LATITUDE"])
        lon = float(user_input["LONGITUDE"])

        # ── 1. User inputs (rename TOTAL_AREA → "TOTAL AREA" to match training) ──
        row: dict = {k: float(user_input[k]) for k in USER_FEATURES}
        row["TOTAL AREA"] = row.pop("TOTAL_AREA")

        # ── 2. Geographic derived ─────────────────────────────────────────────
        row["REGION"]       = self.region_grid.get_code(lat, lon)
        row["segment_code"] = self._get_segment_code(lat, lon)
        city_feats          = self._city_features(lat, lon)
        row.update(city_feats)    # is_almaty, city_target_enc
        region_name         = self.region_grid.get_region_name(lat, lon)

        # ── 3. Statistical / macro features (25+) ────────────────────────────
        stat = self.stat_loader.model_features(lat, lon, region_name=region_name)
        row.update(stat)

        # ── 4. OSM distances (4 features used by model) ───────────────────────
        dist = self.osm_distances.get_distances(lat, lon)
        row.update(dist)

        # ── 5. Price-index features — per-row listing-quarter aware ──────────
        # Training de-nominated PRICE_ln by the row's listing-quarter pi.
        # For /predict without a listing date, use today's quarter.
        if listing_quarter is None:
            listing_quarter = self._default_quarter
        pi_block = self._price_index_block(listing_quarter)
        row.update(pi_block)

        # ── 6. Building panel features ────────────────────────────────────────
        # Priority: per-building lookup from 14-year training panel (407 k buildings).
        # Falls back to region-level median when the building is not in the panel.
        # Fingerprint: "{lat:.4f}_{lon:.4f}_{year}_{total_floors}_{material}"
        year         = int(user_input["YEAR"])
        total_floors = int(user_input["TOTAL_FLOORS"])
        material     = int(user_input["MATERIAL"])
        fingerprint  = f"{lat:.4f}_{lon:.4f}_{year}_{total_floors}_{material}"

        region_code = row["REGION"]
        blp_median  = self._blp_by_code.get(region_code, self._national_blp)
        vol_median  = self._vol_by_code.get(region_code, self._national_vol)

        entry = self._building_blp.get(fingerprint)
        if entry is not None:
            # ── Building found in BLP panel (any count ≥ 1) ──────────────────
            last_date    = pd.Timestamp(entry["last_date"])
            months_since = max(0.0, (pd.Timestamp.now() - last_date).days / 30.44)
            row["building_last_known_price"]             = float(entry["last_real_price"])
            row["months_since_building_last_listed"]     = months_since
            row["building_listing_count_historical"]     = float(entry["count"])
            row["building_appreciation_rate_annualized"] = 0.0
            row["building_price_volatility"]             = vol_median
        else:
            # ── Unknown building — not in 14-year BLP panel ───────────────────
            row["building_last_known_price"]             = blp_median
            row["months_since_building_last_listed"]     = 999.0
            row["building_listing_count_historical"]     = 0.0
            row["building_appreciation_rate_annualized"] = 0.0
            row["building_price_volatility"]             = vol_median

        # Building fixed effect (shrinkage-regularised log-premium, 0.0 for unknown buildings)
        row["building_fixed_effect"] = float(self._building_fe.get(fingerprint, 0.0))

        # ── 7. New derived structural features (v2) ───────────────────────────
        # building_age MUST match training: `2025 - YEAR`, clipped to [0, 120].
        # (Training script uses a fixed 2025 anchor — see scripts/retrain_model_v2_trimmed.py).
        # Changing the anchor here would cause train/inference drift.
        year_val         = float(user_input["YEAR"])
        total_floors_val = float(user_input["TOTAL_FLOORS"])
        floor_val        = float(user_input["FLOOR"])
        material_val     = int(user_input["MATERIAL"])
        row["floor_ratio"]  = floor_val / max(total_floors_val, 1.0)
        row["building_age"] = float(max(0, min(120, 2025 - year_val)))
        row["is_old_panel"] = int(material_val == 2 and year_val < 1991)

        # ── 8. Build DataFrame, fill missing, select model columns ────────────
        df = pd.DataFrame([row])
        for feat in self.feature_list:
            if feat not in df.columns:
                df[feat] = float(self._stat_medians.get(feat, 0.0))

        return df[self.feature_list]

    # ── Display info for UI ───────────────────────────────────────────────────
    def get_display_info(self, lat: float, lon: float) -> dict:
        region_name = self.region_grid.get_region_name(lat, lon)
        return {
            "region_name": region_name,
            "distances":   self.osm_distances.get_distances(lat, lon),
            "stat":        self.stat_loader.display_features(lat, lon, region_name=region_name),
        }
