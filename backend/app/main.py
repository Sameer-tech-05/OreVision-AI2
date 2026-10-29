from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import rasterio

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rasterio.warp import transform, transform_bounds


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="OreVision AI API",
    description=(
        "AI-driven iron ore prospectivity mapping "
        "using Sentinel-2-derived features and XGBoost"
    ),
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://orevision-ai-3.onrender.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROSPECTIVITY_DIR = PROCESSED_DIR / "prospectivity"
GSI_DIR = PROCESSED_DIR / "gsi_sentinel"

SENTINEL_FEATURE_DIR = (
    PROCESSED_DIR
    / "sentinel2"
    / "expanded"
    / "features"
)

MODEL_DIR = BASE_DIR / "models"


# ============================================================
# PROSPECTIVITY MAP FILES
# ============================================================

PROBABILITY_MAP = (
    PROSPECTIVITY_DIR
    / "Iron_Ore_Prospectivity_Probability.tif"
)

CLASS_MAP = (
    PROSPECTIVITY_DIR
    / "Iron_Ore_Prospectivity_Class.tif"
)
# ============================================================
# EXPORT / DOWNLOAD GEOTIFF FILES
# ============================================================

PROBABILITY_DOWNLOAD_MAP = (
    PROSPECTIVITY_DIR
    / "OreVision_AI_Prospectivity_Probability.tif"
)

CLASS_DOWNLOAD_MAP = (
    PROSPECTIVITY_DIR
    / "OreVision_AI_Prospectivity_Class_Color.tif"
)
PREVIEW_MAP = (
    PROSPECTIVITY_DIR
    / "Iron_Ore_Prospectivity_Preview.png"
)

WEB_OVERLAY = (
    PROSPECTIVITY_DIR
    / "Iron_Ore_Prospectivity_WebOverlay.png"
)


# ============================================================
# GSI FILES
# ============================================================

GSI_ANCHORS = (
    GSI_DIR
    / "GSI_Spatial_Anchors_Step4.csv"
)

GSI_LINKED = (
    GSI_DIR
    / "GSI_Sentinel_Linked_Anchors.csv"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

XGB_MODEL_PATH = (
    MODEL_DIR
    / "iron_ore_xgb_validated_model.joblib"
)


# ============================================================
# SENTINEL-2 FEATURE RASTERS
# ============================================================

FEATURE_RASTERS = {
    "NDVI": (
        SENTINEL_FEATURE_DIR
        / "Expanded_NDVI.tif"
    ),

    "NDMI": (
        SENTINEL_FEATURE_DIR
        / "Expanded_NDMI.tif"
    ),

    "B04_B02": (
        SENTINEL_FEATURE_DIR
        / "Expanded_B04_B02.tif"
    ),

    "B04_B11": (
        SENTINEL_FEATURE_DIR
        / "Expanded_B04_B11.tif"
    ),

    "B11_B12": (
        SENTINEL_FEATURE_DIR
        / "Expanded_B11_B12.tif"
    ),
}


VALID_MASK_RASTER = (
    SENTINEL_FEATURE_DIR
    / "Expanded_valid_mask.tif"
)


# ============================================================
# MODEL FEATURE ORDER
# ============================================================

MODEL_FEATURES = [
    "NDVI",
    "NDMI",
    "B04_B02",
    "B04_B11",
    "B11_B12",
]


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECT_NAME = "OreVision AI"

MODEL_NAME = "Validated XGBoost"

DATA_SOURCE = (
    "Sentinel-2-derived spectral features"
)

STUDY_AREA = (
    "Ballari-Vijayanagara-Sandur Iron Ore Belt, Karnataka"
)

NATIVE_CRS = "EPSG:32643"

WEB_CRS = "EPSG:4326"

RASTER_RESOLUTION = 10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def file_exists(path: Path):
    """Return True if the file exists."""
    return path.exists() and path.is_file()


def validate_coordinate(
    latitude: float,
    longitude: float,
):
    """Validate WGS84 latitude and longitude."""

    if not np.isfinite(latitude):
        raise ValueError(
            "Latitude must be a finite number."
        )

    if not np.isfinite(longitude):
        raise ValueError(
            "Longitude must be a finite number."
        )

    if latitude < -90 or latitude > 90:
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if longitude < -180 or longitude > 180:
        raise ValueError(
            "Longitude must be between -180 and 180."
        )


def get_raster_bounds_wgs84(
    raster_path: Path,
):
    """Convert raster bounds to WGS84."""

    if not file_exists(raster_path):
        return None

    try:
        with rasterio.open(raster_path) as src:

            bounds = src.bounds

            west, south, east, north = transform_bounds(
                src.crs,
                WEB_CRS,
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                densify_pts=21,
            )

            return {
                "south": float(south),
                "west": float(west),
                "north": float(north),
                "east": float(east),
            }

    except Exception as exc:

        print(
            "Could not calculate WGS84 bounds:",
            exc,
        )

        return None


def get_raster_metadata(
    raster_path: Path,
):
    """Return useful raster metadata."""

    if not file_exists(raster_path):
        return None

    try:
        with rasterio.open(raster_path) as src:

            return {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),

                "resolution_x": float(
                    src.res[0]
                ),

                "resolution_y": float(
                    src.res[1]
                ),

                "native_bounds": {
                    "left": float(
                        src.bounds.left
                    ),
                    "bottom": float(
                        src.bounds.bottom
                    ),
                    "right": float(
                        src.bounds.right
                    ),
                    "top": float(
                        src.bounds.top
                    ),
                },

                "wgs84_bounds":
                    get_raster_bounds_wgs84(
                        raster_path
                    ),
            }

    except Exception as exc:

        print(
            "Could not read raster metadata:",
            exc,
        )

        return None


# ============================================================
# LOAD GSI REFERENCE LOCATIONS
# ============================================================

def load_gsi_locations():

    locations = []

    # --------------------------------------------------------
    # Primary GSI anchor file
    # --------------------------------------------------------

    if file_exists(GSI_ANCHORS):

        try:

            df = pd.read_csv(
                GSI_ANCHORS
            )

            for _, row in df.iterrows():

                try:

                    latitude = float(
                        row["latitude"]
                    )

                    longitude = float(
                        row["longitude"]
                    )

                    if not (
                        np.isfinite(latitude)
                        and np.isfinite(longitude)
                    ):
                        continue

                    locations.append(
                        {
                            "locality": str(
                                row.get(
                                    "locality",
                                    "GSI locality",
                                )
                            ),

                            "latitude":
                                latitude,

                            "longitude":
                                longitude,

                            "source_report": str(
                                row.get(
                                    "source_report",
                                    "",
                                )
                            ),

                            "toposheet": str(
                                row.get(
                                    "toposheet",
                                    "",
                                )
                            ),

                            "type":
                                "GSI reference locality",
                        }
                    )

                except Exception:
                    continue

            if locations:
                return locations

        except Exception as exc:

            print(
                "Error loading GSI anchors:",
                exc,
            )

    # --------------------------------------------------------
    # Fallback linked file
    # --------------------------------------------------------

    if file_exists(GSI_LINKED):

        try:

            df = pd.read_csv(
                GSI_LINKED
            )

            for _, row in df.iterrows():

                try:

                    latitude = float(
                        row["latitude"]
                    )

                    longitude = float(
                        row["longitude"]
                    )

                    if not (
                        np.isfinite(latitude)
                        and np.isfinite(longitude)
                    ):
                        continue

                    locations.append(
                        {
                            "locality": str(
                                row.get(
                                    "locality",
                                    "GSI locality",
                                )
                            ),

                            "latitude":
                                latitude,

                            "longitude":
                                longitude,

                            "source_report": str(
                                row.get(
                                    "source_report",
                                    "",
                                )
                            ),

                            "toposheet": str(
                                row.get(
                                    "toposheet",
                                    "",
                                )
                            ),

                            "type":
                                "GSI reference locality",
                        }
                    )

                except Exception:
                    continue

        except Exception as exc:

            print(
                "Error loading linked GSI file:",
                exc,
            )

    return locations


# ============================================================
# CHECK SENTINEL FEATURE FILES
# ============================================================

def check_feature_files():

    missing = []

    for feature_name, path in FEATURE_RASTERS.items():

        if not file_exists(path):

            missing.append(
                f"{feature_name}: {path.name}"
            )

    return missing


# ============================================================
# SAMPLE SENTINEL-2 FEATURES
# ============================================================

def sample_sentinel_features(
    latitude: float,
    longitude: float,
):
    """
    Convert WGS84 coordinates into the Sentinel-2 raster CRS
    and sample the five trained model features.
    """

    validate_coordinate(
        latitude,
        longitude,
    )

    # --------------------------------------------------------
    # Check feature rasters
    # --------------------------------------------------------

    missing = check_feature_files()

    if missing:

        raise FileNotFoundError(
            "Missing Sentinel-2 feature raster(s): "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Reference raster
    # --------------------------------------------------------

    reference_raster = FEATURE_RASTERS["NDVI"]

    with rasterio.open(
        reference_raster
    ) as src:

        raster_crs = src.crs

        # WGS84 -> raster CRS
        xs, ys = transform(
            WEB_CRS,
            raster_crs,
            [longitude],
            [latitude],
        )

        x = float(xs[0])
        y = float(ys[0])

        # ----------------------------------------------------
        # Check study area
        # ----------------------------------------------------

        if not (
            src.bounds.left <= x <= src.bounds.right
            and
            src.bounds.bottom <= y <= src.bounds.top
        ):

            raise ValueError(
                "The selected coordinate is outside "
                "the Sentinel-2 study area."
            )

        # ----------------------------------------------------
        # Find raster pixel
        # ----------------------------------------------------

        row, col = src.index(
            x,
            y,
        )

        if (
            row < 0
            or row >= src.height
            or col < 0
            or col >= src.width
        ):

            raise ValueError(
                "The selected coordinate does not "
                "correspond to a valid raster pixel."
            )

    # --------------------------------------------------------
    # Sample five features
    # --------------------------------------------------------

    features = {}

    for feature_name in MODEL_FEATURES:

        raster_path = FEATURE_RASTERS[
            feature_name
        ]

        with rasterio.open(
            raster_path
        ) as src:

            if src.crs != raster_crs:

                raise ValueError(
                    f"CRS mismatch for {feature_name}: "
                    f"{src.crs} != {raster_crs}"
                )

            sampled = next(
                src.sample(
                    [(x, y)]
                )
            )[0]

            value = float(
                sampled
            )

            if not np.isfinite(value):

                raise ValueError(
                    f"{feature_name} has no valid "
                    "Sentinel-2 value at the "
                    "selected coordinate."
                )

            features[
                feature_name
            ] = value

    # --------------------------------------------------------
    # Check valid mask
    # --------------------------------------------------------

    if file_exists(
        VALID_MASK_RASTER
    ):

        with rasterio.open(
            VALID_MASK_RASTER
        ) as src:

            if src.crs != raster_crs:

                raise ValueError(
                    "Sentinel valid-mask CRS does not "
                    "match the feature raster CRS."
                )

            mask_value = float(
                next(
                    src.sample(
                        [(x, y)]
                    )
                )[0]
            )

            if (
                not np.isfinite(mask_value)
                or mask_value <= 0
            ):

                raise ValueError(
                    "The selected coordinate is not "
                    "a valid Sentinel-2 pixel. "
                    "Try another location inside "
                    "the study area."
                )

    return (
        features,
        x,
        y,
        int(row),
        int(col),
    )


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

def load_xgboost_model():

    if not file_exists(
        XGB_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Validated XGBoost model not found: "
            + str(XGB_MODEL_PATH)
        )

    try:

        model = joblib.load(
            XGB_MODEL_PATH
        )

        return model

    except Exception as exc:

        raise RuntimeError(
            "Could not load the validated "
            f"XGBoost model: {exc}"
        )


# ============================================================
# RUN XGBOOST PREDICTION
# ============================================================

def run_xgboost_prediction(
    features: dict,
):
    """
    Run validated XGBoost using the exact five
    features used during model training.
    """

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features
    ]

    if missing:

        raise ValueError(
            "Missing model feature(s): "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Create one-row DataFrame
    # --------------------------------------------------------

    X = pd.DataFrame(
        [
            [
                features[feature]
                for feature in MODEL_FEATURES
            ]
        ],
        columns=MODEL_FEATURES,
    )

    if not np.isfinite(
        X.to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "One or more Sentinel-2 feature "
            "values are invalid."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_xgboost_model()

    if not hasattr(
        model,
        "predict_proba",
    ):

        raise RuntimeError(
            "The saved XGBoost model does not "
            "support probability prediction."
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)

    if probabilities.shape[1] < 2:

        raise RuntimeError(
            "The XGBoost model does not contain "
            "both class probabilities."
        )

    probability = float(
        probabilities[0][1]
    )

    probability = float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )

    score = probability * 100.0

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if score <= 40:

        classification = "LOW"

    elif score <= 70:

        classification = "MEDIUM"

    else:

        classification = "HIGH"

    return (
        probability,
        score,
        classification,
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project":
            PROJECT_NAME,

        "status":
            "running",

        "version":
            "2.0.0",

        "model":
            MODEL_NAME,

        "data_source":
            DATA_SOURCE,

        "study_area":
            STUDY_AREA,

        "features":
            MODEL_FEATURES,

        "prediction_endpoint":
            "/api/predict-location",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status":
            "ok",

        "project":
            PROJECT_NAME,

        "backend":
            "FastAPI",

        "model":
            MODEL_NAME,

        "data_source":
            DATA_SOURCE,
    }


# ============================================================
# PROSPECTIVITY INFORMATION
# ============================================================

@app.get("/api/prospectivity/info")
def prospectivity_info():

    probability_exists = file_exists(
        PROBABILITY_MAP
    )

    class_exists = file_exists(
        CLASS_MAP
    )

    preview_exists = file_exists(
        PREVIEW_MAP
    )

    overlay_exists = file_exists(
        WEB_OVERLAY
    )

    model_exists = file_exists(
        XGB_MODEL_PATH
    )

    feature_status = {}

    for name, path in FEATURE_RASTERS.items():

        feature_status[name] = file_exists(
            path
        )

    raster_info = get_raster_metadata(
        CLASS_MAP
    )

    gsi_locations = load_gsi_locations()

    return {

        "status":
            (
                "ready"
                if (
                    probability_exists
                    and class_exists
                    and model_exists
                )
                else "not_ready"
            ),

        "project":
            PROJECT_NAME,

        "model":
            MODEL_NAME,

        "model_file":
            model_exists,

        "data_source":
            DATA_SOURCE,

        "features":
            MODEL_FEATURES,

        "feature_files":
            feature_status,

        "study_area":
            STUDY_AREA,

        "coordinate_reference_system":
            NATIVE_CRS,

        "web_coordinate_system":
            WEB_CRS,

        "resolution":
            "10 m",

        "classification": {
            "1": "LOW",
            "2": "MEDIUM",
            "3": "HIGH",
        },

        "thresholds": {
            "low": "0-40%",
            "medium": "41-70%",
            "high": "71-100%",
        },

        "files": {

            "probability_map":
                probability_exists,

            "classification_map":
                class_exists,

            "preview_map":
                preview_exists,

            "web_overlay":
                overlay_exists,
        },

        "map": {

            "bounds":
                (
                    raster_info["wgs84_bounds"]
                    if raster_info
                    else None
                ),

            "native_bounds":
                (
                    raster_info["native_bounds"]
                    if raster_info
                    else None
                ),

            "width":
                (
                    raster_info["width"]
                    if raster_info
                    else None
                ),

            "height":
                (
                    raster_info["height"]
                    if raster_info
                    else None
                ),

            "crs":
                (
                    raster_info["crs"]
                    if raster_info
                    else NATIVE_CRS
                ),

            "resolution":
                (
                    raster_info["resolution_x"]
                    if raster_info
                    else RASTER_RESOLUTION
                ),
        },

        "gsi_reference_locations":
            len(gsi_locations),

        "endpoints": {

            "probability":
                "/api/prospectivity/probability",

            "classification":
                "/api/prospectivity/class",

            "preview":
                "/api/prospectivity/preview",

            "web_overlay":
                "/api/prospectivity/web-overlay",

            "gsi_locations":
                "/api/prospectivity/gsi-locations",

            "predict_location":
                "/api/predict-location",

            "model_performance":
                "/api/model-performance",

            "download_png":
                "/api/download/prospectivity-png",

            "download_probability_geotiff":
                "/api/download/prospectivity-geotiff",

            "download_classification_geotiff":
                "/api/download/prospectivity-class",
        },

        "disclaimer": (
            "This output is generated for an "
            "academic mini-project. It is NOT a "
            "validated geological survey result "
            "and must not be used for real "
            "exploration, investment, or "
            "land-use decisions."
        ),
    }


# ============================================================
# PROBABILITY MAP
# ============================================================

@app.get(
    "/api/prospectivity/probability"
)
def get_probability_map():

    if not file_exists(
        PROBABILITY_MAP
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Prospectivity probability "
                "map not found."
            ),
        )

    return FileResponse(
        path=PROBABILITY_MAP,
        media_type="image/tiff",
        filename=PROBABILITY_MAP.name,
    )


# ============================================================
# CLASSIFICATION MAP
# ============================================================

@app.get(
    "/api/prospectivity/class"
)
def get_classification_map():

    if not file_exists(
        CLASS_MAP
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Prospectivity classification "
                "map not found."
            ),
        )

    return FileResponse(
        path=CLASS_MAP,
        media_type="image/tiff",
        filename=CLASS_MAP.name,
    )


# ============================================================
# PREVIEW MAP
# ============================================================

@app.get(
    "/api/prospectivity/preview"
)
def get_preview_map():

    if not file_exists(
        PREVIEW_MAP
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Prospectivity preview "
                "not found."
            ),
        )

    return FileResponse(
        path=PREVIEW_MAP,
        media_type="image/png",
        filename=PREVIEW_MAP.name,
    )


# ============================================================
# WEB OVERLAY
# ============================================================

@app.get(
    "/api/prospectivity/web-overlay"
)
def get_web_overlay():

    if not file_exists(
        WEB_OVERLAY
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Transparent prospectivity "
                "web overlay not found."
            ),
        )

    return FileResponse(
        path=WEB_OVERLAY,
        media_type="image/png",
        filename=WEB_OVERLAY.name,
    )


# ============================================================
# DOWNLOAD PROSPECTIVITY PNG
# ============================================================

@app.get(
    "/api/download/prospectivity-png"
)
def download_prospectivity_png():

    if not file_exists(
        PREVIEW_MAP
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Prospectivity PNG map not found."
            ),
        )

    return FileResponse(
        path=PREVIEW_MAP,
        media_type="image/png",
        filename="OreVision_AI_Prospectivity_Map.png",
    )


# ============================================================
# DOWNLOAD PROBABILITY GEOTIFF
# ============================================================

@app.get(
    "/api/download/prospectivity-geotiff"
)
def download_prospectivity_geotiff():

    if not file_exists(
        PROBABILITY_DOWNLOAD_MAP
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Exported prospectivity probability "
                "GeoTIFF not found."
            ),
        )

    return FileResponse(
        path=PROBABILITY_DOWNLOAD_MAP,
        media_type="image/tiff",
        filename=(
            "OreVision_AI_Prospectivity_Probability.tif"
        ),
    )
# ============================================================
# DOWNLOAD CLASSIFICATION GEOTIFF
# ============================================================

@app.get(
    "/api/download/prospectivity-class"
)
def download_prospectivity_class():

    if not file_exists(
        CLASS_DOWNLOAD_MAP
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Exported prospectivity classification "
                "GeoTIFF not found."
            ),
        )

    return FileResponse(
        path=CLASS_DOWNLOAD_MAP,
        media_type="image/tiff",
        filename=(
            "OreVision_AI_Prospectivity_Class_Color.tif"
        ),
    )
# ============================================================
# GSI LOCATIONS
# ============================================================

@app.get(
    "/api/prospectivity/gsi-locations"
)
def get_gsi_locations():

    locations = load_gsi_locations()

    return {
        "count":
            len(locations),

        "locations":
            locations,
    }


# ============================================================
# REAL LOCATION PREDICTION
# ============================================================

class LocationPredictionRequest(BaseModel):

    latitude: float

    longitude: float


@app.post(
    "/api/predict-location"
)
def predict_location(
    request: LocationPredictionRequest,
):

    try:

        # ----------------------------------------------------
        # Step 1: Sample Sentinel-2 features
        # ----------------------------------------------------

        (
            features,
            x_utm,
            y_utm,
            row,
            col,
        ) = sample_sentinel_features(
            request.latitude,
            request.longitude,
        )

        # ----------------------------------------------------
        # Step 2: Run XGBoost
        # ----------------------------------------------------

        (
            probability,
            score,
            classification,
        ) = run_xgboost_prediction(
            features
        )

        # ----------------------------------------------------
        # Step 3: Return result
        # ----------------------------------------------------

        return {

            "status":
                "success",

            "location": {

                "latitude":
                    request.latitude,

                "longitude":
                    request.longitude,
            },

            "raster_pixel": {

                "row":
                    row,

                "column":
                    col,
            },

            "utm_coordinates": {

                "x":
                    round(
                        x_utm,
                        3,
                    ),

                "y":
                    round(
                        y_utm,
                        3,
                    ),
            },

            "prospectivity_score":
                round(
                    score,
                    2,
                ),

            "probability":
                round(
                    probability,
                    6,
                ),

            "classification":
                classification,

            "model":
                MODEL_NAME,

            "data_source":
                DATA_SOURCE,

            "features": {

                "NDVI":
                    round(
                        float(
                            features["NDVI"]
                        ),
                        6,
                    ),

                "NDMI":
                    round(
                        float(
                            features["NDMI"]
                        ),
                        6,
                    ),

                "B04_B02":
                    round(
                        float(
                            features["B04_B02"]
                        ),
                        6,
                    ),

                "B04_B11":
                    round(
                        float(
                            features["B04_B11"]
                        ),
                        6,
                    ),

                "B11_B12":
                    round(
                        float(
                            features["B11_B12"]
                        ),
                        6,
                    ),
            },

            "feature_order":
                MODEL_FEATURES,

            "interpretation": (
                "The score is an XGBoost "
                "model-estimated prospectivity "
                "probability based on Sentinel-2-"
                "derived spectral features. It is "
                "not an iron concentration or a "
                "confirmation of underground ore."
            ),

            "disclaimer": (
                "Academic prototype only. "
                "Not a validated geological survey "
                "result and not intended for real "
                "exploration, investment, or "
                "land-use decisions."
            ),
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            "Location prediction error:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Location prediction failed: "
                + str(exc)
            ),
        )


# ============================================================
# OLD PREDICTION PAGE COMPATIBILITY
# ============================================================

class PredictionRequest(BaseModel):

    latitude: float

    longitude: float

    magnetic_anomaly: float = 0.0

    gravity_anomaly: float = 0.0

    geological_indicator: float = 0.0

    distance_to_fault_km: float = 0.0

    remote_sensing_index: float = 0.0


@app.post(
    "/api/predict"
)
def old_prediction_compatibility(
    request: PredictionRequest,
):

    try:

        result = predict_location(
            LocationPredictionRequest(
                latitude=request.latitude,
                longitude=request.longitude,
            )
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# GET PREDICTION COMPATIBILITY
# ============================================================

@app.get(
    "/api/predict"
)
def predict_get(
    latitude: Optional[float] = Query(
        None,
        description="Latitude in WGS84",
    ),

    longitude: Optional[float] = Query(
        None,
        description="Longitude in WGS84",
    ),
):

    if (
        latitude is None
        or longitude is None
    ):

        return {

            "status":
                "info",

            "message": (
                "Use POST /api/predict-location "
                "with latitude and longitude."
            ),

            "example": {

                "latitude":
                    15.194444,

                "longitude":
                    76.677778,
            },

            "model":
                MODEL_NAME,

            "features":
                MODEL_FEATURES,
        }

    try:

        return predict_location(
            LocationPredictionRequest(
                latitude=latitude,
                longitude=longitude,
            )
        )

    except HTTPException:
        raise


# ============================================================
# LEGACY GRID ENDPOINT
# ============================================================

@app.get(
    "/api/grid"
)
def legacy_grid(

    minLat: float = 15.05,

    maxLat: float = 15.30,

    minLon: float = 76.35,

    maxLon: float = 76.94,

    step: float = 0.05,
):

    if step <= 0:

        raise HTTPException(
            status_code=400,
            detail=(
                "step must be greater than zero."
            ),
        )

    points = []

    lat = minLat

    while lat <= (
        maxLat + 1e-9
    ):

        lon = minLon

        while lon <= (
            maxLon + 1e-9
        ):

            points.append(
                {
                    "latitude":
                        round(
                            lat,
                            6,
                        ),

                    "longitude":
                        round(
                            lon,
                            6,
                        ),
                }
            )

            lon += step

        lat += step

    return {

        "count":
            len(points),

        "points":
            points,

        "message": (
            "Legacy grid endpoint. "
            "Use the Sentinel-2 XGBoost "
            "prediction endpoint for actual "
            "model predictions."
        ),
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get(
    "/api/model-performance"
)
def get_model_performance():

    return {

        "model":
            "Validated XGBoost",

        "validation":
            "Stratified 5-Fold Cross-Validation",

        "features": [
            "NDVI",
            "NDMI",
            "B04_B02",
            "B04_B11",
            "B11_B12",
        ],

        "metrics": {

            "roc_auc_mean":
                0.6741,

            "roc_auc_std":
                0.0963,

            "accuracy":
                0.8683,

            "precision":
                0.1686,

            "recall":
                0.1867,

            "f1_score":
                0.1722,
        },

        "data_source":
            "Sentinel-2-derived spectral features",

        "resolution":
            "10 m",

        "study_area":
            "Ballari-Vijayanagara-Sandur Iron Ore Belt, Karnataka",

        "interpretation": (
            "These metrics represent prototype-level "
            "model validation using stratified 5-fold "
            "cross-validation. They do not represent "
            "iron concentration accuracy."
        ),

        
    }


# ============================================================
# STARTUP INFORMATION
# ============================================================

@app.on_event(
    "startup"
)
def startup_event():

    print()
    print("=" * 70)

    print(
        "                    OreVision AI"
    )

    print(
        "              FastAPI Backend Started"
    )

    print("=" * 70)
    print()

    print(
        "Study area:"
    )

    print(
        STUDY_AREA
    )

    print()

    print(
        "XGBoost model:"
    )

    print(
        XGB_MODEL_PATH
    )

    print(
        "Model exists:",
        file_exists(
            XGB_MODEL_PATH
        ),
    )

    print()

    print(
        "Sentinel-2 feature directory:"
    )

    print(
        SENTINEL_FEATURE_DIR
    )

    print()

    for feature_name, path in FEATURE_RASTERS.items():

        print(
            f"{feature_name}: "
            f"{file_exists(path)}"
        )

    print()

    print(
        "Valid mask:",
        file_exists(
            VALID_MASK_RASTER
        ),
    )

    print()

    print(
        "Probability map:",
        file_exists(
            PROBABILITY_MAP
        ),
    )

    print(
        "Classification map:",
        file_exists(
            CLASS_MAP
        ),
    )

    print(
        "Preview map:",
        file_exists(
            PREVIEW_MAP
        ),
    )

    print(
        "Web overlay:",
        file_exists(
            WEB_OVERLAY
        ),
    )

    print(
        "GSI anchors:",
        file_exists(
            GSI_ANCHORS
        ),
    )

    print()

    bounds = get_raster_bounds_wgs84(
        CLASS_MAP
    )

    if bounds:

        print(
            "WGS84 map bounds:"
        )

        print(
            bounds
        )

    print()

    print(
        "Prediction endpoint:"
    )

    print(
        "POST /api/predict-location"
    )

    print()

    print(
        "Model performance endpoint:"
    )

    print(
        "GET /api/model-performance"
    )

    print()

    print(
        "Download endpoints:"
    )

    print(
        "GET /api/download/prospectivity-png"
    )

    print(
        "GET /api/download/prospectivity-geotiff"
    )

    print(
        "GET /api/download/prospectivity-class"
    )

    print()

    print(
        "Application startup complete."
    )

    print("=" * 70)

    print()
