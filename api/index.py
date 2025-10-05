from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# --- FastAPI App Initialization ---
app = FastAPI()

# CRITICAL FIX for 405: Explicitly allow POST and OPTIONS methods globally.
# This ensures that CORS preflight requests (which use OPTIONS) succeed,
# preventing a 405 before the actual POST request is even sent.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    # Allow POST (for the /metrics endpoint) and OPTIONS (for CORS preflight)
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Pydantic Data Model for Request Body ---
class MetricsRequest(BaseModel):
    """Input validation for the request body."""
    regions: List[str]
    threshold_ms: int

# --- Telemetry Data Loading (MOCK DATA) ---
# NOTE: Use your actual telemetry bundle data here.
MOCK_TELEMETRY_RECORDS = [
    {"region": "amer", "latency_ms": 100, "is_up": 1},
    {"region": "amer", "latency_ms": 150, "is_up": 1},
    {"region": "amer", "latency_ms": 160, "is_up": 1},
    {"region": "amer", "latency_ms": 140, "is_up": 1},
    {"region": "amer", "latency_ms": 180, "is_up": 0},
    {"region": "emea", "latency_ms": 120, "is_up": 1},
    {"region": "emea", "latency_ms": 155, "is_up": 1},
    {"region": "emea", "latency_ms": 170, "is_up": 1},
    {"region": "emea", "latency_ms": 150, "is_up": 0},
    {"region": "emea", "latency_ms": 130, "is_up": 1},
]

# Load data into a DataFrame
try:
    TELEMETRY_DF = pd.DataFrame(MOCK_TELEMETRY_RECORDS)
except Exception as e:
    print(f"FATAL: Error loading initial telemetry data: {e}")
    TELEMETRY_DF = None


def calculate_metrics(df: pd.DataFrame, threshold: int) -> Dict[str, Any]:
    """Calculates all required metrics for a given DataFrame of records."""
    if df.empty:
        return {
            "avg_latency": 0.0, "p95_latency": 0.0,
            "avg_uptime": 0.0, "breaches": 0
        }

    latencies = df['latency_ms'].values
    uptimes = df['is_up'].values

    # Calculate required metrics
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    avg_uptime = float(np.mean(uptimes))
    breaches = int(np.sum(latencies > threshold))

    return {
        "avg_latency": round(avg_latency, 1),
        "p95_latency": round(p95_latency, 1),
        "avg_uptime": round(avg_uptime, 1),
        "breaches": breaches
    }


# --- Endpoint Definition ---

@app.post("/metrics", response_model=List[Dict[str, Any]])
def post_metrics(request_data: MetricsRequest):
    """
    Accepts a POST request and returns latency and uptime metrics
    for the specified regions against a latency threshold.
    """
    if TELEMETRY_DF is None:
        return [{"error": "Telemetry data could not be initialized."}]

    results = []
    threshold = request_data.threshold_ms
    
    for region_name in request_data.regions:
        region_df = TELEMETRY_DF[TELEMETRY_DF['region'] == region_name.lower()]
        metrics = calculate_metrics(region_df, threshold)

        results.append({
            "region": region_name.lower(),
            **metrics
        })

    return results

# Basic root path for Vercel/health checks
@app.get("/")
def read_root():
    return {"message": "eShopCo Latency Metrics Service is running. Use POST /metrics"}
