from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# --- FastAPI App Initialization ---
app = FastAPI()

# 1. Enable CORS for POST requests from any origin
# The Vercel runtime is implicitly exposed via the handler in this file.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["POST"],  # Only allows POST requests
    allow_headers=["*"],
)

# --- Pydantic Data Model for Request Body ---
class MetricsRequest(BaseModel):
    """Defines the structure of the incoming JSON payload."""
    regions: List[str]
    threshold_ms: int

# --- Mock Telemetry Data ---
# This data simulates the content of the "telemetry bundle" for demonstration.
# In a real-world scenario, you would replace this with actual data loading logic
# (e.g., loading from a database, a file, or another service).
# Data structure: region, latency_ms, is_up (1=up, 0=down)
MOCK_TELEMETRY_RECORDS = [
    # AMER region data
    {"region": "amer", "latency_ms": 100, "is_up": 1},
    {"region": "amer", "latency_ms": 150, "is_up": 1},
    {"region": "amer", "latency_ms": 160, "is_up": 1},
    {"region": "amer", "latency_ms": 140, "is_up": 1},
    {"region": "amer", "latency_ms": 180, "is_up": 0},
    # EMEA region data
    {"region": "emea", "latency_ms": 120, "is_up": 1},
    {"region": "emea", "latency_ms": 155, "is_up": 1},
    {"region": "emea", "latency_ms": 170, "is_up": 1},
    {"region": "emea", "latency_ms": 150, "is_up": 0},
    {"region": "emea", "latency_ms": 130, "is_up": 1},
    # APAC region data (ignored in the test case)
    {"region": "apac", "latency_ms": 80, "is_up": 1},
    {"region": "apac", "latency_ms": 90, "is_up": 1},
]

# Load mock data into a DataFrame for easy metric calculation
try:
    TELEMETRY_DF = pd.DataFrame(MOCK_TELEMETRY_RECORDS)
except Exception as e:
    # Handle case where pandas is not available or data load fails
    TELEMETRY_DF = None
    print(f"Error loading telemetry data: {e}")


def calculate_metrics(df: pd.DataFrame, threshold: int) -> Dict[str, Any]:
    """Calculates all required metrics for a given DataFrame of records."""
    if df.empty:
        return {
            "avg_latency": 0.0, "p95_latency": 0.0,
            "avg_uptime": 0.0, "breaches": 0
        }

    latencies = df['latency_ms'].values
    uptimes = df['is_up'].values

    # Calculate metrics
    avg_latency = float(np.mean(latencies))
    # Note: numpy percentile handles interpolation for smaller datasets
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
        return [{"error": "Telemetry data is not available."}]

    results = []
    threshold = request_data.threshold_ms
    
    # Iterate through the requested regions
    for region_name in request_data.regions:
        # Filter the DataFrame for the current region
        region_df = TELEMETRY_DF[TELEMETRY_DF['region'] == region_name.lower()]

        # Calculate metrics for the filtered data
        metrics = calculate_metrics(region_df, threshold)

        # Append result for the region
        results.append({
            "region": region_name.lower(),
            **metrics
        })

    return results

# Optional: Keep the existing root endpoint for Vercel health checks
@app.get("/")
def read_root():
    return {"message": "eShopCo Latency Metrics Service is running. Use POST /metrics"}
