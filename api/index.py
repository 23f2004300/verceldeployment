from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def telemetry(data: dict):
    regions = data.get("regions", [])
    threshold_ms = data.get("threshold_ms", 180)

    sample_data = {
        "amer": [{"latency": 150, "uptime": 0.99}, {"latency": 170, "uptime": 1.0}, {"latency": 200, "uptime": 0.95}],
        "emea": [{"latency": 120, "uptime": 0.98}, {"latency": 160, "uptime": 0.97}, {"latency": 190, "uptime": 0.96}],
    }

    response = {}
    for region in regions:
        records = sample_data.get(region, [])
        if not records:
            continue

        latencies = [r["latency"] for r in records]
        uptimes = [r["uptime"] for r in records]

        response[region] = {
            "avg_latency": float(np.mean(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "avg_uptime": float(np.mean(uptimes)),
            "breaches": sum(l > threshold_ms for l in latencies),
        }

    return response
