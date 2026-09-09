# from fastapi import FastAPI, Query
# from pydantic import BaseModel, Field
# import cv2
# import time
# import psutil
# import json
# import os
# import sys
# from datetime import datetime
# from pathlib import Path
# import yaml

# BASE_DIR = Path(__file__).resolve().parent
# AI_LAYER_DIR = BASE_DIR 

# if str(AI_LAYER_DIR) not in sys.path:
#     sys.path.insert(0, str(AI_LAYER_DIR))

# from core.pipeline_service import PipelineService

# app = FastAPI()

# #  Load config
# with open(AI_LAYER_DIR / "config.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)

# #  Initialize ONCE
# pipeline_service = PipelineService(config)


# class DetectionEvent(BaseModel):
#     camera_id: str
#     timestamp: str
#     # persons: list[dict] = Field(default_factory=list)
#     firearms: list[dict] = Field(default_factory=list)
#     associations: list[dict] = Field(default_factory=list)
#     # evidence: list[dict] = Field(default_factory=list)
#     cpu: float | None = None
#     latency: float | None = None
#     fps: float | None = None


# # Rule engine
# def generate_alert(class_name, confidence):
#     if class_name == "firearm" and confidence > 0.6:
#         return "HIGH_ALERT"
#     elif class_name == "firearm":
#         return "MEDIUM_ALERT"
#     return "NO_ALERT"


# def get_system_metrics(start_time):
#     latency = time.time() - start_time
#     cpu = psutil.cpu_percent()
#     memory = psutil.virtual_memory().percent
#     return latency, cpu, memory


# @app.post("/predict")
# async def predict():
#     start_time = time.time()

#     frame = cv2.imread("test.jpg")  # demo input

#     if frame is None:
#         return {"error": "Image not found"}

#     #  USE EXISTING PIPELINE
#     persons, firearms, associations = pipeline_service.run(frame, "API")

#     output = []

#     # persons
#     # for p in persons:
#     #     output.append({
#     #         "class": "person",
#     #         "bb": p["bbox"],
#     #         "accuracy": p["confidence"],
#     #         "alert": "NO_ALERT"
#     #     })

#     # firearms
#     for f in firearms:
#         output.append({
#             "class": "firearm",
#             "bb": f["bbox"],
#             "accuracy": f["confidence"],
#             "alert": generate_alert("firearm", f["confidence"])
#         })

#     #  association → key demo feature
#     for a in associations:
#         if a["Person_id"] is not None:
#             output.append({
#                 "class": "person_with_firearm",
#                 "bb": a["person_bbox"],
#                 "accuracy": 1.0,
#                 "alert": "CRITICAL_ALERT"
#             })

#     latency, cpu, memory = get_system_metrics(start_time)

#     return {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "detections": output,
#         "performance": {
#             "cpu": cpu,
#             "memory": memory,
#             "latency": latency,
#             "fps": 1 / latency if latency > 0 else 0
#         }
#     }


# @app.post("/ingest-detections")
# async def ingest_detections(event: DetectionEvent):
#     os.makedirs("received_events", exist_ok=True)
#     log_path = os.path.join("received_events", "detections.jsonl")

#     with open(log_path, "a", encoding="utf-8") as log_file:
#         log_file.write(json.dumps(event.model_dump()) + "\n")
    
#     return {
#     "status": "accepted",
#     "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#     "camera_id": event.camera_id,
#     "firearms_count": len(event.firearms),
#     "associations_count": len(event.associations),
#     "cpu": event.cpu,
#     "latency": event.latency,
#     "fps": event.fps,
# }


# @app.get("/ingest-detections")
# async def get_ingested_detections(limit: int = Query(50, ge=1, le=1000)):
#     log_path = os.path.join("received_events", "detections.jsonl")

#     if not os.path.exists(log_path):
#         return {
#             "status": "no_data",
#             "message": "No detections have been received yet.",
#             "detections": []
#         }

#     detections = []
#     with open(log_path, "r", encoding="utf-8") as log_file:
#         for line in log_file:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 detections.append(json.loads(line))
#             except json.JSONDecodeError:
#                 continue

#     return {
#         "status": "success",
#         "count": len(detections[-limit:]),
#         "detections": detections[-limit:]
#     }











from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import cv2
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent
AI_LAYER_DIR = BASE_DIR

if str(AI_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(AI_LAYER_DIR))

from core.pipeline_service import PipelineService

app = FastAPI()

with open(AI_LAYER_DIR / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

pipeline_service = PipelineService(config)


class DetectionEvent(BaseModel):
    camera_id: str
    timestamp: str
    firearms: list[dict] = Field(default_factory=list)
    associations: list[dict] = Field(default_factory=list)
    cpu: float | None = None
    latency: float | None = None
    fps: float | None = None


def generate_alert(class_name, confidence):
    if class_name == "firearm" and confidence > 0.6:
        return "HIGH_ALERT"
    elif class_name == "firearm":
        return "MEDIUM_ALERT"
    return "NO_ALERT"


@app.post("/predict")
async def predict():
    frame = cv2.imread("test.jpg")

    if frame is None:
        return {"error": "Image not found"}

    result = pipeline_service.run(frame, "API")

    firearms = result["firearms"]
    associations = result["associations"]
    cpu = result["cpu"]
    latency = result["latency"]
    fps = result["fps"]

    output = []

    for f in firearms:
        output.append({
            "class": "firearm",
            "bb": f["bbox"],
            "accuracy": f["confidence"],
            "alert": generate_alert("firearm", f["confidence"])
        })

    for a in associations:
        if a["Person_id"] is not None and a["person_bbox"] is not None:
            output.append({
                "class": "person_with_firearm",
                "bb": a["person_bbox"],
                "accuracy": 1.0,
                "alert": "CRITICAL_ALERT"
            })

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections": output,
        "performance": {
            "cpu": cpu,
            "latency": latency,
            "fps": fps
        }
    }


@app.post("/ingest-detections")
async def ingest_detections(event: DetectionEvent):
    os.makedirs("received_events", exist_ok=True)
    log_path = os.path.join("received_events", "detections.jsonl")

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event.model_dump()) + "\n")

    return {
        "status": "accepted",
        "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": event.camera_id,
        "firearms_count": len(event.firearms),
        "associations_count": len(event.associations),
        "cpu": event.cpu,
        "latency": event.latency,
        "fps": event.fps,
    }


@app.get("/ingest-detections")
async def get_ingested_detections(limit: int = Query(50, ge=1, le=1000)):
    log_path = os.path.join("received_events", "detections.jsonl")

    if not os.path.exists(log_path):
        return {
            "status": "no_data",
            "message": "No detections have been received yet.",
            "detections": []
        }

    detections = []
    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)

                filtered_record = {
                    "camera_id": record.get("camera_id"),
                    "timestamp": record.get("timestamp"),
                    "firearms": record.get("firearms", []),
                    "associations": record.get("associations", []),
                    "cpu": record.get("cpu"),
                    "latency": record.get("latency"),
                    "fps": record.get("fps"),
                }

                detections.append(filtered_record)
            except json.JSONDecodeError:
                continue

    return {
        "status": "success",
        "count": len(detections[-limit:]),
        "detections": detections[-limit:]
    }


@app.get("/latest-detection")
async def get_latest_detection():
    log_path = os.path.join("received_events", "detections.jsonl")

    if not os.path.exists(log_path):
        return {
            "status": "no_data",
            "message": "No detections have been received yet."
        }

    latest = None
    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                latest = {
                    "camera_id": record.get("camera_id"),
                    "timestamp": record.get("timestamp"),
                    "firearms": record.get("firearms", []),
                    "associations": record.get("associations", []),
                    "cpu": record.get("cpu"),
                    "latency": record.get("latency"),
                    "fps": record.get("fps"),
                }
            except json.JSONDecodeError:
                continue

    if latest is None:
        return {
            "status": "no_data",
            "message": "Detection log is empty or invalid."
        }

    return {
        "status": "success",
        "detection": latest
    }
