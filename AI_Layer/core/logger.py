import csv
import os
from datetime import datetime
import threading

class CSVLogger:
    def __init__(self, base_path="logs"):
        os.makedirs(base_path, exist_ok=True)

        self.lock = threading.Lock()

        self.human_file = os.path.join(base_path, "humans.csv")
        self.firearm_file = os.path.join(base_path, "firearms.csv")
        self.association_file = os.path.join(base_path, "associations.csv")

        self._init_file(self.human_file, ["camera_id", "timestamp", "track_id", "bbox", "confidence"])
        self._init_file(self.firearm_file, ["camera_id", "timestamp", "bbox", "confidence"])
        self._init_file(self.association_file, ["camera_id", "timestamp", "bbox", "confidence", "Person_id"])

    def _init_file(self, file_path, headers):
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_humans(self, camera_id, detections):
        self._write(self.human_file, camera_id, detections, "human")

    def log_firearms(self, camera_id, detections):
        self._write(self.firearm_file, camera_id, detections, "firearm")

    def log_associations(self, camera_id, detections):
        self._write(self.association_file, camera_id, detections, "association")

    def _write(self, file_path, camera_id, detections, dtype):
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.lock:
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)

                for d in detections:
                    if dtype == "human":
                        writer.writerow([
                            camera_id,
                            timestamp,
                            d.get("track_id"),
                            d.get("bbox"),
                            d.get("confidence")
                        ])

                    elif dtype == "firearm":
                        writer.writerow([
                            camera_id,
                            timestamp,
                            d.get("bbox"),
                            d.get("confidence")
                        ])

                    elif dtype == "association":
                        writer.writerow([
                            camera_id,
                            timestamp,
                            d.get("bbox"),
                            d.get("confidence"),
                            d.get("Person_id")
                        ])