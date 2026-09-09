import json
import queue
import threading
import time
from datetime import datetime
from urllib import error, request


class ApiPublisher:
    def __init__(self, config):
        api_config = config.get("api", {})
        self.enabled = api_config.get("enabled", False)
        self.endpoint = api_config.get("endpoint")
        self.timeout = api_config.get("timeout_seconds", 5)
        self.include_empty = api_config.get("send_empty_events", False)
        self.queue = queue.Queue(maxsize=api_config.get("queue_size", 100))
        self.stop_event = threading.Event()
        self.worker = None

        if self.enabled and self.endpoint:
            self.worker = threading.Thread(target=self._worker, daemon=True)
            self.worker.start()

    def publish(self, camera_id, firearms, associations, cpu, latency, fps):
        if not self.enabled or not self.endpoint:
            return

        if not self.include_empty and not firearms and not associations:
            return

        payload = {
        "camera_id": camera_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "firearms": firearms,
        "associations": associations,
        "cpu": cpu,
        "latency": latency,
        "fps": fps,
        }


        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(payload)

    def _worker(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                payload = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                body = json.dumps(payload).encode("utf-8")
                req = request.Request(
                    self.endpoint,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout) as response:
                    if response.status >= 400:
                        print(f"[API] Failed to push event: HTTP {response.status}")
            except error.URLError as exc:
                print(f"[API] Failed to push event: {exc}")
            except Exception as exc:
                print(f"[API] Unexpected publisher error: {exc}")
            finally:
                self.queue.task_done()

    def close(self):
        if not self.worker:
            return

        self.stop_event.set()
        self.worker.join(timeout=max(1, self.timeout + 1))
