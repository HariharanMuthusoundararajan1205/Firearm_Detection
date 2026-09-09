# ---------------------------------------------------------
import cv2
import os
import time
from datetime import datetime

class EvidenceSaver:
    def __init__(self, base_path="evidence", cooldown=5):
        self.base_path = base_path
        self.cooldown = cooldown

        os.makedirs(base_path, exist_ok=True)

        self.last_saved_time = {}
        self.saved_ids = {}

    def save(self, frame, camera_id, firearms, associations):
        saved_items = []

        now = time.time()

      
        if camera_id in self.last_saved_time:
            if now - self.last_saved_time[camera_id] < self.cooldown:
                return saved_items

        cam_folder = os.path.join(self.base_path, camera_id)
        os.makedirs(cam_folder, exist_ok=True)

        if camera_id not in self.saved_ids:
            self.saved_ids[camera_id] = set()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        h, w = frame.shape[:2]

       
        for a in associations:
            person_id = a.get("Person_id")

            if person_id is not None:

              
                if person_id in self.saved_ids[camera_id]:
                    return saved_items

                x1, y1, x2, y2 = map(int, a["person_bbox"])

               
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    return saved_items

                filename = f"person_{person_id}_{timestamp}.jpg"
                path = os.path.join(cam_folder, filename)

                cv2.imwrite(path, crop)

                print(f"[EVIDENCE] {camera_id} Person {person_id} saved")
                saved_items.append({
                    "type": "person_with_firearm",
                    "person_id": person_id,
                    "path": path
                })

                self.saved_ids[camera_id].add(person_id)
                self.last_saved_time[camera_id] = now

                return saved_items

       
        if firearms:
            x1, y1, x2, y2 = map(int, firearms[0]["bbox"])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return saved_items

            filename = f"gun_{timestamp}.jpg"
            path = os.path.join(cam_folder, filename)

            cv2.imwrite(path, crop)

            print(f"[EVIDENCE] {camera_id} Gun saved")
            saved_items.append({
                "type": "firearm",
                "path": path
            })

            self.last_saved_time[camera_id] = now

        return saved_items
