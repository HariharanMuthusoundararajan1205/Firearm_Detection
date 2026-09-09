from ultralytics import YOLO

class HumanModel:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def infer(self,frame):
      
       try:
            results = self.model.track(frame, persist = True, verbose = False)[0]
            persons = []

            if results.boxes.id is None or results.boxes.id is None:
                return persons
            
            track_ids = results.boxes.id.int().cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()
            confs = results.boxes.conf.float().cpu().tolist()
            coords = results.boxes.xyxy.int().cpu().tolist()

            for i in range(len(track_ids)):
                if int(classes[i] == 0):
                    persons.append({
                        "track_id":track_ids[i],
                        "bbox": tuple(map(int,coords[i])),
                        "confidence": float(confs[i])
                    })
            return persons 

       except Exception as e:
                    print(f"[ERROR][HumanModel] {e}")
                    return []

      
