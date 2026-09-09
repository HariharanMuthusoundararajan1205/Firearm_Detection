from ultralytics import YOLO

class FirearmModel:

    def __init__(self):
        self.model = YOLO(r"C:\Users\sidhu\OneDrive\Desktop\Paarvai_Test\gun_datasets\Test_Environment\AI Models\Firearm_30_07_2026_11L.pt")

    def infer(self,frame):
        try:
            results = self.model(frame,verbose =  False)[0]

            firearms =[]

            for box in results.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf[0])

                firearms.append({
                    "bbox": (x1,y1,x2,y2),
                    "confidence": conf
                })
                
            return firearms
        except Exception as e:
            print(f"[Error][Firearm Model] {e}")
            return []
    