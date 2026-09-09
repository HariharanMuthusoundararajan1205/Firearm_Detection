import psutil
import time
class Pipeline:
    def __init__(
        self,
        human_detection,
        firearm_detection,
        logger,
        association,
        thresholds,
        evidence_saver,
        api_publisher=None
    ):
        self.human_detection = human_detection
        self.firearm_detection = firearm_detection
        self.logger = logger
        self.association = association
        self.thresholds = thresholds
        self.evidence_saver = evidence_saver
        self.api_publisher = api_publisher



    def process(self, frame, camera_id):
        try:
            start_time = time.time()

            persons = self.human_detection.infer(frame)
            persons = [p for p in persons if p["confidence"] > self.thresholds["human_threshold"]]

            firearms = self.firearm_detection.infer(frame)
            firearms = [f for f in firearms if f["confidence"] > self.thresholds["firearm_threshold"]]

            if firearms:
                print("[Firearm Detected]")

            associations = self.association.associate(firearms, persons)

            self.logger.log_humans(camera_id, persons)
            self.logger.log_firearms(camera_id, firearms)
            self.logger.log_associations(camera_id, associations)
            self.evidence_saver.save(frame, camera_id, firearms, associations)

            latency = time.time() - start_time
            cpu = psutil.cpu_percent()
            fps = 1 / latency if latency > 0 else 0

            if self.api_publisher:
             self.api_publisher.publish(
                camera_id=camera_id,
                firearms=firearms,
                associations=associations,
                cpu=cpu,
                latency=latency,
                fps=fps,
             )


            return {
                "firearms": firearms,
                "associations": associations,
                "cpu": cpu,
                "latency": latency,
                "fps": fps
            }

        except Exception as e:
            print(f"[ERROR][Pipeline] {e}")
            return {
                "firearms": [],
                "associations": [],
                "cpu": 0.0,
                "latency": 0.0,
                "fps": 0.0
            }
