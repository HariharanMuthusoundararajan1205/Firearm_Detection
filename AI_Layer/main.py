import yaml
import threading
import queue
import time
import cv2
import os

from models.human_detection import HumanModel
from models.firearm_detection import FirearmModel
from core.pipeline import Pipeline
from core.logger import CSVLogger
from core.association import Grid
from services.camera_stream import CameraStream
from core.evidence import EvidenceSaver
from core.video_annotator import VideoAnnotator
from core.api_publisher import ApiPublisher


def capture_worker(stream, frame_queue, fps, loop, camera_id):
    interval = 1 / fps

    while True:
        ret, frame = stream.read()

        if not ret:
            print(f"[INFO] End of video: {camera_id}")

            if loop:
                print(f"[INFO] Restarting video: {camera_id}")
                stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                frame_queue.put(None)   # stop signal
                break

        if frame_queue.full():
            frame_queue.get_nowait()

        frame_queue.put(frame)

        time.sleep(interval)


def inference_worker(frame_queue, pipeline, camera_id,annotator):
    while True:
        frame = frame_queue.get()
        if frame is None:
            print(f"[INFO] Stopping Infernce since video ends")
            annotator.release()
            break
        try: 
            result = pipeline.process(frame, camera_id)
            firearms = result["firearms"]
            associations = result["associations"]
            cpu = result["cpu"]
            latency = result["latency"]
            fps = result["fps"]

            annotated = annotator.draw(frame,associations)
            annotator.write(annotated)
        except Exception as e:
            print(f"[ERROR] [Inference Worker] {e}")

def run_camera(camera_config, config, api_publisher):

    camera_id = camera_config["id"]
    source = camera_config["source"]
    mode = config["system"]["mode"]
    loop = config["system"].get("loop", False)
    fps = config["system"]["fps"]
    print(f"{camera_id} → mode: {mode}, loop: {loop}")
    stream = CameraStream(source, mode=mode)
    human_model = HumanModel()
    firearm_model = FirearmModel()
    logger = CSVLogger()
    evidence_saver = EvidenceSaver()
    association = Grid(grid_size=config["grid"]["size"])

    source_fps = stream.cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = config["system"].get("output_fps", fps)

    annotator = VideoAnnotator(
        camera_id,
        output_fps=source_fps,
        processing_fps=fps,
    )
    pipeline = Pipeline(
        human_model,
        firearm_model,
        logger,
        association,
        config["thresholds"],
        evidence_saver,
        api_publisher
)

    frame_queue = queue.Queue(maxsize=config["system"]["queue_size"])
    t1 = threading.Thread(
        target=capture_worker,
        args=(stream, frame_queue, fps, loop, camera_id)
    )

    t2 = threading.Thread(
        target=inference_worker,
        args=(frame_queue, pipeline, camera_id,annotator)
    )

    t1.start()
    t2.start()

    t1.join()
    t2.join()
    annotator.release()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir,"config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    threads = []
    api_publisher = ApiPublisher(config)

    for cam in config["cameras"]:
        t = threading.Thread(target=run_camera, args=(cam, config, api_publisher))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    api_publisher.close()
