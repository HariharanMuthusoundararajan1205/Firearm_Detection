import cv2
import os

class VideoAnnotator:
    def __init__(self, camera_id, output_dir="outputs", output_fps=30, processing_fps=3):
        self.camera_id = camera_id
        self.output_fps = output_fps
        self.repeat_count = 1
        self.writer = None

        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, f"{camera_id}_annotated.mp4")

    def draw(self, frame, associations):
        annotated = frame.copy()

        for a in associations:
            person_id = a.get("Person_id")
            person_bbox = a.get("person_bbox")

            if person_id is not None and person_bbox is not None:
                x1, y1, x2, y2 = map(int, person_bbox)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"Person {person_id} WITH GUN",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        return annotated

    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]

            self.writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.output_fps,
                (w, h)
            )

            if not self.writer.isOpened():
                print(f"[ERROR] Failed to open VideoWriter for {self.output_path}")
                self.writer = None
                return

            print(f"[INFO] VideoWriter initialized: {w}x{h} @ {self.output_fps}fps")

        for _ in range(self.repeat_count):
            self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"[INFO] Video saved: {self.output_path}")
