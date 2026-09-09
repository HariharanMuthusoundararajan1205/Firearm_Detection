import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport:tcp"

class CameraStream:
    def __init__(self, source, mode="rtsp",loop = False):
        self.source = source
        self.loop = loop
        self.mode = mode
        self.cap = None
        self.connect()

    def connect(self):
        if self.mode == "rtsp":
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")

    def read(self):
        ret, frame = self.cap.read()

        if not ret:
            return False, None

        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()