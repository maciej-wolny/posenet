from threading import Thread

import cv2


def gstreamer_pipeline(
        capture_width=3280,
        capture_height=2464,
        display_width=820,
        display_height=616,
        framerate=21,
        flip_method=2,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


class VideoStream:
    def __init__(self):
        # self.stream = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        self.stream = cv2.VideoCapture(0)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def get_dimentions(self):
        frame_width = int(self.stream.get(3))
        frame_height = int(self.stream.get(4))
        return frame_width, frame_height
