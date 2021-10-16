import queue
import time
from threading import Thread

import cv2
import os

import requests

LAST_BIRD_OCCURRENCE_TIMEOUT = 5
RECORDINGS_DIRECTORY = "recordings/"
FRAME_RATE = 30


class BirdDetectionVideoHandler:
    def __init__(self, frame_width, frame_height):
        self.recording = False
        self.last_bird_occurrence = time.time() - LAST_BIRD_OCCURRENCE_TIMEOUT
        self.writer = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.current_file = None
        self.frames_queue = queue.Queue()
        if not os.path.exists(RECORDINGS_DIRECTORY):
            os.makedirs(RECORDINGS_DIRECTORY)

    def new_recording(self, recording_id):
        print("new recording:" + recording_id)

        self.current_file = RECORDINGS_DIRECTORY + recording_id + ".avi"
        self.writer = cv2.VideoWriter(self.current_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                      FRAME_RATE,
                                      (self.frame_width, self.frame_height))
        self.recording = True
        Thread(target=self._write_loop, args=()).start()

    def _write_loop(self):
        while self.recording:
            while not self.frames_queue.empty():
                data = self.frames_queue.get()
                print("pop frame")
                while data['frames'] > 0:
                    self.writer.write(data['frame'])
                    data['frames'] -= 1

    def write_frame(self, frame, current_fps):
        self.frames_queue.put({"frame": frame, "frames": FRAME_RATE / current_fps})

    def bird_detected(self):
        self.last_bird_occurrence = time.time()

    def is_recording(self):
        return self.recording

    def no_recent_occurrences(self):
        return time.time() - self.last_bird_occurrence > LAST_BIRD_OCCURRENCE_TIMEOUT

    def stop_and_publish(self):
        print("sending a video")
        while self.recording:
            time.sleep(0.1)
        requests.post("http://osiris.local:8000/upload",
                      files={'file': open(self.current_file, 'rb')})
        self.writer.release()
        self.recording = False
