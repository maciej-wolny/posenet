from datetime import datetime
import os
import threading

import cv2
import numpy as np
import importlib.util

import requests
import simplejpeg
from tflite_runtime.interpreter import load_delegate, Interpreter

from helper import Helper
from batch_imgae_processor import ImageBatchProcessor
from video_recorder import BirdDetectionVideoHandler
from video_stream import VideoStream
import imagezmq
import argparse
import socket
import time

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='posenet.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution',
                    help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, '
                         'errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--nano', help='This flag indicates that the script is running on Nvidia Jetson Nano',
                    action='store_true')
parser.add_argument('--demo', help='Displays a window with an output of a model',
                    action='store_true')
parser.add_argument('--serverip', help='image zmq broker address',
                    default='localhost')

args = parser.parse_args()

min_conf_threshold = float(args.threshold)

helper = Helper(args)
# labels = helper.get_labels()
interpreter = helper.get_interpreter()
edgetpu_delegate = load_delegate('libedgetpu.1.dylib')
posenet_decoder_delegate = load_delegate(os.path.join(
    'posenet_lib', os.uname().machine, 'posenet_decoder.so'))
interpreter = Interpreter(
    "Sample_TFLite_model/posenet.tflite", experimental_delegates=[edgetpu_delegate, posenet_decoder_delegate])
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream and batch processing
video_stream = VideoStream().start()
time.sleep(1)
# image_batch_processor = ImageBatchProcessor()
sender = imagezmq.ImageSender(connect_to="tcp://osiris.local:5555")
rpiName = socket.gethostname()
frame_width, frame_height = video_stream.get_dimentions()

bird_recording_handler = BirdDetectionVideoHandler(frame_width, frame_height)
frames_counter = 0

bird_discovered_event = threading.Event()
bird_discovered_flag = False
bird_discovery_timer = 0
while True:
    t1 = cv2.getTickCount()
    frame1 = video_stream.read()
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    objects = []
    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
            # object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            # if object_name == "bird":
            #     if not bird_recording_handler.is_recording():
            #         bird_recording_handler.new_recording(datetime.now().time().strftime("%H-%M-%S.%f"))
            #     bird_recording_handler.bird_detected()

            # helper.draw_object_detection_box(frame, boxes[i], object_name, scores[i])
            # TODO - solve conversion problem
            objects.append({
                "label": object_name,
                "score": int(scores[i] * 100),
                "ymin": int(boxes[i][0] * 1000),
                "ymax": int(boxes[i][2] * 1000),
                "xmin": int(boxes[i][1] * 1000),
                "xmax": int(boxes[i][3] * 1000),
            })
    # id = generateID()
    # if len(objects > 0):
    #     image_batch_processor.insert_data({
    #         "timestamp": 0,
    #         "sensorId": 1,
    #         "id": 0,
    #         "version": 1,
    #         "objects": objects
    #     })
    # Draw framerate in corner of frame

    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)
    # cv2.imshow('Object detector', frame)
    # TODO - IF args.preview: cv2.imshow('Object detector', frame)
    # TODO - IF live stream requested: sender.send_image(rpiName, frame)
    if bird_recording_handler.is_recording():
        bird_recording_handler.write_frame(frame, frame_rate_calc)
        if bird_recording_handler.no_recent_occurrences():
            bird_recording_handler.stop_and_publish()

    # jpg_buffer= simplejpeg.encode_jpeg(frame, quality=95,
    #                                         colorspace='BGR')
    # reply_from_mac = sender.send_jpg("fdafs", jpg_buffer)
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    frames_counter += 1
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
video_stream.stop()
