import importlib
import os
import threading
import time

import cv2


class Helper:
    def __init__(self, args):
        self.MODEL_NAME = args.modeldir
        self.GRAPH_NAME = args.graph
        self.LABELMAP_NAME = args.labels
        self.min_conf_threshold = float(args.threshold)
        resW, resH = args.resolution.split('x')
        self.imW, self.imH = int(resW), int(resH)
        self.use_TPU = args.edgetpu
        CWD_PATH = os.getcwd()
        if self.use_TPU:
            if self.GRAPH_NAME == 'detect.tflite':
                self.GRAPH_NAME = 'edgetpu.tflite'
        self.OBJECT_DETECTION_MODEL_PATH = os.path.join(CWD_PATH, self.MODEL_NAME, self.GRAPH_NAME)
        self.PATH_TO_LABELS = os.path.join(CWD_PATH, self.MODEL_NAME, self.LABELMAP_NAME)

    def get_labels(self):
        with open(self.PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            if labels[0] == '???':
                del (labels[0])
        return labels

    def get_interpreter(self):
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate
        if self.use_TPU:
            interpreter = Interpreter(model_path=self.OBJECT_DETECTION_MODEL_PATH,
                                      experimental_delegates=[load_delegate('libedgetpu.1.dylib')])
        #     ADD CONFIG OPTION FOR MAC VS NVIDIA JETSON
        else:
            interpreter = Interpreter(model_path=self.OBJECT_DETECTION_MODEL_PATH)
        return interpreter

    def draw_object_detection_box(self, frame, box, object_name, score):
        ymin = int(max(1, (box[0] * self.imH)))
        xmin = int(max(1, (box[1] * self.imW)))
        ymax = int(min(self.imH, (box[2] * self.imH)))
        xmax = int(min(self.imW, (box[3] * self.imW)))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        # Draw label
        label = '%s: %d%%' % (object_name, int(score * 100))  # Example: 'person: 72%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
        label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                      (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                      cv2.FILLED)  # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                    2)  # Draw label text
        # TODO - solve conversion problem
