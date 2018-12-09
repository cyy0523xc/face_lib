# -*- coding: utf-8 -*-
#
# face detetion
# Author: alex
# Created Time: 2018年12月09日 星期日 11时19分47秒
import cv2
from cv2 import dnn
import dlib


class conf:
    in_width = 300
    in_height = 300
    conf_threshold = 0.5


# support algo
hog_detector = dlib.get_frontal_face_detector()
cnn_detector = None
dnn_detector = None


def set_cnn_model(model_path: str):
    global cnn_detector
    cnn_detector = dlib.cnn_face_detection_model_v1(model_path)


def set_dnn_model(model_path: str, prototxt_path: str):
    global dnn_detector
    dnn_detector = dnn.readNetFromCaffe(prototxt_path, model_path)


def format_rect(rects, shape):
    return rects


def detect(img, model='dnn', number_of_times_to_upsample=1):
    if model == 'hog':
        rects = hog_detector(img, number_of_times_to_upsample)
        return format_rect(rects)
    elif model == 'cnn':
        rects = cnn_detector(img, number_of_times_to_upsample)
        return format_rect(rects)

    # 默认使用dnn
    cols = img.shape[1]
    rows = img.shape[0]
    blob = dnn.blobFromImage(img, 1.0, (conf.in_width, conf.in_height),
                             (104.0, 177.0, 123.0), False, False)
    rects = []
    dnn_detector.setInput(blob)
    detections = dnn_detector.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf.conf_threshold:
            continue
        left = int(detections[0, 0, i, 3] * cols)
        top = int(detections[0, 0, i, 4] * rows)
        right = int(detections[0, 0, i, 5] * cols)
        bottom = int(detections[0, 0, i, 6] * rows)
        rects.append([(left, top), (right, bottom), confidence])

    return rects
