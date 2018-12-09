# -*- coding: utf-8 -*-
#
# face detetion
# Author: alex
# Created Time: 2018年12月09日 星期日 11时19分47秒
import numpy as np
from cv2 import dnn
import dlib
from .resource import predictor_5_point_model_location, \
    predictor_68_point_model_location, \
    cnn_face_detector_model_location, \
    face_recognition_model_location, \
    dnn_prototxt_location, dnn_caffemodel_location


class conf:
    in_width = 300
    in_height = 300
    conf_threshold = 0.6


# support algo
hog_detector = dlib.get_frontal_face_detector()
cnn_detector = None
dnn_detector = None

#
predictor = dlib.shape_predictor(predictor_68_point_model_location())

face_encoder = dlib.face_recognition_model_v1(face_recognition_model_location())


def set_cnn_model(model_path: str = None):
    global cnn_detector
    if model_path == None:
        model_path = cnn_face_detector_model_location()
    cnn_detector = dlib.cnn_face_detection_model_v1(model_path)


def set_dnn_model(model_path: str = None, prototxt_path: str = None):
    global dnn_detector
    if model_path is None:
        model_path = dnn_caffemodel_location()
        prototxt_path = dnn_prototxt_location()
    dnn_detector = dnn.readNetFromCaffe(prototxt_path, model_path)


def set_predictor(use_small=False):
    global predictor
    if use_small:
        predictor = dlib.shape_predictor(predictor_5_point_model_location())
    else:
        predictor = dlib.shape_predictor(predictor_68_point_model_location())


def detect(img, model='dnn', number_of_times_to_upsample=1):
    if model == 'hog':
        rects = hog_detector(img, number_of_times_to_upsample)
        rects = [[1, (r.left(), r.top()), (r.right(), r.bottom())]
                 for r in rects]
        return rects
    elif model == 'cnn':
        rects = cnn_detector(img, number_of_times_to_upsample)
        rects = [[r.confidence, (r.rect.left(), r.rect.top()),
                  (r.rect.right(), r.rect.bottom())]
                 for r in rects]
        return rects

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
        rects.append([confidence, (left, top), (right, bottom)])

    return rects


def encode(img, rects, num_jitters=1):
    rects = [_css_to_rect(rect) for rect in rects]
    landmarks = [predictor(img, rect) for rect in rects]
    return [np.array(face_encoder.compute_face_descriptor(img, landmark, num_jitters))
            for landmark in landmarks]


def distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def _css_to_rect(css):
    _, (left, top), (right, bottom) = css
    return dlib.rectangle(left, top, right, bottom)
