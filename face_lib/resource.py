# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2018年12月09日 星期日 20时09分45秒
from pkg_resources import resource_filename


def predictor_68_point_model_location():
    return resource_filename(__name__,
                             "models/shape_predictor_68_face_landmarks.dat")


def predictor_5_point_model_location():
    return resource_filename(__name__,
                             "models/shape_predictor_5_face_landmarks.dat")


def face_recognition_model_location():
    return resource_filename(__name__,
                             "models/dlib_face_recognition_resnet_model_v1.dat")


def cnn_face_detector_model_location():
    return resource_filename(__name__,
                             "models/mmod_human_face_detector.dat")


def dnn_prototxt_location():
    return resource_filename(__name__, "models/deploy.prototxt")


def dnn_caffemodel_location():
    return resource_filename(__name__,
                             "models/res10_300x300_ssd_iter_140000.caffemodel")


def haarcascade_frontalface_location():
    return resource_filename(__name__,
                             "models/haarcascade_frontalface_default.xml")
