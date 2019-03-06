# -*- coding: utf-8 -*-
#
# 人脸特征点
# Author: alex
# Created Time: 2019年02月27日 星期三 14时56分24秒
import face_lib
import cv2

model = 'cnn'
face_lib.set_threshold(0.4)


def encodings(path):
    """
    鼻尖 30
    鼻根 27
    下巴 8
    左眼外角 36
    左眼内角 39
    右眼外角 45
    右眼内角 42
    嘴中心   66
    嘴左角   48
    嘴右角   54
    左脸最外 0
    右脸最外 16
    """
    image = face_lib.imread(path)
    rects = face_lib.detect(image, model=model)
    shapes = face_lib.landmarks(image, rects[0])
    for i, shape in zip(range(len(shapes)), shapes):
        conf = rects[1][i]
        loc = rects[0][i]
        x, y, xb, yb = loc
        cv2.rectangle(image, (x, y), (xb, yb), (0, 0, 255), thickness=1)
        cv2.putText(image, str(conf)[:6], (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 0), 2)
        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(image, pt_pos, 1, (255, 0, 0), 1)

    return image


if __name__ == '__main__':
    import os
    import sys
    from imutils.paths import list_images
    path = sys.argv[1]
    save_path = sys.argv[2]
    print(model)
    for index, ip in enumerate(list_images(path)):
        print("path: ", ip)
        image = encodings(ip)
        ip = ip.split('/')
        cv2.imwrite(os.path.join(save_path, '%s_%s' % (model, ip[-1])), image)
