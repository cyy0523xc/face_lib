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
    image = face_lib.imread(path)
    rects = face_lib.detect(image, model=model)
    shapes = face_lib.landmarks(image, rects[0])
    for i, shape in zip(range(len(shapes)), shapes):
        print("%s %d: %d" % (path, i, shape.num_parts))
        for index, pt in enumerate(shape.parts()):
            # print('Part {}: {}'.format(index, pt))
            pt_pos = (pt.x, pt.y)
            cv2.circle(image, pt_pos, 1, (255, 0, 0), 1)

    cv2.namedWindow(path, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(path, image)
    cv2.waitKey(0)


if __name__ == '__main__':
    import sys
    from imutils.paths import list_images
    path = sys.argv[1]
    print(model)
    for ip in list_images(path):
        encodings(ip)
