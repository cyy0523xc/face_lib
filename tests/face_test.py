# -*- coding: utf-8 -*-
#
# Test
# Author: alex
# Created Time: 2018年12月09日 星期日 20时25分48秒
import os
import cv2
import time
import face_lib


# init
face_lib.set_cnn_model()
face_lib.set_dnn_model()


def path_detect(path):
    filenames = sorted(os.listdir(path))
    files = []
    # face detection
    start = time.time()
    count = 0
    print('face detection...')
    for fn in filenames:
        if fn.endswith(('jpg', 'jpeg', 'png')) is False:
            continue
        count += 1
        image = cv2.imread(os.path.join(path, fn))
        locations = face_lib.detect(image, model='cnn')
        if len(locations) > 0:
            confidences = [i[0] for i in locations]
            encodings = face_lib.encode(image, locations)
            files.append((fn, locations, encodings, max(confidences)))
            print(fn, encodings)

    print('===> ', time.time()-start, '  image count: ', count)

    # face recognition
    print('face recognition...')
    files = sorted(files, key=lambda x: x[3], reverse=True)
    faces = []
    faces_files = []
    for fn, locations, encodings, _ in files:
        if len(faces) == 0:
            faces = encodings
            annotate = ['Face'+chr(ord('A')+i) for i in range(len(faces))]
            faces_files = [(fn, locations, annotate)]
            continue

        has_new_face = False
        annotate = []
        for encoding in encodings:
            distances = face_lib.distance(faces, encoding)
            distance = min(distances)
            face_index = distances.tolist().index(distance)
            if distance < 0.55:   # 不是新face
                annotate.append('Person'+chr(ord('A')+face_index))
                continue
            print('new face in ', fn, '   distance: ', distance)
            has_new_face = True
            annotate.append('Face'+chr(ord('A')+len(faces)))
            faces.append(encoding)

        if has_new_face:
            faces_files.append((fn, locations, annotate))

    # show images
    print('show images...')
    print('new face count: ', len(faces))
    for fn, locations, annotate in faces_files:
        print(fn, locations)
        image = cv2.imread(os.path.join(path, fn))
        show_image(image, locations, annotate)


def show_image(image, locations, annotate, wait=0):
    for (_, (left, top), (right, bottom)), anno in zip(locations, annotate):
        cv2.rectangle(image, (left, top),
                      (right, bottom), (0, 255, 0))
        labelSize, baseLine = cv2.getTextSize(anno, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, 1)

        cv2.rectangle(image, (left, top - labelSize[1]),
                      (left + labelSize[0], top + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(image, anno, (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("detect", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    import sys
    path_detect(sys.argv[1])
