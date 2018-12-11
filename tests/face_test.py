# -*- coding: utf-8 -*-
#
# Test
# Author: alex
# Created Time: 2018年12月09日 星期日 20时25分48秒
import os
import cv2
import time
import face_lib


def path_detect(path, model_algo):
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
        locations, confidences = face_lib.detect(image, model=model_algo)
        if len(locations) > 0:
            encodings = face_lib.encode(image, locations)
            files.append((fn, locations, encodings,
                          max(confidences), confidences))
            print(fn, locations, confidences)

    print('===> ', time.time()-start, '  image count: ', count)

    # face recognition
    new_confidence = 0.7 if model_algo == 'dnn' else 1
    print('face recognition...')
    files = sorted(files, key=lambda x: x[3], reverse=True)
    faces = []
    faces_files = []
    for fn, locations, encodings, _, confidences in files:
        if len(faces) == 0:
            faces = encodings
            annotate = ['Face'+chr(ord('A')+i) for i in range(len(faces))]
            faces_files = [(fn, locations, annotate)]
            continue

        has_new_face = False
        annotate = []
        for encoding, confidence in zip(encodings, confidences):
            if confidence < new_confidence:
                continue
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
    print('total face image count:', len(faces_files))
    print('new face count: ', len(faces))
    for fn, locations, annotate in faces_files:
        print(fn, locations)
        image = cv2.imread(os.path.join(path, fn))
        show_image(image, locations, annotate)


def show_image(image, locations, annotate, wait=0):
    for ((left, top), (right, bottom)), anno in zip(locations, annotate):
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
    path_detect(sys.argv[2], sys.argv[1])
