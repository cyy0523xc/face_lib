# face_lib
face detection, face recognition and so on. 人脸检测，人脸识别等

see: https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/

## Support

- hog
- cnn
- haar
- dnn

## Install

```sh
# install from source
sudo -H python3 setup.py install

# install from git
pip3 install git+https://github.com/cyy0523xc/face_lib.git
```

## Usages

```python
import face_lib

# read image from file
image = face_lib.imread('/path/to/image.jpg')

# face detection
rects = face_lib.detect(image, model='dnn')

# face encoding
encodings = face_lib.encode(image, rects)

# face distinces between encodings and encoding
# get the distinces from a face to other faces
distinces = face_lib.distince(encodings, encoding)
```



