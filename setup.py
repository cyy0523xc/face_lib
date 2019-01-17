# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2018年12月09日 星期日 20时02分24秒
from setuptools import setup

# 注意：dlib and opencv 可能会分为cpu和gpu版，避免在这里被重装
requirements = [
    'numpy',
    # 'dlib',
    # 'opencv-python',
]

test_requirements = [
]

setup(
    name='face_lib',
    version='0.1.4',
    description="Simple to use face detection and recognition.",
    long_description="Simple to use face detection and recognition.",
    author="Alex Cai",
    author_email='cyy0523xc@gmail.com',
    url='https://github.com/cyy0523xc/face_lib',
    packages=[
        'face_lib',
    ],
    package_dir={'face_lib': 'face_lib'},
    package_data={
        'face_lib': ['models/*']
    },
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='face detection and face recognition',
    test_suite='tests',
    tests_require=test_requirements
)
