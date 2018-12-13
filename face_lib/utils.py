# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2018年12月12日 星期三 15时11分01秒
import cv2


def imread(path):
    """读取image对象"""
    return cv2.imread(path)


def box_area(box):
    """the area of a box
    Args:
        box: (left, top, right, bottom)
    """
    left, top, right, bottom = box
    return (right-left+1) * (bottom-top+1)


def IoU(box1, box2):
    """cal iou
    Args:
        box1: (left, top, right, bottom)
        box2: (left, top, right, bottom)
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(right1, right2)
    bottom = min(bottom1, bottom2)
    if right-left <= 0 or bottom-top <= 0:
        return 0

    inter_area = box_area((left, top, right, bottom))
    area1 = box_area(box1)
    area2 = box_area(box2)
    return inter_area / float(area1 + area2 - inter_area)


if __name__ == '__main__':
    box1 = (1, 5, 3, 7)
    box2 = (2, 4, 5, 9)
    print(IoU(box1, box2))

    box1 = (1, 5, 3, 6)
    box2 = (3, 4, 5, 9)
    print(IoU(box1, box2))
