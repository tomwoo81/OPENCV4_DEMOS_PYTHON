#!/usr/bin/env python3
#coding = utf-8

from OpencvDemo123 import OpencvDemo123
from OpencvDemo124 import OpencvDemo124
from OpencvDemo130 import OpencvDemo130
from OpencvDemo134 import OpencvDemo134
from OpencvDemo135 import OpencvDemo135

# OpenCV DNN 解析网络输出结果
def OpencvDemo136():
    # image classification network
    ret1 = OpencvDemo123()

    # object detection network (SSD/RCNN/Faster-RCNN)
    ret2 = OpencvDemo124()

    # object detection network (YOLO)
    ret3 = OpencvDemo130()

    # image segmentation
    ret4 = OpencvDemo134()

    # image generation
    ret5 = OpencvDemo135()

    return (ret1 or ret2 or ret3 or ret4 or ret5)

if __name__ == "__main__":
    OpencvDemo136()

# end of file
