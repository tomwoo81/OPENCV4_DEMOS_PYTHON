#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

bin_model = "models/googlenet/bvlc_googlenet.caffemodel"
protxt = "models/googlenet/bvlc_googlenet.prototxt"
labels_txt = "models/googlenet/classification_classes_ILSVRC2012.txt"

# OpenCV DNN 为模型运行设置目标设备与计算后台
def OpencvDemo123():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromCaffe(protxt, bin_model)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # load labels
    labels = None
    with open(labels_txt, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    if labels is None or 0 == len(labels):
        logging.error("could not load labels!")
        return cv.Error.StsError
    
    # read an image
    src = cv.imread("images/space_shuttle.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst = np.copy(src)

    input = cv.dnn.blobFromImage(src, 1.0, (224, 224), (104, 117, 123), True, crop=False)

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    # get the class with the highest score
    output = output.flatten()
    classId = np.argmax(output)
    confidence = output[classId]
    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "class: {}, confidence: {:.3f}, time: {:.0f} ms".format(labels[classId], confidence, time)
    logging.info(text)

    # display the result
    cv.putText(dst, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv.imshow("image classification - GoogLeNet model", dst)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo123()

# end of file
