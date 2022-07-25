#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

bin_model = "models/googlenet/bvlc_googlenet.caffemodel"
protxt = "models/googlenet/bvlc_googlenet.prototxt"
labels_txt = "models/googlenet/classification_classes_ILSVRC2012.txt"

# OpenCV DNN 单张与多张图像的推断
def OpencvDemo132():
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
    
    # read image(s)
    src1 = cv.imread("images/cat.jpg")
    src2 = cv.imread("images/aeroplane.jpg")
    if src1 is None or src2 is None:
        logging.error("could not load image(s)!")
        return cv.Error.StsError
    srcs = []
    srcs.append(src1)
    srcs.append(src2)

    dst1 = np.copy(src1)
    dst2 = np.copy(src2)
    dsts = []
    dsts.append(dst1)
    dsts.append(dst2)

    input = cv.dnn.blobFromImages(srcs, 1.0, (224, 224), (104, 117, 123), False, crop=False)

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    for i in range(len(output)):
        # get the class with the highest score for each image
        classId = np.argmax(output[i])
        confidence = output[i][classId]
        time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
        text = "class: {}, confidence: {:.3f}, time: {:.0f} ms".format(labels[classId], confidence, time)
        logging.info(text)

        # display the result
        cv.putText(dsts[i], text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv.imshow("image classification - GoogLeNet model - {:d}".format(i + 1), dsts[i])
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo132()

# end of file
