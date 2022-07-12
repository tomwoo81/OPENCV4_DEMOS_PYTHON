#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

model_bin = "models/ssd/MobileNetSSD_deploy.caffemodel"
model_txt = "models/ssd/MobileNetSSD_deploy.prototxt"

objName = ["background", 
"aeroplane", "bicycle", "bird", "boat", 
"bottle", "bus", "car", "cat", "chair", 
"cow", "diningtable", "dog", "horse", 
"motorbike", "person", "pottedplant", 
"sheep", "sofa", "train", "tvmonitor"]

# OpenCV DNN 基于SSD实现对象检测
def OpencvDemo124():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromCaffe(model_txt, model_bin)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # get layer names and layer IDs
    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)
    logging.info("type of the last layer: {}".format(lastLayer.type))
    
    # read an image
    src = cv.imread("images/dog.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    height, width = src.shape[:2]
    
    dst = np.copy(src)

    input = cv.dnn.blobFromImage(src, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False)
    logging.info("blob - width: {:d}, height: {:d}".format(input.shape[3], input.shape[2]))

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    # get detection results
    score_threshold = 0.5
    for detection in output[0, 0, : , : ]:
        score = float(detection[2])
        if score > score_threshold:
            objIndex = int(detection[1])
            left = detection[3] * width
            top = detection[4] * height
            right = detection[5] * width
            bottom = detection[6] * height

            cv.rectangle(dst, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
            cv.putText(dst, "{} (score: {:.3f})".format(objName[objIndex], score), 
                       (int(left), int(top) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    logging.info(text)
    cv.putText(dst, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("object detection - SSD model", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo124()

# end of file
