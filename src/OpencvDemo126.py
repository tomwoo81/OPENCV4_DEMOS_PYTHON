#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

model_bin = "models/face_detector/opencv_face_detector_uint8.pb"
model_txt = "models/face_detector/opencv_face_detector.pbtxt"

# OpenCV DNN 基于残差网络的人脸检测
def OpencvDemo126():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromTensorflow(model_bin, model_txt)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # read an image
    src = cv.imread("images/persons.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    height, width = src.shape[:2]
    
    dst = np.copy(src)

    input = cv.dnn.blobFromImage(src, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

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
            left = detection[3] * width
            top = detection[4] * height
            right = detection[5] * width
            bottom = detection[6] * height

            cv.rectangle(dst, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 1)
            cv.putText(dst, "{:.3f}".format(score), 
                       (int(left), int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    cv.putText(dst, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("face detection - residual SSD model", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo126()

# end of file
