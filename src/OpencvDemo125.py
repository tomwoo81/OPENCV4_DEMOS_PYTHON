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

# OpenCV DNN 基于SSD实现实时视频检测
def OpencvDemo125():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromCaffe(model_txt, model_bin)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # read a video
    capture = cv.VideoCapture("videos/vtest.avi")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    ret, src = capture.read()
    if not ret:
        logging.error("could not read a frame!")
        return cv.Error.StsError
    
    height, width = src.shape[:2]

    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)

    result = None
    winName = "object detection - SSD model"
    cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    
    while True:
        ret, src = capture.read()
        if not ret:
            break

        dst = np.copy(src)

        input = cv.dnn.blobFromImage(src, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False)

        # run the model
        net.setInput(input)
        output = net.forward()

        # get detection results
        score_threshold = 0.4
        for detection in output[0, 0, : , : ]:
            score = float(detection[2])
            if score > score_threshold:
                objIndex = int(detection[1])
                left = detection[3] * width
                top = detection[4] * height
                right = detection[5] * width
                bottom = detection[6] * height

                cv.rectangle(dst, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=1)
                cv.putText(dst, "{} (score: {:.3f})".format(objName[objIndex], score), 
                        (int(left), int(top) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
        text = "time: {:.0f} ms".format(time)
        cv.putText(dst, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        h, w, ch = src.shape
        if result is None:
            result = np.zeros([h, w * 2, ch], dtype=src.dtype)
        result[0 : h, 0 : w, :] = src
        result[0 : h, w : w * 2, :] = dst
        cv.putText(result, "original frame", (10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
        cv.putText(result, "image with results", (w + 10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
        cv.imshow(winName, result)

        c = cv.waitKey(50)

        if c == 27: # ESC
            break
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo125()

# end of file
