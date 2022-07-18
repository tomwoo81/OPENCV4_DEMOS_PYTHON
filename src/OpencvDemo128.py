#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

model_bin = "models/faster_rcnn_resnet50_coco/frozen_inference_graph.pb"
model_txt = "models/faster_rcnn_resnet50_coco/graph.pbtxt"
labels_txt = "models/faster_rcnn_resnet50_coco/mscoco_label_map.pbtxt"

# OpenCV DNN 直接调用TensorFlow的导出模型
def OpencvDemo128():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromTensorflow(model_bin, model_txt)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # get layer names and layer IDs
    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)
    logging.info("type of the last layer: {}".format(lastLayer.type))
    
    # load labels
    labelMap = read_label_map(labels_txt)
    if labelMap is None or 0 == len(labelMap):
        logging.error("could not load labels!")
        return cv.Error.StsError
    
    # read an image
    src = cv.imread("images/objects.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    height, width = src.shape[:2]
    
    dst = np.copy(src)

    input = cv.dnn.blobFromImage(src, size=(300, 300), swapRB=True, crop=False)

    # run the model
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setInput(input)
    output = net.forward()

    # get detection results
    score_threshold = 0.4
    for detection in output[0, 0, : , : ]:
        score = float(detection[2])
        if score > score_threshold:
            objId = int(detection[1]) + 1
            left = detection[3] * width
            top = detection[4] * height
            right = detection[5] * width
            bottom = detection[6] * height

            cv.rectangle(dst, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            label = labelMap[objId]
            logging.info("ID: {:d}, label: {}, score: {:.3f}".format(objId, label, score))
            cv.putText(dst, "{} ({:.3f})".format(label, score), 
                       (int(left), int(top) - 8), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
    
    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    logging.info(text)
    cv.putText(dst, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("object detection - Faster-RCNN model via TensorFlow Object Detection API", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def read_label_map(filename):
    labelMap = dict()

    with open(filename, 'rt') as f:
        while True:
            one_line = f.readline()
            if not one_line:
                break
            found = one_line.find("id:")

            if found != -1:
                index = found + 4
                id = one_line[index:-1]

                display_name = f.readline()
                found = display_name.find("display_name:")

                index = found + 15
                name = display_name[index:-2]
                labelMap[int(id)] = name
    
    return labelMap

if __name__ == "__main__":
    OpencvDemo128()

# end of file
