#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

model_bin = "models/yolov3/yolov3.weights"
model_txt = "models/yolov3/yolov3.cfg"
labels_txt = "models/yolov3/object_detection_classes_yolov3.txt"

# OpenCV DNN 使用OpenVINO加速
def OpencvDemo138():
    logging.basicConfig(level=logging.DEBUG)

    # load a DNN model
    net = cv.dnn.readNetFromDarknet(model_txt, model_bin)
    if net is None:
        logging.error("could not load a DNN model!")
        return cv.Error.StsError
    
    # get names of unconnected out layers
    outNames = net.getUnconnectedOutLayersNames()
    
    # load labels
    labels = None
    with open(labels_txt, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    if labels is None or 0 == len(labels):
        logging.error("could not load labels!")
        return cv.Error.StsError
    
    # read an image
    src = cv.imread("images/objects.jpg")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    dst1 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_BACKEND_OPENCV, outNames, labels, src)
    dst2 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, outNames, labels, src)
    dst3 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_BACKEND_CUDA, outNames, labels, src)
    dst4 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CUDA_FP16, cv.dnn.DNN_BACKEND_CUDA, outNames, labels, src)
    
    dst1 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_BACKEND_OPENCV, outNames, labels, src)
    dst2 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, outNames, labels, src)
    dst3 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_BACKEND_CUDA, outNames, labels, src)
    dst4 = RunYolov3Model(net, cv.dnn.DNN_TARGET_CUDA_FP16, cv.dnn.DNN_BACKEND_CUDA, outNames, labels, src)
    
    h, w, ch = src.shape
    result = np.zeros([h * 2, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = dst1
    result[0 : h, w : w * 2, :] = dst2
    result[h : h * 2, 0 : w, :] = dst3
    result[h : h * 2, w : w * 2, :] = dst4
    cv.putText(result, "target: CPU, backend: OpenCV", (10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
    cv.putText(result, "target: CPU, backend: OpenVINO", (w + 10, 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
    cv.putText(result, "target: GPU, backend: CUDA", (10, h + 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
    cv.putText(result, "target: GPU (FP16), backend: CUDA", (w + 10, h + 30), cv.FONT_ITALIC, 0.6, (0, 0, 255), 1)
    cv.imshow("computational targets & backends of DNN module", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

def RunYolov3Model(net, targetId, backendId, unconnectedOutLayersNames, labels, src):
    height, width = src.shape[:2]
    
    dst = np.copy(src)

    input = cv.dnn.blobFromImage(src, 1 / 255, (416, 416), None, True, False)

    # run the model
    net.setPreferableTarget(targetId)
    net.setPreferableBackend(backendId)
    net.setInput(input)
    outputs = net.forward(unconnectedOutLayersNames)

    # get detection results
    boxes = []
    classIds = []
    confidences = []
    confidence_threshold = 0.5

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # numbers are [center_x, center_y, width, height]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                boxWidth = int(detection[2] * width)
                boxHeight = int(detection[3] * height)
                left = int(center_x - boxWidth / 2)
                top = int(center_y - boxHeight / 2)

                boxes.append((left, top, boxWidth, boxHeight))
                classIds.append(classId)
                confidences.append(float(confidence))
    
    # perform non maximum suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    for index in indices:
        box = boxes[index]
        label = labels[classIds[index]]
        confidence = confidences[index]
        cv.rectangle(dst, box, (0, 0, 255), 1)
        cv.putText(dst, "{} ({:.3f})".format(label, confidence), \
                   (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    time = net.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time: {:.0f} ms".format(time)
    logging.info(text)
    cv.putText(dst, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return dst

if __name__ == "__main__":
    OpencvDemo138()

# end of file
