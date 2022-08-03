#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

face_model_bin = "models/face_detector/opencv_face_detector_uint8.pb"
face_model_txt = "models/face_detector/opencv_face_detector.pbtxt"

gender_model_bin = "models/cnn_age_gender_models/gender_net.caffemodel"
gender_model_txt = "models/cnn_age_gender_models/gender_deploy.prototxt"

age_model_bin = "models/cnn_age_gender_models/age_net.caffemodel"
age_model_txt = "models/cnn_age_gender_models/age_deploy.prototxt"

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# OpenCV DNN 实现性别与年龄预测
def OpencvDemo137():
    logging.basicConfig(level=logging.DEBUG)

    # load DNN models
    faceNet = cv.dnn.readNet(face_model_bin, face_model_txt)
    if faceNet is None:
        logging.error("could not load a DNN model for face detection!")
        return cv.Error.StsError
    
    genderNet = cv.dnn.readNet(gender_model_bin, gender_model_txt)
    if genderNet is None:
        logging.error("could not load a DNN model for gender prediction!")
        return cv.Error.StsError
    
    ageNet = cv.dnn.readNet(age_model_bin, age_model_txt)
    if ageNet is None:
        logging.error("could not load a DNN model for age prediction!")
        return cv.Error.StsError
    
    # configure the models
    faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)

    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)

    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    # ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)

    # read an image
    src = cv.imread("images/persons.png")
    if src is None:
        logging.error("could not load an image!")
        return cv.Error.StsError
    
    height, width = src.shape[:2]

    dst = np.copy(src)

    faceInput = cv.dnn.blobFromImage(src, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    # run the model for face detection
    faceNet.setInput(faceInput)
    faceOutput = faceNet.forward()

    time = faceNet.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
    text = "time of face detection: {:.0f} ms".format(time)
    logging.info(text)

    # get detection results
    score_threshold = 0.5
    padding = 10
    for detection in faceOutput[0, 0, : , : ]:
        score = float(detection[2])
        if score > score_threshold:
            left = detection[3] * width
            top = detection[4] * height
            right = detection[5] * width
            bottom = detection[6] * height

            cv.rectangle(dst, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 1)
            cv.putText(dst, "{:.3f}".format(score), 
                       (int(left), int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            roi = src[max(0, int(top) - padding):min(int(bottom) + padding, height), 
                      max(0, int(left) - padding):min(int(right) + padding, width)]

            genderAgeInput = cv.dnn.blobFromImage(roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), False, False)

            # run the model for gender prediction
            genderNet.setInput(genderAgeInput)
            genderOutput = genderNet.forward()

            time = genderNet.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
            text = "time of gender prediction: {:.0f} ms".format(time)
            logging.info(text)

            # get the class with the highest score
            gender = genderList[np.argmax(genderOutput.flatten())]

            cv.putText(dst, gender, (int(left), int(bottom) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # run the model for age prediction
            ageNet.setInput(genderAgeInput)
            ageOutput = ageNet.forward()

            time = ageNet.getPerfProfile()[0] * 1000 / cv.getTickFrequency()
            text = "time of age prediction: {:.0f} ms".format(time)
            logging.info(text)

            # get the class with the highest score
            age = ageList[np.argmax(ageOutput.flatten())]

            cv.putText(dst, age, (int(left), int(bottom) + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    h, w, ch = src.shape
    result = np.zeros([h, w * 2, ch], dtype=src.dtype)
    result[0 : h, 0 : w, :] = src
    result[0 : h, w : w * 2, :] = dst
    cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.putText(result, "image with results", (w + 10, 30), cv.FONT_ITALIC, .6, (0, 0, 255), 1)
    cv.imshow("gender & age prediction", result)

    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    OpencvDemo137()

# end of file
