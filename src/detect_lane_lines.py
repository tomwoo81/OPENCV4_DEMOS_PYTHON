#!/usr/bin/env python3
#coding = utf-8

import logging
import numpy as np
import cv2 as cv

class LaneLineDetector:
    def __init__(self):
        self.left_line = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        self.right_line = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    
    def process(self, src):
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        binary = cv.Canny(gray, 150, 300)

        h, w = gray.shape
        binary[0 : h // 2 + 40, 0 : w] = 0
        binary[h // 2 + 40 : h, 0 : 350] = 0
        binary[h // 2 + 40 : h, w - 350 : w] = 0

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours_img = np.zeros((h, w), binary.dtype)
        for i in range(len(contours)):
            # 计算面积与周长
            perimeter = cv.arcLength(contours[i], True)
            area = cv.contourArea(contours[i])
            if perimeter < 5 or area < 10:
                continue
            y = cv.boundingRect(contours[i])[1]
            if y > (h - 50):
                continue
            angle1 = cv.minAreaRect(contours[i])[2]
            angle1 = abs(angle1)
            if angle1 < 20 or angle1 > 160 or angle1 == 90.0:
                continue
            if len(contours[i]) > 5:
                angle2 = cv.fitEllipse(contours[i])[2]
                if angle2 < 5 or angle2 > 160 or 80 < angle2 < 100:
                    continue
            cv.drawContours(contours_img, contours, i, (255), 2)
        
        lines_img = self.fitLines(contours_img)
        dst = cv.addWeighted(src, 0.8, lines_img, 0.5, 0)

        return contours_img, dst
    
    def fitLines(self, binary):
        h, w = binary.shape
        cx = w // 2
        cy = h // 2
        left_pts = []
        right_pts = []

        for col in range(0, cx):
            for row in range(cy, h):
                pv = binary[row, col]
                if pv == 255:
                    left_pts.append((col, row))
        for col in range(cx, w):
            for row in range(cy, h):
                pv = binary[row, col]
                if pv == 255:
                    right_pts.append((col, row))
        
        lines_img = np.zeros((h, w, 3), dtype=np.uint8)

        y1 = h // 2 + 40
        y2 = h

        if len(left_pts) >= 2:
            [vx, vy, x0, y0] = cv.fitLine(np.array(left_pts), cv.DIST_L1, 0, 0.01, 0.01)

            # 直线参数斜率k与截矩b
            k = vy / vx
            b = y0 - k * x0

            x1 = (y1 - b) / k
            x2 = (y2 - b) / k

            cv.line(lines_img, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (0, 0, 255), 8)

            self.left_line['x1'] = np.int(x1)
            self.left_line['y1'] = np.int(y1)
            self.left_line['x2'] = np.int(x2)
            self.left_line['y2'] = np.int(y2)
        else:
            x1 = self.left_line['x1']
            y1 = self.left_line['y1']
            x2 = self.left_line['x2']
            y2 = self.left_line['y2']

            cv.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 8)

        if len(right_pts) >= 2:
            [vx, vy, x0, y0] = cv.fitLine(np.array(right_pts), cv.DIST_L1, 0, 0.01, 0.01)

            # 直线参数斜率k与截矩b
            k = vy / vx
            b = y0 - k * x0

            x1 = (y1 - b) / k
            x2 = (y2 - b) / k

            cv.line(lines_img, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (0, 0, 255), 8)

            self.right_line['x1'] = np.int(x1)
            self.right_line['y1'] = np.int(y1)
            self.right_line['x2'] = np.int(x2)
            self.right_line['y2'] = np.int(y2)
        else:
            x1 = self.right_line['x1']
            y1 = self.right_line['y1']
            x2 = self.right_line['x2']
            y2 = self.right_line['y2']

            cv.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 8)
        
        return lines_img

def detect_lane_lines():
    logging.basicConfig(level=logging.DEBUG)
    
    capture = cv.VideoCapture("videos/road_line.mp4")
    if not capture.isOpened():
        logging.error("could not read a video file!")
        return cv.Error.StsError
    
    width = capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv.CAP_PROP_FPS)
    count = capture.get(cv.CAP_PROP_FRAME_COUNT)
    logging.debug("width: {:.0f}, height: {:.0f}, fps: {:.0f}, count: {:.0f}".format(width, height, fps, count))

    detector = LaneLineDetector()

    result = None
    winName = "lane line detection"
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    while True:
        ret, src = capture.read()
        if ret:
            contours_img, dst = detector.process(src)

            h, w, ch = src.shape
            if result is None:
                result = np.zeros([h * 3, w, ch], dtype=src.dtype)
            result[0 : h, 0 : w, :] = src
            contours_img = cv.cvtColor(contours_img, cv.COLOR_GRAY2BGR) # 1 channel -> 3 channels
            result[h : h * 2, 0 : w, :] = contours_img
            result[h * 2 : h * 3, 0 : w, :] = dst
            cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
            cv.putText(result, "image with contours", (10, h + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
            cv.putText(result, "image with results", (10, h * 2 + 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
            cv.resizeWindow(winName, (w // 2, h * 3 // 2))
            cv.imshow(winName, result)

            c = cv.waitKey(1)

            if c == 27: # ESC
                break
        else:
            break
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()
    
    return cv.Error.StsOk

if __name__ == "__main__":
    detect_lane_lines()

# end of file
