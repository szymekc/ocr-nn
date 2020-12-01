# import the necessary packages
import numpy as np
import cv2
from preprocess import denoise


def find_countours(image):
    ret, thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    return cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def par_to_batch(path):
    par = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    par = denoise(par)
    edged = cv2.Canny(par, 50, 200)
    cnts = find_countours(edged)[0]
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=True)
    rois = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        if h < 16:
            continue
        rois.append(par[y:y + h, x:x + w])
    return rois
