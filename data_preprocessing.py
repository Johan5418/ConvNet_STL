import numpy as np
import matplotlib.pyplot as plt
import cv2

def process_image(img):
    #img = cv2.imread('Camera_for_preprocessing.jpg')
    #image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    imggray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    eq_img = cv2.equalizeHist(imggray)
    reduced_img = cv2.GaussianBlur(eq_img, (7,7), 1.0)

    return reduced_img