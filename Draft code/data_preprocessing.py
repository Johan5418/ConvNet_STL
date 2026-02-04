import numpy as np
import matplotlib.pyplot as plt
import cv2

def process_image(img):
    # If PyTorch CHW → HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)

    img = img.astype(np.uint8)

    resized_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    imggray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    eq_img = cv2.equalizeHist(imggray)
    reduced_img = cv2.GaussianBlur(eq_img, (7,7), 1.0)

    return reduced_img