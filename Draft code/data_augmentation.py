import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def img_augmentation(img, overlaying):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlaying = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    integer = random.randint(1, 3)

    if integer == 1:
        h_flip = cv2.flip(img, 1) #horizontal flip
        (h, w) = h_flip.shape[:2]
        center = (w//2 , h//2)

        angle = 257
        scale = random.uniform(0.8, 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(h_flip, M, (w, h))

        def overlay_with_opacity(background, overlay, alpha_range=(0.1, 0.3)):
            bg = background.copy()

            # Resize overlay to match background
            overlay = cv2.resize(overlay, (bg.shape[1], bg.shape[0]))

            # Random opacity
            alpha = random.uniform(*alpha_range)

            # Alpha blending
            blended = cv2.addWeighted(bg, 1 - alpha, overlay, alpha, 0)
            return blended

        overlay_img = overlay_with_opacity(rotated, overlaying)

        return overlay_img
        
    elif integer == 2:
        h_flip = cv2.flip(img, 0) #vertical
        (h, w) = h_flip.shape[:2]
        center = (w//2 , h//2)

        angle = 36
        scale = random.uniform(0.8, 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(h_flip, M, (w, h))

        def block_mask(img, times, block_size=200):
            h, w = img.shape[:2]
            img = img.copy()

            for i in range(times):
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)

                y1 = max(0, y - block_size // 2)
                y2 = min(h, y + block_size // 2)
                x1 = max(0, x - block_size // 2)
                x2 = min(w, x + block_size // 2)

                img[y1:y2, x1:x2] = 0

            return img

        info_loss_img = block_mask(rotated, 3)

        return info_loss_img

    elif integer == 3:
        h_flip = cv2.flip(img, -1) #both
        (h, w) = h_flip.shape[:2]
        center = (w//2 , h//2)

        angle = 124
        scale = random.uniform(0.8, 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(h_flip, M, (w, h))

        mask = np.ones(rotated.shape[:2], dtype=np.uint8) * 255

        color_aug = cv2.colorChange(
            rotated,
            mask,
            red_mul=1.3,
            green_mul=0.5,
            blue_mul=0.5
        )

        return color_aug

