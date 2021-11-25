import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

cascade = cv2.CascadeClassifier('tests_cascade/cascade.xml')


def detect_object(img):
    img_copy = img.copy()

    img_copy_rect = cascade.detectMultiScale(img_copy,
                                             scaleFactor=1.2,
                                             minNeighbors=5)

    for (x, y, w, h) in img_copy_rect:
        cv2.rectangle(img_copy, (x, y),
                      (x + w, y + h), (255, 255, 255), 10)
    return img_copy


def load_images():
    images = [
        cv2.imread(os.path.join("images", file)) for file in os.listdir("images")
        if file.endswith(".jpg")
    ]
    return images


if __name__ == '__main__':
    for i, img in enumerate(load_images()):
        detected_img = detect_object(img.copy())
        plt.imshow(detected_img)
        plt.show()
        cv2.imwrite(f'results/test_{i}.jpg', detected_img)
