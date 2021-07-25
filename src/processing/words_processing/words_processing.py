import cv2 as cv
import numpy as np


class WordsProcessing:

    @staticmethod
    def add_padding(new_img, old_w, old_h):
        h1, h2 = int((64 - old_h) / 2), int((64 - old_h) / 2) + old_h
        w1, w2 = int((128 - old_w) / 2), int((128 - old_w) / 2) + old_w
        img_pad = np.ones([64, 128]) * 255
        img_pad[h1:h2, w1:w2] = new_img
        return img_pad

    def fix_size(self):
        h, w = self.img.shape[:2]
        if w < 128 and h < 64:
            self.img = self.add_padding(self.img, w, h)
        elif w >= 128 and h < 64:
            new_w = 128
            new_h = int(h * new_w / w)
            new_img = cv.resize(self.img, (new_w, new_h), interpolation=cv.INTER_AREA)
            self.img = self.add_padding(new_img, new_w, new_h)
        elif w < 128 and h >= 64:
            new_h = 64
            new_w = int(w * new_h / h)
            new_img = cv.resize(self.img, (new_w, new_h), interpolation=cv.INTER_AREA)
            self.img = self.add_padding(new_img, new_w, new_h)
        else:
            ratio = max(w / 128, h / 64)
            new_w = max(min(128, int(w / ratio)), 1)
            new_h = max(min(64, int(h / ratio)), 1)
            new_img = cv.resize(self.img, (new_w, new_h), interpolation=cv.INTER_AREA)
            self.img = self.add_padding(new_img, new_w, new_h)

    def preprocess(self, image):

        self.img = image

        self.fix_size()

        self.img = np.clip(self.img, 0, 255)
        self.img = np.uint8(self.img)
        self.img = self.img.astype(np.float32)
        self.img /= 255

        return self.img
