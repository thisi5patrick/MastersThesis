from __future__ import annotations

import cv2 as cv
import numpy as np


class LettersProcessing:
    def __init__(self, img: np.ndarray):
        self.img = img
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        th, self.threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    def get_lines(self) -> dict:
        lines = []
        line_flag = True
        for i in range(self.threshed.shape[0]):
            row = self.threshed[i]
            cnt = np.count_nonzero(row)
            if line_flag:
                if cnt:
                    lines.append({'start': i})
                    line_flag = False
            else:
                if not cnt:
                    lines[-1]['end'] = i
                    line_flag = True

        return lines

    def split_words_in_lines(self, line: dict):
        line_pixels = self.threshed[line['start']: line['end'], :]
        final_thr = cv.dilate(line_pixels, None, iterations=1)

        contours, hierarchy = cv.findContours(final_thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for i in range(self.threshed.shape[1]):
            ...
