import cv2 as cv
import os
import numpy as np

PATH = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = PATH + '/datasets/images'
LABELS_PATH = PATH + '/datasets/labels'
FINAL_IMAGE_HEIGHT = 64
FINAL_IMAGE_WIDTH = 128


def datasetImageProcessing():
    try:
        os.mkdir(f'{PATH}/processed_datasets/images')
    except FileExistsError:
        ...

    for folder in os.listdir(IMAGES_PATH):
        try:
            os.mkdir(f'{PATH}/processed_datasets/images/{folder}')
        except FileExistsError:
            ...

        for subfolder in os.listdir(f'{IMAGES_PATH}/{folder}'):
            try:
                os.mkdir(f'{PATH}/processed_datasets/images/{folder}/{subfolder}')
            except FileExistsError:
                ...

            for file in os.listdir(f'{IMAGES_PATH}/{folder}/{subfolder}'):
                result = np.full((FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH), 255, dtype=np.uint8)
                image = cv.imread(f'{IMAGES_PATH}/{folder}/{subfolder}/{file}', cv.COLOR_BGR2GRAY)

                img_h, img_w = image.shape[:2]
                ratio = FINAL_IMAGE_HEIGHT / img_h
                calculated_image_width = int(img_w * ratio)

                if calculated_image_width > FINAL_IMAGE_WIDTH:
                    result = cv.resize(image, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))
                    result[result[:, :] > 200] = 255

                else:
                    resized_img = cv.resize(image, (calculated_image_width, int(img_h * ratio)))
                    resized_img[resized_img[:, :] > 100] = 255
                    result[:resized_img.shape[0], : resized_img.shape[1]] = resized_img

                cv.imwrite(f'{PATH}/processed_datasets/images/{folder}/{subfolder}/{file}', result)

