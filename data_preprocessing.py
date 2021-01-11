import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pydicom
import cv2
from imgaug import augmenters as iaa
import tensorflow as tf
import tensorflow.keras as keras
from math import ceil

HEIGHT = 256
WIDTH = 256
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)

file_dir = os.path.dirname(__file__)
base_dir = os.path.join(file_dir, 'intracranial_hemorrhage_dataset/')

train_df_path = base_dir + 'final_train_df.csv'
test_df_path = base_dir + 'final_test_df.csv'

sample_dcm_path = base_dir + 'sample_dicoms/'
sample_dcms = [sample_dcm_path + fname for fname in os.listdir(sample_dcm_path)]

train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)


# IMAGE PREPROCESSING
# Fix bad DICOM images
def fix_bad_pxrep(dcm):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        shifted_pixel_arr = dcm.pixel_array + 1000
        mode = 4096
        shifted_pixel_arr = shifted_pixel_arr[shifted_pixel_arr >= mode] - mode
        dcm.PixelData = shifted_pixel_arr.tobytes()
        dcm.RescaleIntercept = -1000


# Crop images to just brain area
def segment_circle(dcm):
    # Make blank mask and convert to grayscale using Otsu's threshold
    original = dcm.copy().astype('uint16')
    mask = np.zeros(original.shape, dtype=np.uint8)
    gray = original
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    # Find contours and filter using area and approximation
    contours = cv2.findContours(close.copy().astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        area = cv2.contourArea(contour)
        if len(approx) > 4 and area > 10000 and area < 500000:
            # Draw circle onto blank mask
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            cv2.circle(mask, (int(x), int(y)), int(r), (255, 255, 255), -1)
            cv2.circle(original, (int(x), int(y)), int(r), (36, 255, 12), 0)

            # Extract ROI
            x, y, w, h = cv2.boundingRect(mask)
            mask_ROI = mask[y: y + h, x: x + w]
            image_ROI = original[y: y + h, x: x + w]

            # Bitwise-and for result
            result = cv2.bitwise_and(image_ROI, image_ROI, mask=mask_ROI)
            return result

    return dcm


# Hounsfield linear transformation
def convert_to_hounsfield_units(dcm):
    img = segment_circle(dcm.pixel_array)
    return (dcm.pixel_array * dcm.RescaleSlope) + dcm.RescaleIntercept


def get_window(dcm, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(dcm, img_min, img_max)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def get_rgb_image(dcm):
    img = convert_to_hounsfield_units(dcm)
    brain_img = get_window(img, 40, 80)
    subdural_img = get_window(img, 80, 200)
    bone_img = get_window(img, 40, 380)
    bsb_img = np.array([brain_img, subdural_img, bone_img]).transpose(1, 2, 0)

    return bsb_img


def image_augmentations():
    sometimes = lambda aug: iaa.Sometimes(1, aug)
    return iaa.Sequential(
        [
            iaa.Fliplr(0.25),
            sometimes(iaa.Crop(px=(0, 25), keep_size=True, sample_independently=False))
        ], random_order=True)


def read_dicom(path, desired_size=(WIDTH, HEIGHT)):
    dcm = pydicom.dcmread(path)
    try:
        img = get_rgb_image(dcm)
    except:
        img = np.zeros(shape=SHAPE)

    print(img.shape)
    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_AREA)
    print(img.shape)

    return img


class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_dir, dataset, labels, augmentation, batch_size=16, img_size=(512, 512),
                 augment=False, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.augment = augment
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data__generation(indices)
        return X, Y

    def augmentor(self, image):
        augment_img = self.augmentation
        image_aug = augment_img.augment_image(image)
        return image_aug

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        np.random.shuffle(self.indices)

    def __data__generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size, 3))
        Y = np.empty((self.batch_size, 6), dtype=np.float32)

        for i, index in enumerate(indices):
            ID = self.ids[index]
            image = read_dicom(self.img_dir + ID, self.img_size)
            if self.augment:
                X[i, ] = self.augmentor(image)
            else:
                X[i, ] = image
            Y[i, ] = self.labels.iloc[index].values

        return X, Y