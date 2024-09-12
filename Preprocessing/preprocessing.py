import torch
import random 
import sys
import time
import cv2
import matplotlib
import os
import pickle
import platform

import nibabel as nib
import pydicom as dicom
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import statsmodels.api as sm
import matplotlib.pyplot as plt

from torch import nn
from skimage.transform import resize, rescale, downscale_local_mean
from scipy.ndimage import rotate as rotate_image
from torch.utils.data import DataLoader
from sklearn import preprocessing        #pip install scikit-learn
from parameters import MetaParameters


class ReadImages():
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file

    def get_nii(self):
        # matplotlib.use('TkAgg')
        img = nib.load(self.path_to_file)

        return img

    def get_dcm(self):
        origin_dicom = dicom.dcmread(self.path_to_file)
        new_dicom = np.array(origin_dicom.pixel_array)
        
        if len(list(new_dicom.shape)) == 2:
            new_dicom = new_dicom[:, :, np.newaxis]
        else:
            new_dicom = new_dicom.transpose(2, 1, 0)

        return new_dicom

    def get_nii_fov(self):
        img = nib.load(self.path_to_file)
        return img.header.get_zooms()

    def view_matrix(self):
        return np.array(self.get_nii().dataobj)

    def get_file_list(self):
        files = os.listdir(self.path_to_file)
        files.sort()
        return files

    def get_file_path_list(self):
        path_list = []

        for root, subfolder, files in os.walk(self.path_to_file):
            for item in files:
                if item.endswith('.nii') or item.endswith('.dcm'):
                    filenamepath = str(os.path.join(root, item))
                    path_list.append(filenamepath)

        return path_list

    def get_dataset_list(self):
        return list(self.get_file_list())


class PreprocessData(MetaParameters):

    def __init__(self, image, mask = None, names = None):    
        super(MetaParameters, self).__init__()
        self.image = image
        self.mask = mask
        self.names = names

    def preprocessing(self, kernel_sz):
        image = np.array(self.image, dtype = np.float32)
        image = self.clipping(image)
        image = self.normalization(image)
        # image = self.z_normalization(image)
        image = self.equalization_matrix(kernel_sz, matrix = image)
        image = self.rescale_matrix(kernel_sz, matrix = image, order = None)
        image = np.array(image.reshape(kernel_sz, kernel_sz, 1), dtype = np.float32)

        if self.mask is not None:
            mask = np.array(self.mask, dtype = np.float32)
            mask = self.equalization_matrix(kernel_sz, matrix = mask)
            mask = self.rescale_matrix(kernel_sz, matrix = mask, order = 0)
            mask = np.array(mask.reshape(kernel_sz, kernel_sz, 1), dtype = np.float32)
        else:
            mask = None
        
        return image, mask

    def clipping(self, image):
        image_max = np.max(image)
        if self.CLIP_RATE is not None:
            image = np.clip(image, self.CLIP_RATE[0] * image_max, self.CLIP_RATE[1] * image_max)
        
        return image

    @staticmethod
    def normalization(image):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        return image / np.max(image)

    @staticmethod
    def z_normalization(image):
        mean, std = np.mean(image), np.std(image)
        image = (image - mean) / std
        image += abs(np.min(image))
        
        return image / np.max(image)

    @staticmethod
    def equalization_matrix(kernel_sz, matrix):
        max_kernel = max(matrix.shape[0], matrix.shape[1])

        if max_kernel <= kernel_sz: 
            zero_matrix = np.zeros((kernel_sz, kernel_sz))
        else:
            zero_matrix = np.zeros((max_kernel, max_kernel))

        zero_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        matrix = zero_matrix

        return matrix

    @staticmethod
    def center_cropping(matrix):
        y, x = matrix.shape
        min_kernel = min(matrix.shape[0], matrix.shape[1])
        startx = (x - min_kernel)//4*3
        starty = (y - min_kernel)//4*3
        
        return matrix[starty:starty + min_kernel, startx:startx + min_kernel]

    def rescale_matrix(self, kernel_sz, matrix, order=None):
        shp = matrix.shape
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        scale =  kernel_sz / max_kernel
        
        return rescale(matrix, (scale, scale), anti_aliasing = False, order=order)

    def shuff_dataset(self):
        temp = list(zip(self.images, self.masks, self.names))
        random.shuffle(temp)
        images, masks, names = zip(*temp)
        
        return list(images), list(masks), list(names)

    def rotate_2d(self, angle):
        image = rotate_image(self.image, angle)
        mask = rotate_image(self.mask, angle)
        
        return image, mask

    def gauss_noise(self, sigma, kernel_sz):
        mean = 0.5
        noise = np.random.normal(mean, sigma**0.5, self.image.shape)
        noisy = self.image + noise
        
        noisy = np.array(noisy.reshape(kernel_sz, kernel_sz, 1), dtype=np.float32)
        mask = np.array(self.mask.reshape(kernel_sz, kernel_sz, 1), dtype=np.float32)
        
        return noisy, mask

    def rician_noise_transforms(self, kernel_sz, random_s, random_v):
        try:
            N = kernel_sz * kernel_sz  # how many samples
            image = np.array(self.image, dtype = np.float32)
            noise = np.random.normal(scale = random_s, size=(N, 2)) + [[random_v, 0]]
            noise = np.linalg.norm(noise, axis = 1)
            noise = noise.reshape(image.shape)
            image = image + noise

        except:
            print('EROORR')
        return image


class EvalPreprocessData(MetaParameters):

    def __init__(self, images = None, masks = None, templates = None):         
        super(MetaParameters, self).__init__()
        self.images = images
        self.masks = masks
        self.templates = templates

    def presegmentation_tissues(self, def_coord, gap_1=None, close_crop=None):
        list_top, list_bot, list_left, list_right = [], [], [], []
        shp = self.images.shape
        count = 0

        if self.CROPP_KERNEL % 16:
            gap = self.CROPP_KERNEL // 2
        else:
            gap = 32
        
        last_top, last_bot, last_left, last_right = (shp[0] // 2 - gap), (shp[1] // 2 - gap), (shp[0] // 2 + gap), (shp[1] // 2 + gap)

        for slc in range(shp[2]):
            image = self.images[:, :, slc]
            mask = self.masks[:, :, slc]

            if (mask != 0).any():
                count += 1
                predict_mask = np.where(mask != 0)
                last_top = np.min(predict_mask[0])
                last_bot = np.max(predict_mask[0])
                last_left = np.min(predict_mask[1])
                last_right = np.max(predict_mask[1])
            else:
                count += 1

            list_top.append(last_top)
            list_bot.append(last_bot)
            list_left.append(last_left)
            list_right.append(last_right)

        mean_top = np.array(list_top).sum() // count
        mean_left = np.array(list_left).sum() // count
        mean_bot = np.array(list_bot).sum() // count 
        mean_right = np.array(list_right).sum() // count

        if def_coord is None:
            center_row = (mean_bot + mean_top) // 2
            center_column = (mean_left + mean_right) // 2
        else:
            center_row, center_column = def_coord
            print(center_row, center_column)

        if self.UNET3 or close_crop:
            for slc in range(shp[2]):
                image_template = np.zeros((shp[0], shp[1])).copy()

                if list_top[slc] == (shp[0] // 2 - gap) and list_bot[slc] == (shp[1] // 2 - gap) and list_left[slc] == (shp[0] // 2 + gap) and list_right[slc] == (shp[1] // 2 + gap):
                    image_template[center_row - 2*gap_1 : center_row + 2*gap_1, center_column - 2*gap_1 : center_column + 2*gap_1] = 1
                elif list_top[slc] > mean_bot and list_bot[slc] > mean_bot and list_left[slc] > mean_bot and list_right[slc] > mean_bot:
                    image_template[center_row - 2*gap_1 : center_row + 2*gap_1, center_column - 2*gap_1 : center_column + 2*gap_1] = 1
                elif list_top[slc] > mean_right and list_bot[slc] > mean_right and list_left[slc] > mean_right and list_right[slc] > mean_right:
                    image_template[center_row - 2*gap_1 : center_row + 2*gap_1, center_column - 2*gap_1 : center_column + 2*gap_1] = 1
                elif list_top[slc] < mean_top and list_bot[slc] < mean_top and list_left[slc] < mean_top and list_right[slc] < mean_top:
                    image_template[center_row - 2*gap_1 : center_row + 2*gap_1, center_column - 2*gap_1 : center_column + 2*gap_1] = 1
                elif list_top[slc] < mean_left and list_bot[slc] < mean_left and list_left[slc] < mean_left and list_right[slc] < mean_left:
                    image_template[center_row - 2*gap_1 : center_row + 2*gap_1, center_column - 2*gap_1 : center_column + 2*gap_1] = 1
                else:
                    image_template[list_top[slc] - gap_1 : list_bot[slc] + gap_1, list_left[slc] - gap_1 : list_right[slc] + gap_1] = 1

                self.images[:, :, slc] = self.images[:, :, slc] * image_template

        images = self.images[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]
        masks = self.masks[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]

        if self.templates is not None:
            templates = self.templates[center_row - gap: center_row + gap, center_column - gap: center_column + gap, :]

        return images, masks, templates, [center_row, center_column]


class ViewData():

    def view_img(self, img):
        width, height, queue = img.shape
        array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        print(width, height, queue)
        num = 1
        for i in range(0, queue, 1):
            img_arr = img.dataobj[:, :, i]
            plt.subplot(4, 5, num)
            plt.imshow(img_arr, cmap='gray')
            num += 1
        plt.show()



