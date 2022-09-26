import os
import time
import numpy as np
import torch
from PIL import Image

np.seterr(divide='ignore', invalid='ignore')


class Eval():
    def __init__(self, num_class, ignore_index):
        self.num_class = num_class + 1
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.ignore_index = ignore_index

    def Pixel_Accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return PA

    def Mean_Pixel_Accuracy(self):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        MPA = np.nanmean(MPA[:self.ignore_index])
        return MPA

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        classMIoU = np.nan_to_num(np.around(MIoU, decimals = 3))
        MIoU = np.nanmean(MIoU[:self.ignore_index])
        return MIoU

    def classMIoU(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        classMIoU = np.nan_to_num(np.around(MIoU, decimals = 3))
        return classMIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        FWIoU = np.multiply(np.sum(self.confusion_matrix, axis=1), np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                         np.diag(self.confusion_matrix))
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(self.confusion_matrix)

        return FWIoU

    def Mean_Precision(self):
        Precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    
    def evaluate_results(self):
        return \
            self.Mean_Intersection_over_Union(), \
            self.classMIoU(), \
            self.Frequency_Weighted_Intersection_over_Union(), \
            self.Mean_Precision(), \
            self.Pixel_Accuracy(),\
            self.Mean_Pixel_Accuracy()

def softmax(k, axis=None):
    exp_k = np.exp(k)
    return exp_k / np.sum(exp_k, axis=axis, keepdims=True)


# def classesMIoU(classes):
#     sorted_indexes = sorted(range(len(classes)), key=lambda k: classes[k], reverse = True)
#     print("Sorted MIoU Class IDS", sorted_indexes)
#     sorted_clses = []
#     for idx in range(len(classes)):
#         sorted_clses.append({
#             'id': sorted_indexes[idx],
#             'name': _classesNames[sorted_indexes[idx]],
#             'miou': classes[sorted_indexes[idx]]
#         })
#     print(sorted_clses)
#     return sorted_clses