import os.path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms
import albumentations as A

class ETRI(Dataset):
    def __init__(
        self, 
        root_dir = "/media/dataset2/etri",
        split = "train",
        transform = None,
        img_paths = False,
        trans_img = True,
        trans_label = True,
    ):
        self.root = root_dir
        self.transform = transform
        self.paintings = []
        self.split = split
        self.img_paths = img_paths
        self.trans_img = trans_img
        self.trans_label = trans_label

        if self.split in ["train", "val", "all"]:
            file_list = os.path.join(self.root, 'image_set/' + self.split + ".txt")
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.paintings = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def __len__(self):
        return len(self.paintings)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_id = self.paintings[idx]

        image_path = os.path.join(self.root, "image", image_id + ".jpg")
        label_path = os.path.join(self.root, "annotation", image_id + ".png")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[::,::,::-1] # (BGR to RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        norm = transforms.Normalize(mean, std)
        totensor = transforms.ToTensor()

        if self.transform:
            transformed = self.transform(image=image, mask=label)

            if self.trans_img == True:
                image = transformed['image']
                image = norm(totensor(image))

            if self.trans_label == True:
                label = transformed['mask']

        if self.img_paths == True:
            return image_id, image_path, image, label
        
        return image, label