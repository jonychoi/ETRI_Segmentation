import numpy as np
import torch
import cv2
import albumentations as al
import random 

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# albumentation github가서 어떻게 쓰는지 체크해보기

def get_augmentation():
    return al.Compose([
        # al.RandomResizedCrop(480, 480, scale=(0.2, 1.)), #512
        al.RandomResizedCrop(480, 480, scale=(0.5, 1.)),     # 0809 edited by hj
        al.Compose([
            # NOTE: RandomBrightnessContrast replaces ColorJitter
            al.RandomBrightnessContrast(p=1),
            al.HueSaturationValue(p=1),
        ], p=0.8),
        al.ToGray(p=0.2),
        al.GaussianBlur(5, p=0.5),
    ])

def get_geo_augmentation(bg, w, h):
    geo_aug_list = [al.RandomResizedCrop(480, 480, scale= (0.5, 1.))]
    geo_aug_list.extend(random.choices(
        [al.Affine(rotate=(-30, 30), cval_mask = bg, p =0.8), 
        al.Affine(shear = {"x" : (-30, 30)}, cval_mask =bg, p=0.8 ), 
        al.Affine(shear = {"y" : (-30, 30)},  cval_mask = bg, p = 0.8),
        al.Affine(translate_percent = {"x" : (-0.3, 0.3)}, cval_mask =bg, p = 0.8),
        al.Affine(translate_percent = {"y" : (-0.3, 0.3)}, cval_mask = bg, p = 0.8)], k= 1))
    geo_aug_list.extend([al.CoarseDropout(max_holes = 1, max_height = 0.3, max_width = 0.3, mask_fill_value = bg, p =0.8)])
    return al.Compose(geo_aug_list)

def augment(images, labels, aug):
    """Augments both image and label. Assumes input is a PyTorch tensor with 
       a batch dimension and values normalized to N(0,1)."""

    # Transform label shape: B, C, W, H ==> B, W, H, C
    labels_are_3d = (len(labels.shape) == 4)
    if labels_are_3d:
        labels = labels.permute(0, 2, 3, 1)

    # Transform each image independently. This is slow, but whatever.
    aug_images, aug_labels = [], []
    for image, label in zip(images, labels):

        # Step 1: Undo normalization transformation, convert to numpy
        # image = cv2.cvtColor(image.numpy().transpose(
        #     1, 2, 0) + IMG_MEAN, cv2.COLOR_BGR2RGB).astype(np.uint8)
        image = (image.numpy().transpose(1, 2, 0) * IMG_STD) + IMG_MEAN # range(0,1)
        # denorm
        image = image*255. # range(0, 255)
        image = image.astype(np.uint8)
        label = label.numpy()  # convert to np
        
        #import pdb; pdb.set_trace()

        # Step 2: Perform transformations on numpy images
        # albumentation
        data = aug(image=image, mask=label)
        image, label = data['image'], data['mask']

        
        # Step 3: Convert back to PyTorch tensors
        #image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)
        image = image/255. # range(0, 1), shape=(h,w,c)
        image = (image - IMG_MEAN) / IMG_STD
        image = torch.from_numpy(image.transpose(2, 0, 1))
        label = torch.from_numpy(label)
        if not labels_are_3d:
            label = label.long()

        # Add to list
        aug_images.append(image)
        aug_labels.append(label)

    # Stack
    images = torch.stack(aug_images, dim=0)
    labels = torch.stack(aug_labels, dim=0)

    # Transform label shape back: B, C, W, H ==> B, W, H, C
    if labels_are_3d:
        labels = labels.permute(0, 3, 1, 2)

    return images, labels


