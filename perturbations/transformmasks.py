"""https://github.com/WilhelmT/ClassMix.git"""

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erfinv
import torch

def generate_cutout_mask(img_size, seed = None):
    np.random.seed(seed)

    cutout_area = img_size[0] * img_size[1] / 2

    w = np.random.randint(img_size[1] / 2, img_size[1] + 1)
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)
'''
def generate_bernoulli_mask(img_size, sigma, p, seed=None):
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = N
    #Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    # Compute threshold
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    return (Ns > t).astype(float) # Apply threshold and return'''

def generate_cow_mask(img_size, sigma, p, seed=None):
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    # Compute threshold
    t = erfinv(p*2 - 1) * (2**0.5) * Ns.std() + Ns.mean()
    return (Ns > t).astype(float) # Apply threshold and return
'''
def generate_cloud_mask(img_size, sigma, p,seed=None):
    T=10
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    Ns_norm = (Ns-Ns.mean())/Ns.std()
    Ns_sharp = np.tanh(T*Ns_norm)
    Ns_normalised = (Ns_sharp - np.min(Ns_sharp))/np.ptp(Ns_sharp)
    return Ns_normalised'''

# generate_class_mask(argmax_u_w[image_i], classes)

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

### segmento edited ###
### function for generate mask_a and mask_b


'''
def generate_cow_class_mask(pred, classes, sigma, p,):
    N=np.zeros(pred.shape)
    pred = np.array(pred.cpu())
    for c in classes:
        N[pred==c] = generate_cow_mask(pred.shape,sigma,p)[pred==c]
    return N'''


import numpy as np
# import kornia
import torch
import random
import torch.nn as nn

"""https://github.com/WilhelmT/ClassMix.git"""
"""transformsgpu"""
# def colorJitter(colorJitter, img_mean, data = None, target = None, s=0.25):
#     # s is the strength of colorjitter
#     #colorJitter
#     if not (data is None):
#         if data.shape[1]==3:
#             if colorJitter > 0.2:
#                 img_mean, _ = torch.broadcast_tensors(img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3), data)
#                 seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s,hue=s))
#                 data = (data+img_mean)/255
#                 data = seq(data)
#                 data = (data*255-img_mean).float()
#     return data, target

# def gaussian_blur(blur, data = None, target = None):
#     if not (data is None):
#         if data.shape[1]==3:
#             if blur > 0.5:
#                 sigma = np.random.uniform(0.15,1.15)
#                 kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
#                 kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
#                 kernel_size = (kernel_size_y, kernel_size_x)
#                 seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
#                 data = seq(data)
#     return data, target

# def flip(flip, data = None, target = None):
#     #Flip
#     if flip == 1:
#         if not (data is None): data = torch.flip(data,(3,))#np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
#         if not (target is None):
#             target = torch.flip(target,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
#     return data, target

def cowMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def normalize(MEAN, STD, data = None, target = None):
    #Normalize
    if not (data is None):
        if data.shape[1]==3:
            STD = torch.Tensor(STD).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            MEAN = torch.Tensor(MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            STD, data = torch.broadcast_tensors(STD, data)
            MEAN, data = torch.broadcast_tensors(MEAN, data)
            data = ((data-MEAN)/STD).float()
    return data, 



def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    IMG_MEAN = [0.485, 0.456, 0.406]
    data, target = mix(mask = parameters["Mix"], data = data, target = target)
    # data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    # data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = None)
    # data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    return data, target

def getWeakInverseTransformParameters(parameters):
    return parameters

def getStrongInverseTransformParameters(parameters):
    return parameters