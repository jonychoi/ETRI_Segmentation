"""refer to : https://github.com/WilhelmT/ClassMix.git"""

import numpy as np
import random
import torch
import torch.nn as nn

# generate_class_mask
# transformmasks.strongTransform

# def strongTransform(parameters, data=None, target=None):
#     assert ((data is not None) or (target is not None))
#     IMG_MEAN = [0.485, 0.456, 0.406]
#     data, target = mix(mask = parameters["Mix"], data = data, target = target)
#     # data, target = transformsgpu.colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
#     # data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = None)
#     # data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
#     return data, target

def strongTransform(mixmask, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = mix(mask = mixmask, data = data, target = target)
    return data, target

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

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



def generate_mask_batch(batch_size, pseudo_label ):
    for image_i in range(batch_size):
        classes = torch.unique(pseudo_label[image_i])
        classes = classes[classes != ignore_index]
        nclasses = classes.shape[0]
        classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
        if image_i == 0:
            MixMask = transformmasks.generate_class_mask(pseudo_label[image_i], classes).unsqueeze(0).cuda()
        else:
            MixMask = torch.cat((MixMask,transformmasks.generate_class_mask(pseudo_label[image_i], classes).unsqueeze(0).cuda()))
    return MixMask

## classmix combine 

def classmix_combine(x_t, pseudo_label, batch_size, ignore_index):
    MixMask = generate_mask_batch(batch_size, pseudo_label, ignore_index)
    
    x_classmixed, _ = strongTransform(MixMask, data = batch)   # classmixed img
    y_classmixed, _ = strongTransform(MixMask, data = pseudo_label)    #classmixed pseduolabel

    return x_classmixed, y_classmixed
