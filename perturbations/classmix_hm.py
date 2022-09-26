import random 
import torch 
import numpy as np 
random.seed(1998)

# by classmix 
def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    mask = (N == 1)
    return mask

   
def select_class(label, ignore_index):
    classes = torch.unique(label) # count class
    classes = classes[classes != ignore_index]  
    nclasses = classes.shape[0]
    # select class
    classes = (classes[torch.LongTensor(np.random.choice(nclasses, int((nclasses - nclasses % 2)/2), replace = False))])
    return classes

def classmix_combine(tgt_images, pseudo_label, batch_size, ignore_index):
    assert batch_size % 2 == 0
    img_idx = list(range(batch_size))
    random.shuffle(img_idx) # 1, 2, 0, 4
    mixed_images = []
    mixed_labels = []
    for _ in range(batch_size // 2):
        id1, id2 = img_idx[0], img_idx[1]
        del img_idx[:2]
        # select class in img1 
        label_a = pseudo_label[id1]
        class_a = select_class(label_a, ignore_index)
        mask = generate_class_mask(label_a, class_a)
        mixed_image_a = torch.where(mask,tgt_images[id1], tgt_images[id2] )
        mixed_label_a = torch.where(mask,pseudo_label[id1], pseudo_label[id2])

        # select class in img2 
        label_b = pseudo_label[id2]
        class_b = select_class(label_b, ignore_index)
        mask = generate_class_mask(label_b, class_b)
        mixed_image_b = torch.where(mask,tgt_images[id2], tgt_images[id1])
        mixed_label_b = torch.where(mask,pseudo_label[id2], pseudo_label[id1])
        
        # append images and labels to list
        mixed_images.append(mixed_image_a)
        mixed_images.append(mixed_image_b)
        mixed_labels.append(mixed_label_a)
        mixed_labels.append(mixed_label_b)

    # generate batch by list
    mixed_images = torch.stack(mixed_images, 0)
    mixed_labels = torch.stack(mixed_labels, 0)
    return mixed_images, mixed_labels 
