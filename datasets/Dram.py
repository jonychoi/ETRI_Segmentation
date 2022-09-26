import os
import numpy as np
from torch.utils import data
from PIL import Image
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
# from torchvision.io import read_img, ImageReadeMode
import torchvision.transforms as transforms
import albumentations as A

class DramDataSet(data.Dataset):
    def __init__(
        self,
        root_dir = "/media/dataset2/DRAM_processed",
        split="train",
        count = 1464,
        num_classes=12,
        transform = None,
        trans_img = True, 
        trans_label = True,
        max_iters=None,
        img_id = False
    ):

        self.root = root_dir
        self.split = split 
        self.transform = transform 
        self.img_id = img_id    
        self.data_list = []
        self.count = count      #shfuffle 1464 ver? 2000 ver? 
        self.trans_img = trans_img 
        self.trans_label = trans_label

        
        if self.split == "train":   # train or test
            list_name = "shuffled{}.txt".format(self.count)
            data_list_path = os.path.join(self.root, split, list_name)
        else :  # test 
            list_name = "dram_with_unseen.txt"
            data_list_path = os.path.join(self.root, split, list_name)

        with open(data_list_path, "r") as handle:
            content = handle.read().splitlines()

        # 이 data_list 에는 img경로, label 경로, 제목이 dict로 포함

        for fname in content:
            name_split = fname.split("/")
            if len(name_split) <= 3:
                name = "/".join(name_split[1:])
            else:  # unseen
                name = "/".join(name_split[2:])

            self.data_list.append(
                {
                    "img": os.path.join(self.root, self.split, "%s.jpg" % fname),
                    "label": os.path.join(self.root, 'labels', "%s.png" % fname),
                    "name": name,
                }
            )    


        if max_iters is not None :
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index) :
        datafiles = self.data_list[index]

        image = np.array(Image.open(datafiles["img"]).convert('RGB'))
        image_original = Image.open(datafiles["img"]).convert('RGB')
        image_path = datafiles["img"]
        name = datafiles["name"]

        if self.split == "train" :  # label은 이미지 복사한걸 뱉어냄 (쓰지않음)
            label = image.copy()
        else : 
            label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
            label = Image.fromarray(label)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        norm = transforms.Normalize(mean, std)
        totensor = transforms.ToTensor()

        if self.transform:
            transformed = self.transform(image = image, mask = label)

            if self.trans_img == True :
                image = transformed['image']
                image = norm(totensor(image))       # 0~1 사이로 바꾼다음 (totensor) normalize      

            if self.trans_label == True:
                label = transformed['mask']

        if self.img_id == True :    # img_id 까지 받고싶다면 
            return image, label, name

        return image, label

