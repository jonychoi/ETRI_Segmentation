import os
import numpy as np
from torch.utils import data
from PIL import Image


class DramDataSet(data.Dataset):
    def __init__(
        self,
        root,
        movement = "dram",
        max_iters=None,
        num_classes=12,
        split="train",
        transform=None,
        ignore_label=255,
        debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = root
        self.data_list = []
        self.movement = movement

        data_list_path = os.path.join(self.data_root, split, movement + ".txt")
        with open(data_list_path, "r") as handle:
            content = handle.read().splitlines()

        for fname in content:
            name_split = fname.split("/")
            if len(name_split) <= 3:
                name = "/".join(name_split[1:])
            else:  # unseen
                name = "/".join(name_split[2:])

            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, split, "%s.jpg" % fname),
                    "label": os.path.join(self.data_root, 'labels', "%s.png" % fname),
                    "name": name,
                }
            )
        print("Loaded DRAM {} {} set with {} images".format(self.movement, self.split, len(self.data_list)))

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
        
        # pascal => dram
        self.id_to_trainid = {
            0: 0,
            3: 1,
            4: 2,
            5: 3,
            8: 4,
            9: 5,
            10: 6,
            12: 7,
            13: 8,
            15: 9,
            16: 10,
            17: 11,
        }
        self.trainid2name = {
            0: 'background',
            1: 'bird',
            2: 'boat',
            3: 'bottle',
            4: 'cat',
            5: 'chair',
            6: 'cow',
            7: 'dog',
            8: 'horse',
            9: 'person',
            10: 'pottedplant',
            11: 'sheep',
        }

        ## SEGMENTO EDITED ##
        self.pascallabel2segmento_id = {
            0: 0,   #back
            3: 7,   #bird 
            4: 5,   #boat
            5: 19,  #bottle
            8: 8,   #cat
            9: 31,  #chair
            10: 12, #cow
            12: 9,  #dog
            13: 10, #horse
            15: 1,  #person
            16: 32, #pottedplant
            17: 11, #sheep
        }
        ## --------------- ##

        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        image_path = datafiles["img"]
        name = datafiles["name"]
        size = np.array(np.array(image).shape[:2])
        if self.split != 'train':
            label = np.array(Image.open(datafiles["label"]),dtype=np.float32)
            # np unique array([ 0.,  8.,  9., 15.], dtype=float32)
            
            # re-assign labels to match the format of Cityscapes
            # label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            # for k, v in self.id_to_trainid.items():
            #     label_copy[label == k] = v
            # label = Image.fromarray(label_copy)

            ## SEGMENTO EDITED ## => not for the cityscapes, but for the DRAM test labels
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.pascallabel2segmento_id.items():
                label_copy[label == k] = v
            label = Image.fromarray(label_copy)
        else:
            label = image.copy()  # decoy for later use

        ## SEGMENTO EDITED ##

        # if self.transform is not None:
        #     image, label = self.transform(image, label)

        # if self.split != 'train':
        #     return image, label, name
        # else:
        #     return image, size, name

        if self.transform is not None:
            image, label = self.transform(image, label)

        if self.split != 'train':
            return name, image_path, image, label
        else:
            return image, label