import os, sys, glob, cv2, argparse
#sys.path.append("/home/cvlab02/project/etri")
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as data
import torch.optim as optim
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

import albumentations as A
import numpy as np

from utils.label_final import configuration, label_mapper, get_labelnames
from utils.args import str2bool
from utils.eval import Eval 
from utils.io import read_image, write_segm_img
from utils.pallete import get_mask_pallete
from utils.get_model import get_model

from datasets.etri import ETRI

def run(args):

    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    net_w = net_h = 480
    resize_size = 520

    semi_model = get_model(args)

    # finetune_model.to(device)
    semi_model.to(device)
    semi_model, model_name, top_k, num_classes, background_ignore_index = configuration(semi_model, args)

    valid_dataset = ETRI(
        split = "val", 
        img_paths = True,
        trans_label = False, 
        transform=A.Compose([
            A.Resize(net_w, net_h),        
        ]),
        root = args.source_root
    )

    valid_dataloader =  data.DataLoader(valid_dataset, batch_size=args.batch_size)

    experiment_name = args.experiment_name
    evaluator = Eval(num_classes, background_ignore_index)

    save_dir = args.save_path+"{}/{}/{}/".format("test", experiment_name, "top{}".format(top_k))
    os.makedirs(save_dir, exist_ok=True)

    blend = False
    contain_pre = True

    for idx, (name, image_path, x, y) in enumerate(tqdm(valid_dataloader)):

        with torch.no_grad():
            semi_model.eval()

            img_name = name[0]
            img_path = image_path[0]

            print("processing {} ({}/{})".format(img_name, idx + 1, len(valid_dataloader)))

            real_img = read_image(img_path)

            x = x.to(device)
            y = y.to(device)

            semi_semantic_segmentation = semi_model(x)
            
            semi_semantic_segmentation = torch.nn.functional.interpolate(
                semi_semantic_segmentation, size=real_img.shape[:2], mode="bicubic", align_corners=False
            )
            semi_semantic_segmentation = torch.argmax(semi_semantic_segmentation, dim=1).cpu().numpy()
            
            segmentation_maps = label_mapper(y.cpu().numpy(), top_k, "etri_merge_top6")
            filename = os.path.join(save_dir, img_name+".png")


            if blend == True:
                write_segm_img(filename, real_img, semantic_segmentation, top_k = top_k, alpha=0.5)
            else:
                if contain_pre == True:
                    fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, figsize=(8,6))

                    ax1.imshow(real_img)
                    ax1.axis("off")

                    gt = get_mask_pallete(segmentation_maps, "segmento", top_k)
                    gt = gt.convert("RGBA")
                    ax2.imshow(gt)
                    ax2.axis("off")

                    # finetune = get_mask_pallete(finetune_semantic_segmentation, "segmento", top_k)
                    # finetune = finetune.convert("RGBA")
                    # ax3.imshow(finetune)
                    # ax3.axis("off")
                    
                    semi = get_mask_pallete(semi_semantic_segmentation, "segmento", top_k)
                    semi = semi.convert("RGBA")
                    ax3.imshow(semi)
                    ax3.axis("off")

                    # filename = os.path.join(
                    #     save_dir, os.path.splitext(os.path.basename(img_name))[0]
                    # )

                    # plt.savefig(filename, dpi = 800)
                    # plt.close()

            evaluator.add_batch(segmentation_maps, semi_semantic_segmentation)

    # mIOU
    miou  = evaluator.Mean_Intersection_over_Union()
    classes_miou = evaluator.classMIoU()
    #fwiou
    fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()
    # mp
    mp = evaluator.Mean_Precision()
    # pa
    pa = evaluator.Pixel_Accuracy()
    # accuracy
    mpa = evaluator.Mean_Pixel_Accuracy()

    print("---------------------------------------------------")
    print("Mean Intersection over Union: {:4f}".format(miou))
    print("Mean Precision: {:4f}".format(mp))
    print("Pixel Accuracy: {:4f}".format(pa))
    print("Mean Pixel Accuracy: {:4f}".format(mpa))
    print("Frequently Weighted MIoU: {:4f}".format(fwiou))
    print("---------------------------------------------------", end="\n\n")

    print("CLASSES MIOU EVAL---------------------------------------------------")
    print("Pixmatch: {}".format(classes_miou))
    print("---------------------------------------------------", end="\n\n")

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model configuration
    parser.add_argument("--save_path", type=str, default=None, help="train model saving path")
    parser.add_argument("--weights", default=None, help="path to the trained weights of model")
    parser.add_argument("--backbone", default =None)
    parser.add_argument("--source_root", type = str)
    # parser.add_argument("--finetune_weights", default=None, help="path to the trained weights of model")
    parser.add_argument("--model_name", default="SegmentoDPT-6", help="model name")
    parser.add_argument("--top_k", type=int, default = 6, help="Number of classes")

    # label merged
    parser.add_argument("--merged", type = str, default = "", help = "label merge category")

    # train configuration
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--experiment_name", type=str, default="Segmento", help="experiment name")
    parser.add_argument("--ignore_index", type=int, default=-1, help="ignore_index")

    args = parser.parse_args()

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(args)
