import os, sys, glob, cv2, argparse
sys.path.append("/home/cvlab02/project/segmento")
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

import albumentations as A


from segmento_dpt import SegmentoDPT
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from utils.label import configuration, label_mapper, get_labelnames
from utils.args import str2bool
from utils.eval import Eval, AverageMeter, intersectionAndUnionGPU
from utils.io import read_image, write_segm_img
from utils.pallete import get_mask_pallete

from datasets.etri import ETRI


def run(args):
    
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    net_w = net_h = 480
    resize_size = 520

    # load network
    model = SegmentoDPT(
        num_classes = args.top_k,
        weights=args.weights, 
    )
    model.to(device)
    model, model_name, top_k, num_classes, background_ignore_index = configuration(model, args)

    # dataset and dataloader

    valid_dataset = ETRI(
        split = "val",
        img_paths = True,
        trans_label = False,
        transform=A.Compose([
            A.Resize(net_w, net_h),
            A.Normalize(),
        ])
    )
    
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=args.batch_size)

    # experiment name
    # save_path "/media/dataset2/segmento/dpt/test/" 

    experiment_name = "hj" + args.experiment_name

    evaluator = Eval(num_classes, background_ignore_index)

    save_dir = args.save_path+"{}/{}/".format(experiment_name, "top{}".format(top_k))
    os.makedirs(save_dir, exist_ok=True)

    blend = False
    
    
    for idx, (img_name, img_path, x, y) in enumerate(tqdm(valid_dataloader)):

        with torch.no_grad():

            model.eval()

            img_name = img_name[0]
            img_path = img_path[0]

            print("processing {} ({}/{})".format(img_name, idx + 1, len(valid_dataloader)))
            
            real_img = read_image(img_path)
            
            x = x.to(device)
            y = y.to(device)

            semantic_segmentation = model(x)
            semantic_segmentation = torch.nn.functional.interpolate(
                semantic_segmentation, size=real_img.shape[:2], mode="bicubic", align_corners=False
            )
            semantic_segmentation = torch.argmax(semantic_segmentation, dim=1).cpu().numpy()
            
            segmentation_maps = label_mapper(y.cpu().numpy(), top_k)

            filename = os.path.join(
                save_dir, os.path.splitext(os.path.basename(img_name))[0]
            )

            if blend == True:
                write_segm_img(filename, real_img, semantic_segmentation, top_k = top_k, alpha=0.5)
            else:
                fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, figsize=(8,6))

                ax1.imshow(real_img)
                ax1.axis("off")

                gt = get_mask_pallete(segmentation_maps, "segmento", top_k)
                gt = gt.convert("RGBA")
                ax2.imshow(gt)
                ax2.axis("off")
                
                segmento = get_mask_pallete(semantic_segmentation, "segmento", top_k)
                segmento = segmento.convert("RGBA")
                ax3.imshow(segmento)
                ax3.axis("off")

                plt.savefig(filename, dpi = 550)
                plt.close()
            
            # Evaluation
            evaluator.add_batch(segmentation_maps, semantic_segmentation)
            
    # mIOU
    miou, classes_miou = evaluator.Mean_Intersection_over_Union()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model configuration
    parser.add_argument("--save_path", type=str, default=None, help="train model saving path")
    parser.add_argument("--weights", default=None, help="path to the trained weights of model")
    parser.add_argument("--model_name", default="SegmentoDPT-6", help="model name")
    parser.add_argument("--top_k", type=int, default = 6, help="Number of classes")

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