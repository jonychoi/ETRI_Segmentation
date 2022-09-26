import os, sys, glob, cv2, argparse
sys.path.append("/home/cvlab02/project/etri")
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

import albumentations as A 

from utils.ema import EMA
from utils.label_final import configuration, label_mapper, get_labelnames
from utils.args import str2bool
from utils.eval import Eval
from utils.poly import poly_lr_scheduler
from utils.save_checkpoints import save_checkpoints
from utils.get_model import get_model

from perturbations.augmentations import augment, get_augmentation, get_geo_augmentation
import perturbations.randaug as randaug

from style_transfer.styletf import style_transferred

from datasets.etri import ETRI 
from datasets.dram.dram import DramDataSet

def run(args):

    print("initialize")

    random.seed(2022)
    np.random.seed(1998)
    torch.random.manual_seed(1995)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : {}".format(device))

    net_w = net_h = 480
    resize_size = 520 

    model = get_model(args)
    model.to(device)    
    ema = EMA(model, args.ema_decay)

    model, model_name, top_k, num_classes, background_ignore_index = configuration(model, args)
    
    writer = SummaryWriter('/tensorboard/' + args.experiment_name + "/" + args.model_name)

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_MEAN = torch.Tensor(IMG_MEAN).reshape(1,3,1,1)
    IMG_STD = [0.229, 0.224, 0.225]
    IMG_STD = torch.Tensor(IMG_STD).reshape(1,3,1,1)

    if args.target_dataset == "DRAM":
        tgt_train_dataset = DramDataSet(
            split = "train", 
            transform = A.Compose([
                A.LongestMaxSize(resize_size),
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(net_h, net_w, ratio=(0.5, 2.0)),
            ]),
            root = args.source_root
        )
        tgt_test_dataset = DramDataSet(
            split = "test", 
            transform=A.Compose([
                A.Resize(net_w, net_h),
            ]),
            root = args.source_root
        )

        tgt_train_dataloader = data.DataLoader(tgt_train_dataset, batch_size=args.batch_size, shuffle=True)
        tgt_test_dataloader = data.DataLoader(tgt_test_dataset, batch_size = args.batch_size)

    if args.source_dataset == "etri":
        train_dataset = ETRI(
            split = "train", 
            transform=A.Compose([
                A.LongestMaxSize(resize_size),
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(net_h, net_w, ratio=(0.5, 2.0)),
            ]),
            data_root = args.target_root
        )
        valid_dataset = ETRI(
            split = "val", 
            img_paths = True,
            transform=A.Compose([
                A.Resize(net_w, net_h),
            ]),
            data_root = args.target_root
        )

        src_train_dataloader = data.DataLoader(src_train_dataset, batch_size=args.batch_size, shuffle=True)
        src_valid_dataloader = data.DataLoader(src_val_dataset, batch_size = args.batch_size)

    # exp name
    experiment_name = args.experiment_name 

    # evaluator 
    evaluator = Eval(num_classes, background_ignore_index)

    # define loss 
    sup_cross_entropy = nn.CrossEntropyLoss(ignore_index = -1)
    semi_cross_entropy = nn.CrossEntropyLoss(ignore_index = -1)

    # define optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr= args.init_lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), weight_decay = args.weight_decay)        
    
    # makedirs for checkpoints & pics
    save_dir = args.save_path+"{}/{}/source/".format(experiment_name, "top{}".format(top_k))
    os.makedirs(save_dir, exist_ok=True)

    save_dir = args.save_path+"{}/{}/".format(experiment_name, "top{}".format(top_k))

    initial_epoch = target_best_epoch = source_best_epoch = int(args.weights.split("::")[1]) if args.weights != "" else 0

    source_best_MIou = 0

    total = min(len(src_train_dataloader), len(tgt_train_dataloader))

    print(len(src_train_dataloader), len(tgt_train_dataloader))
    
    for epoch_idx, epoch in enumerate(tqdm(range(initial_epoch + 100))):
        
        epoch = initial_epoch + epoch 

        model.train()

        for batch_idx, (batch_s, batch_t) in enumerate(tqdm(zip(src_train_dataloader, tgt_train_dataloader), total = total, desc=f"Train Epoch {epoch + 1}" )):
            iter = epoch_idx * total + batch_idx

            # REARRANGE LEARNING RATE BY EVERY ITERS
            poly_lr_scheduler(args, iter, optimizer)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], iter)

            losses = {}
            
            """ sup loss of PASCAL """
            x_s, y_s = batch_s
            x_t, _ = batch_t
            x_t_c = x_t.to(device)
            
            sup_seg_maps = label_mapper(y_s.numpy(), top_k, args.merged)
            sup_seg_map = torch.LongTensor(sup_seg_maps).to(device)

            x_s_sup = x_s.to(device)
            prediction = model(x_s_sup)
            
            supervised_loss = sup_cross_entropy(prediction, sup_seg_map)
            supervised_loss.backward()
            
            losses['Source Loss'] = supervised_loss.cpu().item()

            """ style sup loss with PASCAL"""
            if args.lam_style > 0:
                x_s_st = x_s.to(device)
                x_t_st = x_t.to(device)
                
                s_weight = random.random()
                x_stylized = style_transferred(args, x_s_st, x_t_st, s_weight)  # contents: x_s_st, style : x_t_st
                style_seg_maps = torch.LongTensor(sup_seg_maps).to(device)

                stylized_pred = model(x_stylized)
        
                style_loss =  semi_cross_entropy(stylized_pred, style_seg_maps) * args.lam_style 
                style_loss.backward()

                losses['Style Loss'] = style_loss.cpu().item() 


            # pseudolabel for tgt(DRAM) consistency loss

            with torch.no_grad():
                model.eval()
                pred = model(x_t_c)
                pred = F.softmax(pred, dim = 1)
                x_t_pred = pred.detach()
                maxpred, argpred = torch.max(x_t_pred, dim = 1)

                T = args.pseudolabel_threshold
                mask = maxpred > T
                ignore_tensor = torch.ones(1).to(device, dtype = torch.long) * args.ignore_index

                pseudo_label = torch.argmax(pred.detach(), dim = 1)
                pseudo_label = torch.where(mask, pseudo_label, ignore_tensor)
            
            model.train()

            '''consistency regularization with DRAM : rand aug  '''                 
            if args.lam_randaug > 0:

                x_randauged = randaug.randomAug(x_t_c, 3, args.batch_size)
                y_randauged = pseudo_label.detach().cpu()

                augmentation = get_geo_augmentation(args.top_k, net_h, net_w)

                x_strongauged, y_strongauged = augment(
                    images = x_randauged, 
                    labels = y_randauged, 
                    aug= augmentation
                )      

                x_strongauged = x_strongauged.to(device).float()
                y_strongauged = y_strongauged.to(device)
                
                augmented_pred = model(x_strongauged)

                randaug_loss = semi_cross_entropy(augmented_pred, y_strongauged) * args.lam_randaug

                randaug_loss.backward()
                losses['target loss : randaug'] = randaug_loss.cpu().item()
            
            '''consistency regularization with DRAM : style aug '''
            if args.lam_styleaug > 0 :

                s_weight = random.uniform(0.5, 1.0)
                augmentation = get_geo_augmentation(args.top_k, net_h, net_w)

                '''-- photometric aug --'''
                contents = x_t_c
                styles = torch.flip(x_t_c, dims = (0,))
                x_styleauged = style_transferred(args, contents, styles, s_weight).cpu()
                y_styleauged = pseudo_label.detach().cpu()
                ''' -------------------'''

                '''-- geometric aug  --''' 
                x_style_geo, y_style_geo = augment(
                    images = x_styleauged, 
                    labels = y_styleauged, 
                    aug= augmentation
                )      
                ''' -------------------'''
            
                x_style_geo = x_style_geo.to(device).float()
                y_style_geo = y_style_geo.to(device)

                styleauged_pred = model(x_style_geo)

                styleaug_loss = semi_cross_entropy(styleauged_pred, y_style_geo) * args.lam_styleaug

                styleaug_loss.backward() 
                losses["target loss : styleaug"] = styleaug_loss.cpu().item()
           
            optimizer.step()
            optimizer.zero_grad()

            ema.update_params()

            total_loss = sum(losses.values())
            for name, loss in losses.items():
                writer.add_scalar("train/{}".format(name), loss, iter)
            writer.add_scalar("train/total_loss", total_loss, iter)  

        ema.update_buffer()

        # Use EMA params to evaluate performance
        ema.apply_shadow()
        ema.model.eval()
        ema.model.cuda()
        # validation  :  source 
        source_best_epoch, source_best_MIou = validate('source', initial_epoch, source_best_epoch, epoch, evaluator, model, ema, src_valid_dataloader, device, source_best_MIou, top_k, writer, save_dir, model_name, optimizer)

        ema.restore()
        epoch = epoch + 1


def validate(
    mode, 
    initial_epoch, 
    best_epoch,
    epoch,
    evaluator,
    model, 
    ema,
    valid_dataloader,
    device, 
    best_MIou, 
    top_k,
    writer, 
    save_dir, 
    model_name, 
    optimizer
):
    with torch.no_grad():
        model.eval()
        evaluator.reset()

        for idx, (x, y) in enumerate(tqdm(
            valid_dataloader, total = len(valid_dataloader), desc="Valid Epoch {}".format(epoch + 1)
        )) :
            x = x.to(device)
            y = y.to(device)

            semantic_segmentation = model(x)
            semantic_segmentation = torch.argmax(semantic_segmentation, dim = 1).cpu().numpy()

            segmentation_maps = label_mapper(y.cpu().numpy(), args.top_k, args.merged).astype(np.int64)

            evaluator.add_batch(semantic_segmentation, segmentation_maps)

    epoch_miou, classes_miou, epoch_fwiou, epoch_mp, epoch_pa, epoch_mpa = evaluator.evaluate_results()

    writer.add_scalar('{}/MIoU: Mean Intersection over Union'.format(mode), epoch_miou, epoch)
    writer.add_scalar('{}/PA: Pixel Accuracy'.format(mode), epoch_pa, epoch)
    writer.add_scalar('{}/MP: Mean Precision'.format(mode), epoch_mp, epoch)
    writer.add_scalar('{}/MPA: Mean Pixel Accuracy'.format(mode), epoch_mpa, epoch)
    writer.add_scalar('{}/FWIoU: Frequently Weighted IoU'.format(mode), epoch_fwiou, epoch)

    is_best = epoch_miou > best_MIou
    baseline = 1 / (top_k + 1)

    print(is_best, epoch_miou, best_MIou)

    if is_best :
        best_MIou = epoch_miou 
        best_epoch = epoch
        print("------{}------".format(mode))
        print("Baseline MIoU: {:3f}, improved {:3f}".format(baseline, epoch_miou - baseline))
        print("=====> Epoch {} - MIoU: {:3f} MPA: {:3f}".format(epoch, epoch_miou, epoch_mpa))
        print("=====> Saving a new best checkpoint...")
        print("=====> The best val MIoU is now {:.3f} from epoch {}".format(best_MIou, best_epoch))

    else:
        best_epoch = initial_epoch if best_epoch is None else best_epoch
        print("------{}------".format(mode))
        print("Baseline MIoU: {:3f}, improved {:3f}".format(baseline, epoch_miou - baseline))
        print("=====> Epoch {} - MIoU: {:3f} MPA: {:3f}".format(epoch, epoch_miou, epoch_mpa))
        print("=====> The MIoU of val did not improve.")
        print("=====> The best val MIoU is still {:.3f} from epoch {:3f}".format(best_MIou, best_epoch))


    return best_epoch, best_MIou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model configuration
    parser.add_argument("--backbone", type=str, default="dpt-hybrid", help="model backbone")
    parser.add_argument("--save_path", type=str, default=None, help="train model saving path")
    parser.add_argument("--weights", default=None, help="path to the trained weights of model")
    parser.add_argument("--model_name", default="etriDPT-6", help="model name")
    parser.add_argument("--top_k", type=int, default = 6, help="Number of classes")

    # train configuration
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--experiment_name", type=str, default="etri", help="experiment name")
    parser.add_argument("--ignore_index", type=int, default=-1, help="ignore_index")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")

    # learning rate
    parser.add_argument("--init_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--lr_max_iter", type=int, default=20000, help="max iteration")
    parser.add_argument("--poly_power", type=float, default=0.9, help="poly power")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
 
    # ema decay
    parser.add_argument("--ema_decay", type=float, default=0.999, help="ema decay")

    # dataset
    parser.add_argument("--source_dataset", type=str, default='etri', help="source dataset")
    parser.add_argument("--source_root", type=str, default="/media/dataset2/etri", help="source root")

    parser.add_argument("--target_dataset", type=str, default='DRAM', help="target dataset")
    parser.add_argument("--target_root", type=str, default='/media/dataset2/DRAM_processed', help="target root")

    # pretrained
    parser.add_argument("--pretrained", type=str, default='', help="pretrained root")
    
    # losses
    parser.add_argument("--pseudolabel_threshold", type=float, default=0.0, help="pseudolabel_threshold")
    parser.add_argument("--lam_sup", type = float, default = 1.0)
    parser.add_argument("--lam_randaug", type=float, default = 0.0, help = "lam rand aug")
    parser.add_argument("--lam_style", type = float, default = 0.0, help = "lam style transfer")
    parser.add_argument("--lam_styleaug", type = float, default = 0.0, help = "lam style augmentation")

    # label merge
    parser.add_argument("--merged", type = str, default = "", help = "label merge category")
    args = parser.parse_args()

    # style_transfer 
    
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(args)