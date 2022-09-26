import os, sys, glob, cv2, argparse
sys.path.append("/home/cvlab02/project/etri")
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision.transforms import Compose

import albumentations as A


from etri_dpt import etriDPT
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from utils.label import configuration, label_mapper, get_labelnames
from utils.args import str2bool
from utils.eval import Eval
from utils.poly import poly_lr_scheduler

from datasets.etri import ETRI


def run(args):
    
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    net_w = net_h = 480
    resize_size = 520

    # load network
    model = etriDPT(
        num_classes = args.top_k,
        weights=args.weights, 
    )
    model.to(device)
    model, model_name, top_k, num_classes, background_ignore_index = configuration(model, args)

    # dataset and dataloader

    ## Inference: different shape between w, h
    ## Train / Valid: Same size on w, h since anno exists
    # def dpt_resize(net_w, net_h):
    #     return Resize(
    #         net_w,
    #         net_h,
    #         resize_target=None,
    #         keep_aspect_ratio=True,
    #         ensure_multiple_of=32,
    #         resize_method="minimal",
    #         image_interpolation_method=cv2.INTER_CUBIC,
    #     )

    train_dataset = ETRI(
        split = "train", 
        transform=A.Compose([
            A.LongestMaxSize(resize_size),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(net_h, net_w, ratio=(0.5, 2.0)),
        ])
    )
    valid_dataset = ETRI(
        split = "val", 
        transform=A.Compose([
            A.Resize(net_w, net_h),
        ])
    )
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=args.batch_size)

    # experiment name
    experiment_name = "hj" + args.experiment_name

    evaluator = Eval(num_classes, background_ignore_index)

    cross_entropy = nn.CrossEntropyLoss(ignore_index = args.ignore_index)

    optimizer = torch.optim.SGD(model.parameters(), lr= args.lr, weight_decay=1e-6, momentum=0.9)
    
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
    
    save_dir = args.save_path+"{}/{}/".format(experiment_name, "top{}".format(top_k))
    os.makedirs(save_dir, exist_ok=True)
    
    initial_epoch = int(args.weights.split("::")[1]) if args.weights != "" else 0

    best_MIou = 0

    for epoch in tqdm(range(initial_epoch + 200)):

        epoch = initial_epoch + epoch
        epoch_loss = 0 
        
        model.train()

        for batch_idx, (x, y) in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}")):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            prediction = model(x)

            segmentation_maps = y.cpu().numpy()
            segmentation_maps = label_mapper(segmentation_maps, top_k)
            segmentation_maps = torch.LongTensor(segmentation_maps).to(device)
            
            _loss = cross_entropy(prediction, segmentation_maps)
            _loss.backward()
            
            optimizer.step()
            epoch_loss += _loss.item()

            poly_lr_scheduler(args, optimizer)
            print("Learning Rate is: {}".format(optimizer.param_groups[0]["lr"]), end="\n\n")

        with torch.no_grad():

            model.eval()

            evaluator.reset()

            for val_idx, (x, y) in enumerate(tqdm(valid_dataloader, desc=f"Valid Epoch {epoch + 1}")):
                
                x = x.to(device)
                y = y.to(device)
                
                out = model(x)

                semantic_segmentation = torch.argmax(out, dim=1)
                semantic_segmentation = semantic_segmentation.cpu().numpy()

                segmentation_maps = y.cpu().numpy()
                segmentation_maps = label_mapper(segmentation_maps, top_k)

                evaluator.add_batch(segmentation_maps, semantic_segmentation)

        # mIOU
        epoch_miou, classes_miou = evaluator.Mean_Intersection_over_Union()
        #fwiou
        epoch_fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()
        # mp
        epoch_mp = evaluator.Mean_Precision()
        # pa
        epoch_pa = evaluator.Pixel_Accuracy()
        # accuracy
        epoch_mpa = evaluator.Mean_Pixel_Accuracy()

        print("---------------------------------------------------")
        print("Evaluation Results of Epoch: {}".format(epoch), end="\n\n")
        print("Mean Intersection over Union: {:4f}".format(epoch_miou))
        print("Mean Precision: {:4f}".format(epoch_mp))
        print("Pixel Accuracy: {:4f}".format(epoch_pa))
        print("Mean Pixel Accuracy: {:4f}".format(epoch_mpa))
        print("Frequently Weighted MIoU: {:4f}".format(epoch_fwiou))
        print("---------------------------------------------------", end="\n\n")
        

        # Save checkpoint if new best model

        is_best = epoch_miou > best_MIou
        baseline = 1 / (top_k + 1)
        if is_best:
            best_MIou = epoch_miou
            best_epoch = epoch
            print("Baseline MIoU: {:3f}, improved {:3f}".format(baseline, epoch_miou - baseline))
            print("=====> Epoch {} - MIoU: {:3f} MPA: {:3f}".format(epoch, epoch_miou, epoch_mpa))
            print("=====> Saving a new best checkpoint...")
            print("=====> The best val MIoU is now {:.3f} from epoch {}".format(best_MIou, best_epoch))
            
            filename = os.path.join(save_dir, 'Epoch::{}:: | Model: {}  MIoU: {:.3f} | MPA: {:3f}.pth'.format(best_epoch, args.model_name, best_MIou, epoch_mpa))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_MIou': best_MIou,
                'epoch_mpa': epoch_mpa,
                'epoch_classes_miou': classes_miou,
                'epoch_fwiou': epoch_fwiou,
                'epoch_mp': epoch_mp,
                'epoch_pa': epoch_pa,
            }, filename)
        else:
            print("Baseline MIoU: {:3f}, improved {:3f}".format(baseline, epoch_miou - baseline))
            print("=====> Epoch {} - MIoU: {:3f} MPA: {:3f}".format(epoch, epoch_miou, epoch_mpa))
            print("=====> The MIoU of val did not improve.")
            print("=====> The best val MIoU is still {:.3f} from epoch {:3f}".format(best_MIou, best_epoch))
        epoch += 1

    # Save final checkpoint

    print("=====> The best MIoU was {:.3f} at epoch {}".format(best_MIou, best_epoch))
    print("=====> Saving the final checkpoint")
    filename = os.path.join(save_dir, 'Epoch::{}:: | Model: {}  MIoU: {:.3f} | MPA: {:3f}.pth'.format(best_epoch, args.model_name, best_MIou, epoch_mpa))
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_MIou': best_MIou,
        'epoch_mpa': epoch_mpa,
        'epoch_classes_miou': classes_miou,
        'epoch_fwiou': epoch_fwiou,
        'epoch_mp': epoch_mp,
        'epoch_pa': epoch_pa,
    }, filename)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model configuration
    parser.add_argument("--save_path", type=str, default=None, help="train model saving path")
    parser.add_argument("--weights", default=None, help="path to the trained weights of model")
    parser.add_argument("--model_name", default="etriDPT-6", help="model name")
    parser.add_argument("--top_k", type=int, default = 6, help="Number of classes")

    # train configuration
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--experiment_name", type=str, default="etri", help="experiment name")
    parser.add_argument("--ignore_index", type=int, default=-1, help="ignore_index")

    args = parser.parse_args()

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(args)