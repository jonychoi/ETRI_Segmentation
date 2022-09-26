from segmento.framework.perturbations.augmentations import augment, get_augmentation
from segmento.framework.perturbations.fourier import fourier_mix
from segmento.framework.perturbations.cutmix import cutmix_combine

from utils.label import label_mapper

import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper
def unpack(x):
    return (x[0], x[1]) if isinstance(x, tuple) else (x, None)

#########################
# Source supervised loss #
##########################

def segmento_loss(model, ema, iter, optimizer, losses, loss, batch_s, batch_t, args, writer):
    
    source_sup_loss(model, losses, loss, batch_s, args)
    target_pseudolabel(model, losses, loss, batch_s, batch_t, args)

    # Step optimizer if accumulated enough gradients
    optimizer.step()
    optimizer.zero_grad()

    # Update model EMA parameters each step
    ema.update_params()

    # Calculate total loss
    total_loss = sum(losses.values())

    # Log main losses
    for name, loss in losses.items():
        writer.add_scalar(f'train/{name}', loss, iter)

def source_sup_loss(model, losses, loss, batch_s, args):
    # args.source_fourier
    # args.aux
    # args.fourier_beta
    # args.lam_aux
    x, y = batch_s
    
    if True:  # For VS Code collapsing

        # Data
        x = x.to(args.device)
        y = y.squeeze(dim=1).to(device=args.device, dtype=torch.long, non_blocking=True).cpu().numpy()
        y= torch.LongTensor(label_mapper(y, args.top_k)).to(args.device)
        # Fourier mix: source --> target
        if args.source_fourier:
            x = fourier_mix(src_images=x, tgt_images=batch_t[0].to(device), L=args.fourier_beta)

        # Forward
        pred = model(x)
        pred_1, pred_2 = unpack(pred)

        # Loss (source)
        loss_source_1 = loss(pred_1, y)
        if args.aux:
            loss_source_2 = loss(pred_2, y) * args.lam_aux
            loss_source = loss_source_1 + loss_source_2
        else:
            loss_source = loss_source_1

        # Backward
        loss_source.backward()

        # Clean up
        losses['source_main'] = loss_source_1.cpu().item()
        if args.aux:
            losses['source_aux'] = loss_source_2.cpu().item()
            del x, y, loss_source, loss_source_1, loss_source_2
        else:
            del x, y, loss_source, loss_source_1


######################
# Target Pseudolabel #
######################

def target_pseudolabel(model, losses, loss, batch_s, batch_t, args):
    # args.pseudolabel_threshold
    x, _ = batch_t
    x = x.to(args.device)

    # First step: run non-augmented image though model to get predictions
    with torch.no_grad():

        # Substep 1: forward pass
        pred = model(x.to(args.device))
        pred_1, pred_2 = unpack(pred)

        # Substep 2: convert soft predictions to hard predictions
        pred_P_1 = F.softmax(pred_1, dim=1)
        label_1 = torch.argmax(pred_P_1.detach(), dim=1)

        maxpred_1, argpred_1 = torch.max(pred_P_1.detach(), dim=1)

        T = args.pseudolabel_threshold

        mask_1 = (maxpred_1 > T)

        ignore_tensor = torch.ones(1).to(args.device, dtype=torch.long) * args.ignore_index

        label_1 = torch.where(mask_1, label_1, ignore_tensor)

        if args.aux:
            pred_P_2 = F.softmax(pred_2, dim=1)
            maxpred_2, argpred_2 = torch.max(pred_P_2.detach(), dim=1)

            pred_c = (pred_P_1 + pred_P_2) / 2
            maxpred_c, argpred_c = torch.max(pred_c, dim=1)
            
            mask = (maxpred_1 > T) | (maxpred_2 > T)
            label_2 = torch.where(mask, argpred_c, ignore_tensor)
    
    if args.aux:
        aug_loss(model, losses, loss, args, x, label_1, label_2)
        fourier_loss(model, losses, loss, args, batch_s, x, label_1, label_2)
        cutmix_loss(model, losses, loss, args, x, batch_s, label_1)
    else:
        aug_loss(model, losses, loss, args, x, label_1)
        fourier_loss(model, losses, loss, args, batch_s, x, label_1)
        cutmix_loss(model, losses, loss, args, x, batch_s, label_1)
        

############
# Aug loss #
############

def aug_loss(model, losses, loss, args, x, label_1, label_2 = None):
    # args.lam_aug
    # args.aux
    
    # if args.lam_aug > 0:
    #     aug = get_augmentation()

    if args.lam_aug > 0:
        aug = get_augmentation()
        # Second step: augment image and label
        x_aug, y_aug_1 = augment(images=x.cpu(), labels=label_1.detach().cpu(), aug=aug)
        x_aug = x_aug.type(torch.FloatTensor)
        y_aug_1 = y_aug_1.to(device=args.device, non_blocking=True)
        if args.aux:
            _, y_aug_2 = augment(images=x.cpu(), labels=label_2.detach().cpu(), aug=aug)
            y_aug_2 = y_aug_2.to(device=args.device, non_blocking=True)

        # Third step: run augmented image through model to get predictions
        pred_aug = model(x_aug.to(args.device))
        pred_aug_1, pred_aug_2 = unpack(pred_aug)

        # Fourth step: calculate loss
        loss_aug_1 = loss(pred_aug_1, y_aug_1) * args.lam_aug
        if args.aux:
            loss_aug_2 = loss(pred_aug_2, y_aug_2) * args.lam_aug * args.lam_aux
            loss_aug = loss_aug_1 + loss_aug_2
        else:
            loss_aug = loss_aug_1

        # Backward
        loss_aug.backward()

        # Clean up
        losses['aug_main'] = loss_aug_1.cpu().item()
        if args.aux:
            losses['aug_aux'] = loss_aug_2.cpu().item()
            del pred_aug, pred_aug_1, pred_aug_2, loss_aug, loss_aug_1, loss_aug_2
        else:
            del pred_aug, pred_aug_1, pred_aug_2, loss_aug, loss_aug_1
################
# Fourier Loss #
################

def fourier_loss(model, losses, loss, args, batch_s, x, label_1, label_2 = None):
    if args.lam_fourier > 0:

        # Second step: fourier mix
        x_fourier = fourier_mix(
            src_images=x.to(args.device),
            tgt_images=batch_s[0].to(args.device),
            L=args.fourier_beta)

        # Third step: run mixed image through model to get predictions
        pred_fourier = model(x_fourier.to(args.device))
        pred_fourier_1, pred_fourier_2 = unpack(pred_fourier)

        # Fourth step: calculate loss
        loss_fourier_1 = loss(pred_fourier_1, label_1) * args.lam_fourier

        if args.aux:
            loss_fourier_2 = loss(pred_fourier_2, label_2) * args.lam_fourier * args.lam_aux
            loss_fourier = loss_fourier_1 + loss_fourier_2
        else:
            loss_fourier = loss_fourier_1

        # Backward
        loss_fourier.backward()

        # Clean up
        losses['fourier_main'] = loss_fourier_1.cpu().item()
        if args.aux:
            losses['fourier_aux'] = loss_fourier_2.cpu().item()
            del pred_fourier, pred_fourier_1, pred_fourier_2, loss_fourier, loss_fourier_1, loss_fourier_2
        else:
            del pred_fourier, pred_fourier_1, pred_fourier_2, loss_fourier, loss_fourier_1

###############
# CutMix Loss #
###############

def cutmix_loss(model, losses, loss, args, x, batch_s, label_1):  
    if args.lam_cutmix > 0:

        # Second step: CutMix
        label_2 = batch_s[1].unsqueeze(dim=1).to(args.device, dtype=torch.long).cpu().numpy()
        label_2 = torch.LongTensor(label_mapper(label_2, args.top_k)).to(args.device)
        x_cutmix, y_cutmix = cutmix_combine(
            images_1=x,
            labels_1=label_1.unsqueeze(dim=1),
            images_2=batch_s[0].to(args.device),
            labels_2=label_2)
        y_cutmix = y_cutmix.squeeze(dim=1)

        # Third step: run mixed image through model to get predictions
        pred_cutmix = model(x_cutmix)
        pred_cutmix_1, pred_cutmix_2 = unpack(pred_cutmix)

        # Fourth step: calculate loss
        loss_cutmix_1 = loss(pred_cutmix_1, y_cutmix) * args.lam_cutmix
        if args.aux:
            loss_cutmix_2 = loss(pred_cutmix_2, y_cutmix) * args.lam_cutmix * args.lam_aux
            loss_cutmix = loss_cutmix_1 + loss_cutmix_2
        else:
            loss_cutmix = loss_cutmix_1

        # Backward
        loss_cutmix.backward()

        # Clean up
        losses['cutmix_main'] = loss_cutmix_1.cpu().item()
        if args.aux:
            losses['cutmix_aux'] = loss_cutmix_2.cpu().item()
            del pred_cutmix, pred_cutmix_1, pred_cutmix_2, loss_cutmix, loss_cutmix_1, loss_cutmix_2
        else:
            del pred_cutmix, pred_cutmix_1, pred_cutmix_2, loss_cutmix, loss_cutmix_1