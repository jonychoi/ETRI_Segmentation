from utils.pallete import get_mask_pallete
from utils.save_checkpoints import save_checkpoints
from utils.io import read_image

def tf_logger(
    img_path,
    img_name,
    model,
    ema,
    semantic_segmentation, 
    segmentation_maps, 
    model_name,
    best_MIou, 
    top_k, 
    evaluator, 
    epoch, 
    writer,
    save_dir
):
    epoch_miou, classes_miou, epoch_fwiou, epoch_mp, epoch_pa, epoch_mpa = evaluator.evaluate_results()
    
    # # Show Images
    # vis_imgs = 2

    # real_img = read_image(img_path)

    # gt = get_mask_pallete(segmentation_maps, "segmento", top_k)
    # gt = gt.convert("RGBA")
    
    # segmento = get_mask_pallete(semantic_segmentation, "segmento", top_k)
    # segmento = segmento.convert("RGBA")

    # for index, (img, lab, pred) in enumerate(zip(real_img, gt, segmento)):
    #     writer.add_image(str(index) + '/images', img, epoch)
    #     writer.add_image(str(index) + '/labels', lab, epoch)
    #     writer.add_image(str(index) + '/preds', pred, epoch)

    # Show graphs
    writer.add_scalar('MIoU: Mean Intersection over Union', epoch_miou, epoch)
    writer.add_scalar('PA: Pixel Accuracy', epoch_pa, epoch)
    writer.add_scalar('MP: Mean Precision', epoch_mp, epoch)
    writer.add_scalar('MPA: Mean Pixel Accuracy', epoch_mpa, epoch)
    writer.add_scalar('FWIoU: Frequently Weighted IoU', epoch_fwiou, epoch)

    is_best = epoch_miou > best_MIou
    baseline = 1 / (top_k + 1)

    # Save checkpoint if new best model
    if is_best:
        best_MIou = epoch_miou
        best_epoch = epoch
        print("Baseline MIoU: {:3f}, improved {:3f}".format(baseline, epoch_miou - baseline))
        print("=====> Epoch {} - MIoU: {:3f} MPA: {:3f}".format(epoch, epoch_miou, epoch_mpa))
        print("=====> Saving a new best checkpoint...")
        print("=====> The best val MIoU is now {:.3f} from epoch {}".format(best_MIou, best_epoch))
        
        save_checkpoints(save_dir, model_name, ema, epoch, optimizer, best_MIoU, epoch_mpa, classes_miou, epoch_fwiou, epoch_mp, epoch_pa)
    else:
        print("Baseline MIoU: {:3f}, improved {:3f}".format(baseline, epoch_miou - baseline))
        print("=====> Epoch {} - MIoU: {:3f} MPA: {:3f}".format(epoch, epoch_miou, epoch_mpa))
        print("=====> The MIoU of val did not improve.")
        print("=====> The best val MIoU is still {:.3f} from epoch {:3f}".format(best_MIou, best_epoch))
    epoch += 1

    return epoch_miou, classes_miou, epoch_fwiou, epoch_mp, epoch_pa, epoch_mpa