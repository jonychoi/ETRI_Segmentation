import torch
import os

def save_checkpoints(save_dir, mode, model_name, ema, epoch, optimizer, best_MIou, epoch_mpa, classes_miou, epoch_fwiou, epoch_mp, epoch_pa):
    
    filename = os.path.join(save_dir + mode + '/', 'Epoch::{}:: | Model: {}  MIoU: {:.3f} | MPA: {:3f}.pth'.format(epoch, model_name, best_MIou, epoch_mpa))
    
    torch.save({
        'epoch': epoch + 1,
        'state_dict': ema.model.state_dict(),
        'shadow': ema.shadow,            
        'optimizer': optimizer.state_dict(),
        'best_MIou': best_MIou,
        'epoch_mpa': epoch_mpa,
        'epoch_classes_miou': classes_miou,
        'epoch_fwiou': epoch_fwiou,
        'epoch_mp': epoch_mp,
        'epoch_pa': epoch_pa,
        'best_MIou': best_MIou
    }, filename)