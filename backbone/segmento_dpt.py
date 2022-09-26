import torch
import torch.nn as nn
from etri.backbone.dpt.models import DPTSegmentationModel

def etriDPT(weights, num_classes):
    
    output_channels = num_classes + 1

    model = DPTSegmentationModel(
        num_classes = output_channels,
        backbone = "vitb_rn50_384",
    )

    return load_model(model, weights)


def load_model(model, weights):

    if weights == "":
        dpt_hybrid_ade20k = "/media/dataset2/etri/dpt_hybrid-ade20k-53898607.pt"
        ade20k = torch.load(dpt_hybrid_ade20k)

        model.initialize_weights()
        etri_dpt = model.state_dict().copy()
        
        heads = [
            'scratch.output_conv.4.weight', 
            'scratch.output_conv.4.bias', 
            'auxlayer.4.weight', 
            'auxlayer.4.bias'
            ]

        for key in ade20k:
            if not key in heads:
                etri_dpt[key] = ade20k[key]
                
        model.load_state_dict(etri_dpt)
        print("Train from ADE20K Pretrained")
        
    else:
        saved_weights = torch.load(weights)

        if 'state_dict' in saved_weights.keys():
            saved_weights = saved_weights['state_dict']

        model.load_state_dict(saved_weights)
        print("Train from saved weights")

    return model