import sys
sys.path.append("/home/cvlab02/project/etri")

from backbone.etri_dpt import etriDPT

def get_model(args):
    if args.backbone == "dpt-hybrid":
        model = etriDPT(
            num_classes = args.top_k,
            weights=args.weights, 
        )
    else:
        raise NotImplementedError()
    return model


