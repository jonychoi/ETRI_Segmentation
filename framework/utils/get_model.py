import sys
sys.path.append("/home/cvlab06/project/etri")

from segmento_pascal.backbone.segmento_dpt import SegmentoDPT

def get_model(args):
    if args.backbone == "dpt-hybrid":
        model = SegmentoDPT(
            num_classes = args.top_k,
            weights=args.weights, 
        )
    else:
        raise NotImplementedError()
    return model


