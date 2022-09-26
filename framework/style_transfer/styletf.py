
import torch 
import random
import torch.nn as nn
from torchvision import transforms 

from .net import dec, enc
from .function import adaptive_instance_normalization, coral

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:   # interpolation O
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:   # interpoaltion X 
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


# # content 한장이랑 style 여러장 들어올듯
def mixed_style_transfer(vgg, decoder, content, style, alpha = 1.0, interpolation_weights = None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    styles_f = torch.mean(style_f, dim = 0) # style feature 의 평균 
    styles_f = styles_f.unsqueeze(0) # 
    # print(styles_f.size())
    # print(content_f.size())
    # import pdb; pdb.set_trace()
    feat = adaptive_instance_normalization(content_f, styles_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


vgg_path = "/home/cvlab06/project/segmento_hj2/segmento_pascal/framework/style_transfer/vgg_normalised.pth"
decoder_path = "/home/cvlab06/project/segmento_hj2/segmento_pascal/framework/style_transfer/decoder.pth"

# no interpolation.. pascal - dram 1:1 로

def style_transferred(args, contents, styles, alpha):
    decoder = dec()
    vgg = enc()

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    vgg.to(device)

    # denormalize
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_MEAN = torch.Tensor(IMG_MEAN).reshape(1,3,1,1)
    IMG_STD = [0.229, 0.224, 0.225]
    IMG_STD = torch.Tensor(IMG_STD).reshape(1,3,1,1)

    contents = (contents.cpu()*IMG_STD) + IMG_MEAN
    styles = (styles.cpu()*IMG_STD) + IMG_MEAN
    contents = contents.to(device)
    styles = styles.to(device)

    # content_tf = test_transform(args.content_size, args.crop)
    # style_tf = test_transform(args.style_size, args.crop)
    new_batch = []
    for i in range(args.batch_size):
        content = contents[i].unsqueeze(0)
        style = styles[i].unsqueeze(0)
        
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, alpha)
            new_batch.append(output)
    
    new_batch_ = torch.cat(new_batch, dim = 0)
    # re - normalize 
    new_batch_ = (new_batch_.cpu() - IMG_MEAN) / IMG_STD
    new_batch_ = new_batch_.to(device)

    return new_batch_ 


## editing ...
def style_transferred_batch(args, contents, styles, alpha):
    decoder = dec()
    vgg = enc()

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    vgg.to(device)

    # denormalize
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_MEAN = torch.Tensor(IMG_MEAN).reshape(1,3,1,1)
    IMG_STD = [0.229, 0.224, 0.225]
    IMG_STD = torch.Tensor(IMG_STD).reshape(1,3,1,1)

    contents = (contents.cpu()*IMG_STD) + IMG_MEAN
    styles = (styles.cpu()*IMG_STD) + IMG_MEAN
    contents = contents.to(device)
    styles = styles.to(device)      # style batch 가 들어온 것

    # content_tf = test_transform(args.content_size, args.crop)
    # style_tf = test_transform(args.style_size, args.crop)
    # 
    
    new_batch = []
    for i in range(args.batch_size):
        k = random.randrange(2, args.batch_size)
        # style = styles[i].unsqueeze(0)
        idx = [ _ for _ in range(args.batch_size)]
        idx = random.sample(idx, k)
        style_b = []
        for id_ in idx:
            style_b.append(styles[id_])
        style_b = torch.stack(style_b, dim = 0)
        
        content = contents[i].unsqueeze(0)
        with torch.no_grad():
            output = mixed_style_transfer(vgg, decoder, content, style_b, alpha)
            new_batch.append(output)
    
    new_batch_ = torch.cat(new_batch, dim = 0)
    # re - normalize 
    new_batch_ = (new_batch_.cpu() - IMG_MEAN) / IMG_STD
    new_batch_ = new_batch_.to(device)
    return new_batch_ 





    
    