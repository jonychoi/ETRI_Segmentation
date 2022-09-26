import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}


def get_attention(name):
    def hook(module, input, output):
        # import pdb; pdb.set_trace();
        # len(input) == 1, but input is tuple so input[0]
        # input[0].shape, torch.Size([4, 901, 768])
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def forward_vit(pretrained, x, large = False):
    b, c, h, w = x.shape

    glob = pretrained.model.forward_flex(x)

    layer_1_act = pretrained.activations["1"]
    layer_2_act = pretrained.activations["2"]
    layer_3_act = pretrained.activations["3"]
    layer_4_act = pretrained.activations["4"]

    if large:
        layer_1_attn = pretrained.attention["attn_1"]
        layer_2_attn = pretrained.attention["attn_2"]
        layer_3_attn = pretrained.attention["attn_3"]
        layer_4_attn = pretrained.attention["attn_4"]
        layer_5_attn = pretrained.attention["attn_5"]
        layer_6_attn = pretrained.attention["attn_6"]
        layer_7_attn = pretrained.attention["attn_7"]
        layer_8_attn = pretrained.attention["attn_8"]
        layer_9_attn = pretrained.attention["attn_9"]
        layer_10_attn = pretrained.attention["attn_10"]
        layer_11_attn = pretrained.attention["attn_11"]
        layer_12_attn = pretrained.attention["attn_12"]
        layer_13_attn = pretrained.attention["attn_13"]
        layer_14_attn = pretrained.attention["attn_14"]
        layer_15_attn = pretrained.attention["attn_15"]
        layer_16_attn = pretrained.attention["attn_16"]
        layer_17_attn = pretrained.attention["attn_17"]
        layer_18_attn = pretrained.attention["attn_18"]
        layer_19_attn = pretrained.attention["attn_19"]
        layer_20_attn = pretrained.attention["attn_20"]
        layer_21_attn = pretrained.attention["attn_21"]
        layer_22_attn = pretrained.attention["attn_22"]
        layer_23_attn = pretrained.attention["attn_23"]
        layer_24_attn = pretrained.attention["attn_24"]
    else:
        layer_1_attn = pretrained.attention["attn_1"]
        layer_2_attn = pretrained.attention["attn_2"]
        layer_3_attn = pretrained.attention["attn_3"]
        layer_4_attn = pretrained.attention["attn_4"]
        layer_5_attn = pretrained.attention["attn_5"]
        layer_6_attn = pretrained.attention["attn_6"]
        layer_7_attn = pretrained.attention["attn_7"]
        layer_8_attn = pretrained.attention["attn_8"]
        layer_9_attn = pretrained.attention["attn_9"]
        layer_10_attn = pretrained.attention["attn_10"]
        layer_11_attn = pretrained.attention["attn_11"]
        layer_12_attn = pretrained.attention["attn_12"]
    

    layer_1 = pretrained.act_postprocess1[0:2](layer_1_act)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2_act)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3_act)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4_act)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )
    
    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    layers = layer_1, layer_2, layer_3, layer_4
    activations = layer_1_act, layer_2_act, layer_3_act, layer_4_act
    attentions = layer_1_attn, layer_2_attn, layer_3_attn, layer_4_attn, layer_5_attn, layer_6_attn, layer_7_attn, layer_8_attn, layer_9_attn, layer_10_attn, layer_11_attn, layer_12_attn

    return layers, activations, attentions

def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
    large = False,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if large:
        pretrained.model.blocks[0].attn.register_forward_hook(get_attention("attn_1"))
        pretrained.model.blocks[1].attn.register_forward_hook(get_attention("attn_2"))
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_3"))
        pretrained.model.blocks[3].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.model.blocks[4].attn.register_forward_hook(get_attention("attn_5"))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_6"))
        pretrained.model.blocks[6].attn.register_forward_hook(get_attention("attn_7"))
        pretrained.model.blocks[7].attn.register_forward_hook(get_attention("attn_8"))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_9"))
        pretrained.model.blocks[9].attn.register_forward_hook(get_attention("attn_10"))
        pretrained.model.blocks[10].attn.register_forward_hook(get_attention("attn_11"))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_12"))
        pretrained.model.blocks[12].attn.register_forward_hook(get_attention("attn_13"))
        pretrained.model.blocks[13].attn.register_forward_hook(get_attention("attn_14"))
        pretrained.model.blocks[14].attn.register_forward_hook(get_attention("attn_15"))
        pretrained.model.blocks[15].attn.register_forward_hook(get_attention("attn_16"))
        pretrained.model.blocks[16].attn.register_forward_hook(get_attention("attn_17"))
        pretrained.model.blocks[17].attn.register_forward_hook(get_attention("attn_18"))
        pretrained.model.blocks[18].attn.register_forward_hook(get_attention("attn_19"))
        pretrained.model.blocks[19].attn.register_forward_hook(get_attention("attn_20"))
        pretrained.model.blocks[20].attn.register_forward_hook(get_attention("attn_21"))
        pretrained.model.blocks[21].attn.register_forward_hook(get_attention("attn_22"))
        pretrained.model.blocks[22].attn.register_forward_hook(get_attention("attn_23"))
        pretrained.model.blocks[23].attn.register_forward_hook(get_attention("attn_24"))
        pretrained.attention = attention
    else:
        if enable_attention_hooks:
            pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
                get_attention("attn_1")
            )
            pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
                get_attention("attn_2")
            )
            pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
                get_attention("attn_3")
            )
            pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
                get_attention("attn_4")
            )
            pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_vit_b_rn50_backbone(
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()

    pretrained.model = model

    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
            get_activation("1")
        )
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
            get_activation("2")
        )
    '''
    Edited x => error since the feature size should be changed either
    '''
    # pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    # pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    ''''''
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    if enable_attention_hooks: # if use blocks8, 11, same as base == large == hybrid
        # pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_1"))
        # pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_2"))
        # pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_3"))
        # pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.model.blocks[0].attn.register_forward_hook(get_attention("attn_1"))
        pretrained.model.blocks[1].attn.register_forward_hook(get_attention("attn_2"))
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_3"))
        pretrained.model.blocks[3].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.model.blocks[4].attn.register_forward_hook(get_attention("attn_5"))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_6"))
        pretrained.model.blocks[6].attn.register_forward_hook(get_attention("attn_7"))
        pretrained.model.blocks[7].attn.register_forward_hook(get_attention("attn_8"))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_9"))
        pretrained.model.blocks[9].attn.register_forward_hook(get_attention("attn_10"))
        pretrained.model.blocks[10].attn.register_forward_hook(get_attention("attn_11"))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_12"))
        pretrained.attention = attention

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    if use_vit_only == True:
        pretrained.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        pretrained.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
    else:
        pretrained.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        pretrained.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)

    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_deit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_distil_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model(
        "vit_deit_base_distilled_patch16_384", pretrained=pretrained
    )

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        start_index=2,
        enable_attention_hooks=enable_attention_hooks,
    )
