import torch
import torch.hub

from torchvision.models import resnet18, resnet50, resnet101, resnet152
from torchvision.models import mobilenet_v2, shufflenet_v2_x1_0, mobilenet_v3_small
from torchvision.models import convnext_tiny
from .dummy_model import dummy_model
from .clip import clip_vit_b16, clip_vit_b32, clip_vit_l14, clip_r50, clip_r50x4, clip_r50x16, clip_r50x64, clip_r101

from .register import MODEL


@MODEL.register
def PyTorchHub(repo: str, name: str, **kwargs):
    return torch.hub.load(repo, name, **kwargs)

@MODEL.register
def deit_tiny_patch16_224(pretrained=True):
    model = torch.hub.load('codebase/third_party/deit', 'deit_tiny_patch16_224', pretrained=pretrained, source="local")
    return model

@MODEL.register
def deit_small_patch16_224(pretrained=True):
    model = torch.hub.load('codebase/third_party/deit', 'deit_small_patch16_224', pretrained=pretrained, source="local")
    return model

@MODEL.register
def deit_base_patch16_224(pretrained=True):
    model = torch.hub.load('codebase/third_party/deit', 'deit_base_patch16_224', pretrained=pretrained, source="local")
    return model

MODEL.register(resnet18)
MODEL.register(resnet50)
MODEL.register(resnet101)
MODEL.register(resnet152)
MODEL.register(mobilenet_v2)
MODEL.register(mobilenet_v3_small)
MODEL.register(shufflenet_v2_x1_0)
MODEL.register(convnext_tiny)
MODEL.register(clip_vit_b16)
MODEL.register(clip_vit_b32)
MODEL.register(clip_vit_l14)
MODEL.register(clip_r50)
MODEL.register(clip_r50x4)
MODEL.register(clip_r50x16)
MODEL.register(clip_r50x64)
MODEL.register(clip_r101)
