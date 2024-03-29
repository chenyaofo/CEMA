from .dali import build_dali_imagenet_loader
from .native import build_imagenet_ra_loader

from ..register import DATA


@DATA.register
def imagenet_c(root, image_size, mean, std, batch_size, num_workers, dali_gpu, percent, **kwargs):
    return build_dali_imagenet_loader(root, image_size, mean, std, batch_size, num_workers, dali_gpu, percent)

@DATA.register
def imagenet_a(root, image_size, mean, std, batch_size, num_workers, dali_gpu, percent, **kwargs):
    return build_imagenet_ra_loader(root, image_size, mean, std, batch_size, num_workers, dali_gpu, percent)

@DATA.register
def imagenet_r(root, image_size, mean, std, batch_size, num_workers, dali_gpu, percent, **kwargs):
    return build_imagenet_ra_loader(root, image_size, mean, std, batch_size, num_workers, dali_gpu, percent)