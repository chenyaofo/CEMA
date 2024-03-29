import copy
import functools
from typing import Callable

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as functional

import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False):
    # if conf.args.dataset in ['cifar10', 'cifar100']:
    #     img_shape = (32, 32, 3)
    # else:
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        GaussianNoise(0, gaussian_std),
        Clip(clip_min, clip_max)
    ])
    return tta_transforms


# def softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

# @torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def update_ema(ema_model: nn.Module, model: nn.Module, alpha_teacher: float):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def stochastic_restore(model: nn.Module, restoration_factor: float, model_state: dict):
    for nm, m in model.named_modules():
        for npp, p in m.named_parameters():
            if npp in ['weight', 'bias'] and p.requires_grad:
                mask = (torch.rand(p.shape, device="cuda:0") < restoration_factor).float()
                with torch.no_grad():
                    p.data = model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)


def configure_model(model: nn.Module):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
        else:
            m.requires_grad_(True)
    return


def collect_params(model: nn.Module):
    """Collect all trainable parameters.
    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:  # isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    # print(nm, np)
    return params, names


class CoTTA:
    def __init__(
        self,
        model: nn.Module,
        ema_factor: float,
        restoration_factor: float,
        n_aug: int,
        aug_threshold: float,
        tta_lr: float
    ) -> None:
        self.model = model
        self.ema_factor = ema_factor
        self.restoration_factor = restoration_factor
        self.aug_threshold = aug_threshold
        self.n_aug = n_aug
        self.tta_lr = tta_lr

        configure_model(self.model)
        params, names = collect_params(self.model)

        self.optimizer = optim.SGD(params,
                                   lr=self.tta_lr,
                                   momentum=0.9,
                                   dampening=0,
                                   weight_decay=0.0,
                                   nesterov=True)

        self.model_state = copy.deepcopy(self.model.state_dict())
        self.anchor_model = copy.deepcopy(self.model)
        self.ema_model = copy.deepcopy(self.model)

        self.model = self.model.cuda()
        self.anchor_model = self.anchor_model.cuda()
        self.ema_model = self.ema_model.cuda()

        for param in self.ema_model.parameters():
            param.detach_()


        self.transform = get_tta_transforms()


    def infer(self, x: torch.Tensor):
        outputs: torch.Tensor = self.model(x)

        with torch.no_grad():
            anchor_prob: torch.Tensor = functional.softmax(self.anchor_model(x), dim=1).max(1)[0]
            self.ema_model.train()
            standard_ema: torch.Tensor = self.ema_model(x)

        # Threshold choice discussed in supplementary
        # enable data augmentation for vision datasets
        outputs_emas = []
        # if anchor_prob.mean(0) < self.aug_threshold:
        #     for i in range(self.n_aug):
        #         # print("start aug")
        #         outputs_ = self.ema_model(self.transform(x)).detach()
        #         outputs_emas.append(outputs_)
        #     outputs_ema = torch.stack(outputs_emas).mean(0)
        # else:
        outputs_ema = standard_ema
        # Student update
        loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Teacher update
        with torch.no_grad():
            self.ema_model = update_ema(ema_model=self.ema_model, model=self.model, alpha_teacher=self.ema_factor)

        stochastic_restore(self.model, self.restoration_factor, self.model_state)

        return outputs_ema
