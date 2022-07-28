import random

from PIL import ImageFilter
from torchvision import transforms


INTERP = transforms.InterpolationMode.BICUBIC
IMAGENET_SHAPE = (3, 224, 224)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    Code from https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class DINONormTrans:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, img):
        return self.transform(img)


class DINOConsistentTrans:
    def __init__(self) -> None:
        # based on DINO global transforms
        # https://github.com/facebookresearch/dino/blob/main/main_dino.py
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        # guaranteed blur transform
        blur_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGENET_SHAPE[1], scale=(.4, 1),
                                         interpolation=INTERP),
            flip_and_color_jitter,
            GaussianBlur(1.0),
        ])
        # chance of blur and/or solarize transform
        blur_solarize_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGENET_SHAPE[1], scale=(.4, 1),
                                         interpolation=INTERP),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            transforms.RandomSolarize(128, p=.2),
        ])
        # randomly apply one of two above
        self.transform = TransformTwice(transforms.Compose([
            transforms.Resize(IMAGENET_SHAPE[1], interpolation=INTERP),
            transforms.RandomChoice([blur_transform, blur_solarize_transform]),
            DINONormTrans(),
        ]))

    def __call__(self, img):
        return self.transform(img)


class DINOCropTrans:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(IMAGENET_SHAPE[1], interpolation=INTERP),
            transforms.RandomResizedCrop(IMAGENET_SHAPE[1], scale=(.4, 1),
                                         interpolation=INTERP),
            transforms.RandomHorizontalFlip(),
            DINONormTrans(),
        ])

    def __call__(self, img):
        return self.transform(img)


class DINOTestTrans:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(IMAGENET_SHAPE[1], interpolation=INTERP),
            DINONormTrans(),
        ])

    def __call__(self, img):
        return self.transform(img)
