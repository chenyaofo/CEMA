import os
import typing
# corruption types for ImageNet-C

IMAGENET_C = dict(
    noise=[
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
    ],

    blur=[
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
    ],

    weather=[
        "snow",
        "frost",
        "fog",
        "brightness",
    ],

    digital=[
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
)


# cnstant in webdataset

WDS_BUFFER_SIZE = 10000


def get_imagenet_c_corruption_type(hint: str):
    coarse_types = IMAGENET_C.keys()
    fined_types = []
    for k, v in IMAGENET_C.items():
        fined_types += v
    if hint in coarse_types:
        return IMAGENET_C[hint]
    elif hint in fined_types:
        return [hint]
    else:
        raise ValueError(f"hint={hint} is not in coarse_types or fined_types of ImageNet-C")


def iter_imagenet_c(
    root: str,
    corruptions: typing.List[str],
    severities: typing.List[int],
    n_repeats: int = 1,
    mixed: bool = False,
    pass_through: bool = False
):
    if pass_through:
        return [["pass_through", 0, root]]
    rev = []
    for _ in range(n_repeats):
        for corrup in corruptions:
            for corrup_fine_grained_type in get_imagenet_c_corruption_type(corrup):
                for seve in severities:
                    rev.append(
                        (
                            corrup_fine_grained_type,
                            seve,
                            os.path.join(root,
                                         corrup_fine_grained_type,
                                         f"{seve}"
                                         )
                        )
                    )
    if mixed:
        data_paths = []
        for (corrup, seve, data_path) in rev:
            data_paths.append(data_path)
        return [["mixed", 0, data_paths]]
    else:
        return rev
