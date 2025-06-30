import os
from enum import Enum
from pathlib import Path
from typing import List, Union

import PIL
import torch
from torchvision import transforms
from torchvision.transforms.v2.functional import pil_to_tensor

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
# OpenCLIP preprocessing
IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset for AD.
    """

    def __init__(
        self,
        source: Path,
        classname: Union[str, List],
        resize: Union[int, list] = 256,
        square: bool = True,
        split: DatasetSplit = DatasetSplit.TRAIN,
        train_val_split: float = 1.0,
        # train_val_split: float = 0.9,
        rotate_degrees: float = 0,
        translate: float = 0,
        brightness_factor: float = 0,
        contrast_factor: float = 0,
        saturation_factor: float = 0,
        gray_p: float = 0,
        h_flip_p: float = 0,
        v_flip_p: float = 0,
        scale: float = 0,
        **kwargs,
    ):
        """
        Args:
            source: [Path]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int or list[int]]. Size the loaded image initially gets resized to.
                    If square=True, image is resized to a square with side=resize
                    If square=False, smaller edge of the image will be matched to resize, maintaining aspect ratio
            square: [bool]. Whether to resize to a square or non-square image.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = (
            [classname] if not isinstance(classname, list) else classname
        )
        self.train_val_split = train_val_split
        self.resize = resize
        # 调用 get_image_data 方法获取图像路径和需要迭代的数据
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        # 定义了一系列随机图像变换操作，如颜色抖动、水平翻转、垂直翻转、灰度转换、仿射变换等。
        self.random_transform_img = [
            # transforms.RandomRotation(
            #   rotate_degrees, transforms.InterpolationMode.BILINEAR
            # ),
            transforms.ColorJitter(
                brightness_factor, contrast_factor, saturation_factor
            ),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
        # 使用 transforms.Compose 将这些变换组合成一个变换链
        self.random_transform_img = transforms.Compose(self.random_transform_img)

        if not isinstance(resize, list):
            resize = [resize]

        self.transform_img = []
        self.transform_mask = []

        # Multiple resize transforms
        # 根据 resize 参数，定义多个图像和掩码的变换操作。
        # 每个变换操作包括调整大小、转换为张量、归一化等步骤。
        # 使用 transforms.Compose 将这些变换组合成一个变换链
        for sz in resize:
            transform_img = [
                transforms.Resize(
                    (sz, sz) if square else sz,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                # transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img.append(transforms.Compose(transform_img))

            transform_mask = [
                transforms.Resize((sz, sz) if square else sz),
                # transforms.CenterCrop(imagesize),
            ]
            self.transform_mask.append(transforms.Compose(transform_mask))

    def __getitem__(self, idx):
        resize = self.resize if isinstance(self.resize, list) else [self.resize]
        # 获取图像路径、类别名、异常状态和掩码路径
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        # 加载图像并转换为 RGB 格式
        image = PIL.Image.open(image_path).convert("RGB")
        # 记录原始图像尺寸
        original_img_width, original_img_height = image.size
        # 对图像进行随机变换
        image = self.random_transform_img(image)
        # 根据 resize 参数对图像进行不同尺寸的变换
        image = {
            sz: transform_img(image)
            for sz, transform_img in zip(resize, self.transform_img)
        }
        # 是否为测试集且存在掩码路径?shi 加载掩码并转换为张量 fou 生成全零的掩码张
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = (pil_to_tensor(mask) != 0).float()
        else:
            mask = torch.zeros([1, original_img_height, original_img_width])
        #  对掩码进行不同尺寸的变换
        mask = {
            sz: (transform_mask(mask) > 0.5).float()
            for sz, transform_mask in zip(resize, self.transform_mask)
        }
        # resize 是否为列表? shi 返回多尺寸的图像和掩码
        if not isinstance(self.resize, list):
            image = next(iter(image.values()))
            mask = next(iter(mask.values()))

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.relpath(image_path, self.source).split(".")[0],
            "image_path": str(image_path),
            "mask_path": "None" if mask_path is None else str(mask_path),
            "original_img_height": original_img_height,
            "original_img_width": original_img_width,
        }
    # 定义了一个特殊方法 __len__，用于返回对象的长度。具体来说，它返回 self.data_to_iterate 的长度
    def __len__(self):
        return len(self.data_to_iterate)
    # 定义了一个名为 get_image_data 的方法，该方法在类中声明但未实现具体功能。
    # 当调用此方法时，会抛出 NotImplementedError 异常，提示子类需要重写该方法来提供具体的实现
    def get_image_data(self):
        raise NotImplementedError


