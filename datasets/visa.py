from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from datasets.base import BaseDataset, DatasetSplit


class VisADataset(BaseDataset):
    CLASSNAMES = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]
    # 定义了一个名为 get_image_data 的方法，用于从 CSV 文件中读取图像路径和掩码路径，
    # 并根据不同的数据集分割（训练、验证、测试）进行处理
    def get_image_data(self):
        imgpaths_per_class = dict()
        maskpaths_per_class = dict()
        # 读取CSV文件
        file = Path(self.source) / "split_csv" / "1cls.csv"

        dt = pd.read_csv(file)
        # 遍历CSV文件
        for i, row in dt.iterrows():
            classname, set, label, image_path, mask_path = row
            # 类别名在使用列表中?
            if classname not in self.classnames_to_use:
                continue
            # 类别名是否已存在?
            if classname not in imgpaths_per_class:
                imgpaths_per_class[classname] = defaultdict(list)
                maskpaths_per_class[classname] = defaultdict(list)
            # 处理标签
            if label == "normal":
                label = "good"
            else:
                label = "anomaly"
            # 处理图像路径
            img_src_path = self.source / image_path
            # 掩码路径是否存在?
            if not pd.isna(mask_path) and mask_path:
                # 处理掩码路径
                msk_src_path = self.source / mask_path
            else:
                msk_src_path = None
            # 是否属于当前数据集分割?
            if (self.split == DatasetSplit.TEST and set == "test") or (
                self.split in [DatasetSplit.TRAIN, DatasetSplit.VAL] and set == "train"
            ):
                # 添加到相应字典
                imgpaths_per_class[classname][label].append(img_src_path)
                maskpaths_per_class[classname][label].append(msk_src_path)
        # 是否需要训练/验证集分割?
        if self.train_val_split < 1.0:
            for classname in imgpaths_per_class:
                for label in imgpaths_per_class[classname]:
                    n_images = len(imgpaths_per_class[classname][label])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    # 是否为训练集?
                    if self.split == DatasetSplit.TRAIN:
                        # 分割训练集
                        imgpaths_per_class[classname][label] = imgpaths_per_class[
                            classname
                        ][label][:train_val_split_idx]
                        maskpaths_per_class[classname][label] = maskpaths_per_class[
                            classname
                        ][label][:train_val_split_idx]
                    # 是否为验证集?
                    elif self.split == DatasetSplit.VAL:
                        # 分割验证集
                        imgpaths_per_class[classname][label] = imgpaths_per_class[
                            classname
                        ][label][train_val_split_idx:]
                        maskpaths_per_class[classname][label] = maskpaths_per_class[
                            classname
                        ][label][train_val_split_idx:]

        # Unrolls the data dictionary to an easy-to-iterate list.
        # 生成迭代数据
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
