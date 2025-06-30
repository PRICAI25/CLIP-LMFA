"""
  @Author: 王权
  @FileName: Retina.py
  @DateTime: 2025/1/22 16:37
  @SoftWare: PyCharm
"""
"""
  @Author: 王权
  @FileName: Liver.py
  @DateTime: 2025/1/22 15:43
  @SoftWare: PyCharm
"""
from collections import defaultdict
from pathlib import Path

import torch

from datasets.base import BaseDataset, DatasetSplit


class RetinaDataset(BaseDataset):
    CLASSNAMES = [
        "Retina_OCT2017_AD",
    ]
    # 定义了一个名为 get_image_data 的方法，用于从指定的数据集中获取图像路径和掩码路径
    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        catalog = defaultdict(list)

        for classname in self.classnames_to_use:
            # 获取类别路径和掩码路径
            split_value = 'valid' if self.split.value == 'train' else self.split.value
            classpath = self.source / classname / split_value
            maskpath = self.source / classname / split_value
            print(f"Classpath: {classpath}")
            print(f"Maskpath: {maskpath}")
            anomaly_paths = classpath.glob("*")

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            # 遍历异常路径
            for anomaly_path in anomaly_paths:
                print(f"anomaly_path: {anomaly_path}, split_value: {split_value}")
                if str(anomaly_path) == '/home/wsc/BMVC-FADE-main/data/Med/Retina_OCT2017/Retina_OCT2017_AD/valid/Ungood' and split_value == 'valid':
                    print("Condition met, breaking loop")
                    break  # 跳出循环
                anomaly = anomaly_path.name
                anomaly_path = anomaly_path / "img"
                print(f"anomaly_path_two: {anomaly_path}")
                # print(f" anomaly_path : {anomaly_path}" )
                anomaly_files = sorted(anomaly_path.glob("*"))
                # 获取图像文件路径
                imgpaths_per_class[classname][anomaly] = anomaly_files
                # 存储图像路径
                catalog["classname"].append([classname] * len(anomaly_files))
                catalog["anomaly"].extend([anomaly] * len(anomaly_files))
                catalog["img_path"].extend(anomaly_files)
                # 处理训练/验证集分割
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    # 根据分割比例分割图像路径
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]
                # 处理测试集掩码路径
                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = maskpath / anomaly / "anomaly_mask"
                    anomaly_mask_files = sorted(anomaly_mask_path.glob("*"))
                    # print(f"anomaly_mask_files : {anomaly_mask_files}" )
                    maskpaths_per_class[classname][anomaly] = anomaly_mask_files
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        # 展开数据字典，使其成为一个易于迭代的列表。
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
        # 返回图像路径和数据列表
        return imgpaths_per_class, data_to_iterate
