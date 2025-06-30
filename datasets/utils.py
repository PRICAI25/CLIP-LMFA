import numpy as np
import torch
from open_clip import OPENAI_DATASET_STD, OPENAI_DATASET_MEAN

from datasets.mvtec import MVTecDataset
from datasets.visa import VisADataset
from datasets.mpdd import MPDDDataset
from datasets.btad import BTADDataset
from datasets.mvtec_3d import MVTec_3DDataset
from datasets.Brain import BrainDataset
from datasets.Liver import LiverDataset
from datasets.Retina_RESC import Retina_RESCDataset
# 定义了一个名为 undo_transform 的函数，用于将经过标准化处理的图像数据还原为原始格式
def undo_transform(image: torch.Tensor, unorm: bool = True) -> np.array:
    # 是否进行反标准化？：根据 unorm 参数的值决定是否进行反标准化操作
    if unorm:
        # 乘以标准差并加上均值：如果 unorm 为 True，则对图像数据进行反标准化操作
        image = (

            image * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]
        ) + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]
    # 调整维度并转换类型：将图像数据的维度从 [C, H, W] 转换为 [H, W, C]，并将其转换为 uint8 类型的 NumPy 数组
    return (image.permute(1, 2, 0) * 255).type(torch.uint8).numpy()

# 定义了一个名为 min_max_normalization 的函数，用于对输入的三维数组（形状为 (batch_size, height, width)）进行最小-最大归一化
def min_max_normalization(arr: np.ndarray) -> np.ndarray:
    # Normalization per image in the batch
    # arr: (batch_size, height, width)
    # 计算每个图像在高度和宽度方向上的最小值 arr_min 和最大值 arr_max
    arr_min = arr.min(axis=(1, 2), keepdims=True)
    arr_max = arr.max(axis=(1, 2), keepdims=True)
    # 使用公式 (arr - arr_min) / (arr_max - arr_min) 对数组进行归一化
    return (arr - arr_min) / (arr_max - arr_min)
