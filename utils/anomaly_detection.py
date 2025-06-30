from typing import Union

import cv2
import numpy as np

from utils.adapter import ImageHead, TextHead
from utils.embeddings import retrieve_image_embeddings
from utils.image_model import ImageModel
from utils.text_model import TextModel


def predict_classification(
    text_model: TextModel,
    image_embeddings: dict,
    img_size: int,
    feature_type: str,
) -> np.ndarray:
    """
    Language-guided zero-shot classification
    Perform classification at a single image resolutions

    Args:
        text_model: Text embedding model
        image_embeddings: Dictionary of all embeddings
        img_size: Image resolution of the embeddings to use
        feature_type: 'clip' or 'gem'

    Returns:
        Classification scores of dimension (batch_size)
    """
    # 提取图像嵌入：调用 retrieve_image_embeddings 函数，传入 image_embeddings、img_size、feature_type 和 token_type="cls"，获取指定分辨率和特征类型的 cls 嵌入
    cls_embeddings = retrieve_image_embeddings(
        image_embeddings,
        img_size=img_size,
        feature_type=feature_type,
        token_type="cls",
    )

    #
    # image_head = ImageHead(feature_dim = emb_dim, out_dim = emb_dim)
    # text_head = TextHead(feature_dim = emb_dim, out_dim = emb_dim)
    # patch_embeddings, _ = image_head(patch_embeddings, patch_embeddings)
    # text_model = text_head(patch_embeddings, patch_embeddings)

    # 使用文本模型预测概率：调用 text_model.predict_proba 方法，传入提取的 cls 嵌入，得到分类概率
    scores = text_model.predict_proba(cls_embeddings)
    # 返回预测概率的第二列，通常表示正类的概率
    return scores[:, 1]

# 定义了一个名为 text_image_matching 的函数，用于通过匹配图像块嵌入和文本嵌入进行分割
def text_image_matching(
    text_model: TextModel,
    patch_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Segmentation by matching image patch embeddings to the text embeddings.

    Args:
        text_model: Text embedding model
        patch_embeddings: Query patch embeddings of dimensions (batch_size, num_patches, emb_dim)

    Returns:
        Segmentation scores of dimension (batch_size, num_patches)
    """

    # 获取 patch_embeddings 的形状，提取 batch_size、num_patches 和 emb_dim。
    batch_size, num_patches, emb_dim = patch_embeddings.shape

    # image_head = ImageHead(feature_dim = emb_dim, out_dim = emb_dim)
    # text_head = TextHead(feature_dim = emb_dim, out_dim = emb_dim)
    # patch_embeddings, _ = image_head(patch_embeddings, text_model)
    # text_model = text_head(patch_embeddings, text_model)

    # 将 patch_embeddings 重塑为 (batch_size * num_patches, emb_dim)。使用 text_model 预测这些嵌入的概率，
    # 结果维度为 (batch_size * num_patches, 2)。将预测结果重塑为 (batch_size, num_patches, 2)
    segmentations = text_model.predict_proba(
        patch_embeddings.reshape((batch_size * num_patches, emb_dim))
    ).reshape((batch_size, num_patches, 2))
    # 返回预测结果的第三维的最后一列（通常表示正类的概率）
    return segmentations[..., 1]


def image_image_matching(
    image_model: ImageModel,
    patch_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Segmentation by matching image patch embeddings to the training image patch embeddings.

    Args:
        image_model: Image embedding model
        patch_embeddings: Query patch embeddings of dimensions (batch_size, num_patches, emb_dim)

    Returns:
        Segmentation scores of dimension (batch_size, num_patches)
    """
    batch_size, num_patches, emb_dim = patch_embeddings.shape
    # 重塑 patch_embeddings：将 patch_embeddings 从 (batch_size, num_patches, emb_dim) 重塑为 (batch_size * num_patches, emb_dim)，以便输入到 image_model 中进行预测。
    # 使用 image_model 预测分割概率：调用 image_model.predict_proba 方法，传入重塑后的 patch_embeddings，得到每个图像块的分割概率。
    # 重塑预测结果：将预测结果从 (batch_size * num_patches, num_classes) 重塑为 (batch_size, num_patches)，以便返回分割分数
    # image_head = ImageHead(feature_dim=emb_dim, out_dim=emb_dim)
    #
    # patch_embeddings, _ = image_head(patch_embeddings, patch_embeddings)

    segmentations = image_model.predict_proba(
        patch_embeddings.reshape((batch_size * num_patches, emb_dim))
    ).reshape((batch_size, num_patches))
    # 返回最终的分割分数，形状为 (batch_size, num_patches)
    return segmentations

# 定义了一个函数 resize_and_aggregate_segmentations，用于处理多分辨率的分割图像数据。
def resize_and_aggregate_segmentations(
    segmentations_multi_resolution: list[np.ndarray],
    output_size: int,
) -> np.ndarray:
    """
    Resize and aggregate segmentations of different resolutions by batch.

    Args:
        segmentations_multi_resolution: A list of arrays where each array contains segmentations of
                                        one resolution. Each array has dimension (batch_size, height, width).
        output_size: The size to resize the segmentations to before aggregation.

    Returns:
        Segmentations array of dimension (batch_size, output_size, output_size)
    """
    # 获取批次大小
    batch_size = segmentations_multi_resolution[0].shape[0]
    return np.stack( # np.stack([...]) 存储结果：
        [
            # 计算调整后图像的平均值
            np.mean(
                [
                    # 调整每个分辨率的图像大小
                    cv2.resize(
                        segmentations_single_resolution[i], (output_size, output_size)
                    )
                    for segmentations_single_resolution in segmentations_multi_resolution
                ],
                axis=0,
            )
            for i in range(batch_size)
        ]
    )

# 定义了一个名为 predict_segmentation 的函数，用于执行语言引导（零样本）或视觉引导（少样本）的图像分割
def predict_segmentation(
    model: Union[TextModel, dict[int, ImageModel]],
    image_embeddings: dict,
    img_sizes: list[int],
    feature_type: str,
    patch_size: tuple[int, int],
    segmentation_mode: str,
) -> np.ndarray:
    """
    Language-guided (zero-shot) or vision-guided (few-shot) segmentation
    Perform segmentation at multiple image resolutions. Then, resize and aggregate into a single segmentation map.

    Args:
        model: Model used for the segmentation
               A single text embedding model if segmentation_mode='language'.
               A dictionary of multi-resolution image embedding models if segmentation_mode='vision'.
        image_embeddings: Dictionary of all embeddings
        img_sizes: List of image resolutions of the embeddings to use
        feature_type: 'clip' or 'gem'
        patch_size: Patch size used in the GEM/CLIP model
        segmentation_mode: 'language': Zero-shot language-guided segmentation
                           'vision': Few-shot vision-guided segmentation

    Returns:
        Segmentation scores of dimension (batch_size, height, width)
        height and width correspond to the highest resolution segmentation
    """
    '''
    assert：Python 中的断言关键字，用于在调试阶段检查某个条件是否为真。如果条件为假，程序会抛出 AssertionError 异常
    (   segmentation_mode == "vision"
        and isinstance(model, dict)
        and all(isinstance(x, ImageModel) for x in model.values())
    检查 segmentation_mode 是否为 "vision"。
    同时检查 model 是否是字典类型。
    再次检查字典中的每个值是否都是 ImageModel 类型的对象。
    '''
    # 参数校验：检查 segmentation_mode 是否为 "language" 或 "vision"。
    # 如果是 "language" 模式，确保 model 是 TextModel 类型。
    # 如果是 "vision" 模式，确保 model 是一个字典，且字典中的值都是 ImageModel 类型
    assert (segmentation_mode == "language" and isinstance(model, TextModel)) or (
        segmentation_mode == "vision"
        and isinstance(model, dict)
        and all(isinstance(x, ImageModel) for x in model.values())
    )

    # Anomaly segmentation at multiple resolutions 多分辨率分割
    segmentations_multi_resolution = []
    for img_size in img_sizes:
        # 获取对应分辨率的图像嵌入向量
        patch_embeddings = retrieve_image_embeddings(
            image_embeddings,
            img_size=img_size,
            feature_type=feature_type,
            token_type="patch",
        )
        if segmentation_mode == "language":
            segmentations_flatten = text_image_matching(model, patch_embeddings)
        elif segmentation_mode == "vision":
            segmentations_flatten = image_image_matching(
                model[img_size], patch_embeddings
            )
        else:
            raise ValueError(
                f"segmentation_mode can only be set to 'language' or 'vision'. ({segmentation_mode=})"
            )
        segmentations_single_resolution = segmentations_flatten.reshape(
            (
                -1,
                img_size // patch_size[0],
                img_size // patch_size[1],
            )
        )
        # 将匹配结果重塑为合适的形状，并存储在 segmentations_multi_resolution 列表中。
        segmentations_multi_resolution.append(segmentations_single_resolution)

    # Resize and aggregate segmentation of multiple resolutions
    # 调用 resize_and_aggregate_segmentations 函数，将不同分辨率下的分割结果调整到最高分辨率，并进行聚合。返回最终的分割分数矩阵
    return resize_and_aggregate_segmentations(
        segmentations_multi_resolution,
        max(img_sizes) // patch_size[0],  # Resize to largest resolution
    )
