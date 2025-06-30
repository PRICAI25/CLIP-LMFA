from collections import defaultdict

import gem
import numpy as np
import torch

import datasets.base
from utils.embeddings import extract_image_embeddings, retrieve_image_embeddings, extract_image_embeddings_add, \
    extract_image_embeddings_one
from utils.test_adapter import train_model
from utils.test_adapter_two import ImageHead_image


# 图像的记忆库
class ImageModel:
    """
    This is the image patch memory bank built from the reference images. predict_proba() computes the distance between
    a query patch and its kth nearest neighbour in the memory bank as the anomaly score.
    这是从参考图像构建的图像块内存库。predict_proba() 计算查询图像块与其在内存库中的第 k 个最近邻之间的距离作为异常得分
    """
    # 定义了一个类的初始化方法 __init__，该方法接受一个参数 kth，默认值为1
    def __init__(self, kth: int = 1):
        """
        Args:

            kth: predict_proba() will compute the distance to the kth nearest neighbour in the memory bank
            kth 参数用于指定在调用 predict_proba() 方法时，计算内存库中第 kth 个最近邻的距离。
        """
        # self.image_head = ImageHead_image(feature_dim=640, out_dim=640)
        self.kth = kth

    # 定义了一个名为 fit 的方法，用于接收训练图像的嵌入向量
    def fit(self, train_image_embeddings: np.ndarray):
        """
        Args:

            train_image_embeddings: Patch embeddings from training images.
                                    Numpy array of dimension (num_samples*num_patches, emb_dim)
            接收参数 train_image_embeddings，这是一个形状为 (num_samples * num_patches, emb_dim) 的 NumPy 数组，表示训练图像的补丁嵌入。
将接收到的嵌入向量存储在类实例的属性 self.train_image_embeddings 中。
        """
        self.train_image_embeddings = train_image_embeddings

    # 定义了一个 predict_proba 方法，用于计算图像嵌入的异常分数
    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        """
        Args:

            image_embeddings: Patch embeddings from query images.
                              Numpy array of dimension (num_samples*num_patches, emb_dim)

        Returns:
            Anomaly scores of dimension (num_samples*num_patches)
        """
        # print("image_embeddings.shape: ",image_embeddings.shape)  # 225 640
        num_patches, emb_dim = image_embeddings.shape
        # with torch.no_grad():
        #     # image = ImageHead_image(feature_dim= emb_dim, out_dim=emb_dim)
        #     image_embeddings = self.image_head(image_embeddings, self.train_image_embeddings)
        # # 计算余弦相似度：
        cosine_sim = image_embeddings @ self.train_image_embeddings.T
        # 计算异常分数
        anomaly_scores = 0.5 * (1 - cosine_sim)
        # np.partition 函数会部分排序数组，使得第 self.kth - 1 个位置之前的元素都不大于该位置的元素，而该位置之后的元素都不小于该位置的元素。这个操作不会完全排序数组，而是部分排序，从而提高效率
        # axis=1 表示沿着列方向进行部分排序，即对每一行进行部分排序  [:, self.kth - 1]这是一个切片操作，从部分排序后的数组中提取每一行的第 self.kth - 1 个元素。
        # 结果是一个一维数组，其中每个元素对应一个样本的第 k 个最近邻的距离。
        return np.partition(anomaly_scores, self.kth - 1, axis=1)[:, self.kth - 1]

# 定义了一个名为 extract_ref_patch_embeddings 的函数，用于从少量样本参考图像中提取多分辨率的补丁嵌入向量。
def extract_ref_patch_embeddings(
    ref_dataset: datasets.base.BaseDataset, # 这是一个数据集对象，继承自 datasets.base.BaseDataset。这个数据集包含了参考图像（通常是正常样本），用于提取特征嵌入
    gem_model: gem.gem_wrapper.GEMWrapper, # 这是一个 GEM 模型的实例，封装了多模态模型（如 CLIP）并提供了额外的功能，如特征提取、注意力机制等
    feature_type: str,
    shots: int,
    seed: int,
    device: str,
) -> dict[int, np.ndarray]:
    """
    Extract multi-resolution patch embeddings from the few-shot reference images

    Args:

        ref_dataset: Full reference dataset
        gem_model: GEM model to extract patch embeddings
        feature_type: 'clip' or 'gem'
        shots: Number of few-shot examples to be sampled from training dataset
        seed: Random seed used for random sampling few--shot examples
        device: gpu or cpu

    Returns:
        Dictionary of numpy arrays containing embeddings extracted at different resolution or img_size.
        Key of dict is the img_size and value is the patch embeddings with dimension (num_samples,
        num_patches, emb_dim)
    """
    # 创建一个 DataLoader 对象 ref_dataloader，用于从 ref_dataset 中加载数据。设置 batch_size=1，确保每次只加载一张图像。
    ref_dataloader = torch.utils.data.DataLoader(
        ref_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    # ref_dataloader 是一个数据加载器对象，通常用于从数据集中批量加载数据。iter(ref_dataloader) 创建一个迭代器，用于遍历数据加载器中的数据批次。
    # next(iter(ref_dataloader)) 从迭代器中获取下一个批次的数据。由于这是第一次调用 next，所以它会返回第一个批次的数据。
    data = next(iter(ref_dataloader))
    # data["image"] 是当前批次中所有图像的数据。通常，这会是一个包含图像张量的列表或张量。 img_sizes 现在是一个包含当前批次中所有图像的列表
    img_sizes = list(data["image"])
    # defaultdict 是 dict 类的一个子类，它提供了当访问的键不存在时的默认值。在这一行代码中，list 被用作工厂函数，这意味着当访问一个不存在的键时，
    # defaultdict 会自动为该键创建一个空列表作为其值
    patch_embeddings = defaultdict(list)
    # 设置随机种子 torch.manual_seed(seed)，确保采样的随机性是可重复的
    torch.manual_seed(seed)

    # 如果需要训练请加载下面的训练函数
    train_model(ref_dataloader, device, shots)
    # #
    # train_model(in_channels = 3, feature_dim = None, seq_length = None,  train_dataloader = ref_dataloader)

    # 使用 for 循环遍历 ref_dataloader，每次迭代获取一个批次的数据。对每个图像，提取其多尺度版本 multiscale_images。调用 extract_image_embeddings 函数，提取图像的嵌入向量 image_embeddings。
    for i, data in enumerate(ref_dataloader):
        # print(f"data:", data)
        multiscale_images = {sz: data["image"][sz] for sz in img_sizes}
        image_embeddings = extract_image_embeddings_add(
            multiscale_images, gem_model, device
        )
        # 将提取的嵌入向量按图像尺寸存储在字典 patch_embeddings 中。字典的键是图像尺寸，值是嵌入向量列表
        for img_size in img_sizes:
            patch_embeddings[img_size].append(
                retrieve_image_embeddings(
                    image_embeddings,
                    img_size=img_size,
                    feature_type=feature_type,
                    token_type="patch",
                )
            )
        # 当达到指定的样本数量 shots 后，使用 np.concatenate 将嵌入向量列表拼接成一个数组，并返回字典 patch_embeddings。
        if i + 1 == shots:
            break
    return {
        img_size: np.concatenate(patch_embeddings[img_size]) for img_size in img_sizes
    }

# 定义了一个名为 extract_query_patch_embeddings 的函数，用于从给定的图像嵌入中提取多分辨率的补丁嵌入
def extract_query_patch_embeddings(
    image_embeddings: dict[int, dict[str, dict[str, torch.Tensor]]],
    img_sizes: list,
    feature_type: str,
) -> dict[int, np.ndarray]:
    """
    Extract multi-resolution patch embeddings from the query image
    遍历 img_sizes：检查是否还有未处理的图像尺寸。
    调用 retrieve_image_embeddings：对于当前的图像尺寸，调用 retrieve_image_embeddings 函数，传入必要的参数。
    存储结果：将 retrieve_image_embeddings 返回的结果存储在字典中。
    继续遍历：继续检查下一个图像尺寸。
    结束：所有图像尺寸处理完毕，返回结果字典。
    """
    return {
        img_size: retrieve_image_embeddings(
            image_embeddings,
            img_size=img_size,
            feature_type=feature_type,
            token_type="patch",
        )
        for img_size in img_sizes
    }

# 定义了一个函数 combine_patch_embeddings，用于合并参考图像和查询图像的补丁嵌入向量
def combine_patch_embeddings(
    ref_patch_embeddings: dict[int, np.ndarray],
    query_patch_embeddings: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Combine the reference and query patch embeddings
    检查 ref_patch_embeddings 和 query_patch_embeddings 的键是否相同。如果不同，抛出异常
    """
    assert (
        ref_patch_embeddings.keys() == query_patch_embeddings.keys()
    ), "Different image scales in ref_patch_embeddings and query_patch_embeddings"
    return {
        img_size: np.concatenate(
            [ref_patch_embeddings[img_size], query_patch_embeddings[img_size]]
        )
        for img_size in ref_patch_embeddings
    }

# 定义了一个名为 build_image_models 的函数，用于构建多分辨率图像模型
def build_image_models(
    train_patch_embeddings: dict[int, np.ndarray],
    use_query_img_in_vision_memory_bank: bool,
) -> dict[int, ImageModel]:
    """
    Build multi-resolution image models using patch embeddings from reference and/or query images

    Args:

        train_patch_embeddings: Dictionary of numpy arrays containing embeddings extracted at different resolution or
                                img_size. Key of dict is the img_size and value is the patch embeddings with dimension
                                (num_samples, num_patches, emb_dim)
        use_query_img_in_vision_memory_bank: Whether the query image patch embeddings is used to build the memory bank
                                             for vision-guided anomaly classification and segmentation.

    Returns:
        Dictionary of ImageModel built from patch embeddings at different resolution or img_size.
        Key of dict is img_size and value is ImageModel
    """
    image_models = {}
    for img_size, patch_embeddings_single_resolution in train_patch_embeddings.items():
        num_samples, num_patches, emb_dim = patch_embeddings_single_resolution.shape
        # If query image is used to construct the memory bank, we need to find the distance to the 2nd nearest neighbour
        # because the 1st nearest neighbour will be the query patch itself.
        image_models[img_size] = (
            ImageModel(kth=2)
            if use_query_img_in_vision_memory_bank
            else ImageModel(kth=1)
        )
        # patch_embeddings_single_resolution.reshape((-1, emb_dim)) 作为输入数据传递给 fit 方法，用于训练 image_models[img_size] 模型。
        image_models[img_size].fit(
            patch_embeddings_single_resolution.reshape((-1, emb_dim))
        )
    return image_models
