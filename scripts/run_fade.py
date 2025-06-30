

from collections import defaultdict
from pathlib import Path

import click
import cv2
# import gem
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from pkg_resources import require
from tqdm import tqdm
import sys
import os
import torch.nn.functional as F

sys.path.append('/home/hy/wsc/BMVC-FADE-main')
# 假设 datasets 目录在 /home/wsc/BMVC-FADE-main/datasets

from datasets.mvtec_3d import MVTec_3DDataset
from datasets.base import DatasetSplit, BaseDataset
from datasets.mvtec import MVTecDataset
from datasets.utils import undo_transform, min_max_normalization
from datasets.visa import VisADataset
from datasets.mpdd import MPDDDataset
from datasets.btad import BTADDataset
from datasets.Brain import BrainDataset
from datasets.Liver import LiverDataset

from datasets.Retina_RESC import Retina_RESCDataset
from datasets.Retina_oct import Retina_OCTDataset
from evaluation.utils import compute_and_store_final_results, evaluation
from utils.anomaly_detection import predict_classification, predict_segmentation
from utils.embeddings import extract_image_embeddings, extract_image_embeddings_one, extract_image_embeddings_add

from utils.image_model import (
    extract_ref_patch_embeddings,
    build_image_models,
    extract_query_patch_embeddings,
    combine_patch_embeddings,
)
from utils.plots import plot_segmentation_images
from utils.text_model import build_text_model


# 定义了一个名为 load_dataset 的函数，用于加载数据集。函数根据传入的 dataset_name 参数选择不同的数据集类实例化并返回。
def load_dataset(dataset_name: str, dataset_source: str, **kwargs) -> BaseDataset:
    if dataset_name == "mvtec":
        return MVTecDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "visa":
        return VisADataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "mpdd":
        return MPDDDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "btad":
        return BTADDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "mvtec_3d":
        return MVTec_3DDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "Brain":
        return BrainDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "Liver":
        return LiverDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "Retina_RESC":
        return Retina_RESCDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    elif dataset_name == "Retina_oct":
        return Retina_OCTDataset(
            source=Path(dataset_source),
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid {dataset_name=}")

# 定义了一个名为 load_classnames 的函数，功能是根据传入的数据集名称返回相应的类别名称列表
def load_classnames(dataset_name: str) -> list:
    if dataset_name == "mvtec":
        return MVTecDataset.CLASSNAMES
    elif dataset_name == "visa":
        return VisADataset.CLASSNAMES
    elif dataset_name == "mpdd":
        return MPDDDataset.CLASSNAMES
    elif dataset_name == "btad":
        return BTADDataset.CLASSNAMES
    elif dataset_name == "mvtec_3d":
        return MVTec_3DDataset.CLASSNAMES
    elif dataset_name == "Brain":
        return BrainDataset.CLASSNAMES
    elif dataset_name == "Liver":
        return LiverDataset.CLASSNAMES
    elif dataset_name == "Retina_RESC":
        return Retina_RESCDataset.CLASSNAMES
    elif dataset_name == "Retina_oct":
        return Retina_OCTDataset.CLASSNAMES
    else:
        raise ValueError(f"Invalid {dataset_name=}")

# click 库来定义一个命令行接口（CLI），允许用户通过命令行参数来配置和运行一个实验
@click.command()
@click.option("--dataset-name", type=click.Choice(["mvtec", "visa", "mpdd", "btad", "mvtec_3d","Brain", "Liver", "Retina_RESC", "Retina_oct"]), default="mvtec")
@click.option("--dataset-source", type=str, required=True, default="/home/wsc/BMVC-FADE-main/data/mvtec")
@click.option(
    "--experiment-name",
    type=str,
    default="mvtec/fewshot/cm_both_sm_both/img_size_448/1shot/seed0",
)
@click.option("--model-name", type=str, default="ViT-B/16-plus-240")
@click.option(
    "--pretrained",
    type=str,
    # default="models/openclip/clip/vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
    default="/home/hy/wsc/BMVC-FADE-main/vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
    # default="/home/wsc/BMVC-FADE-main/ViT-L-14-336px.pt",
)
@click.option(
    "--classification-mode",
    type=click.Choice(["none", "language", "vision", "both"]),
    default="both",
)
@click.option(
    "--segmentation-mode",
    type=click.Choice(["none", "language", "vision", "both"]),
    default="both",
)
@click.option(
    "--language-classification-feature",
    type=click.Choice(["clip", "gem"]),
    default="clip",
    help="Feature used for language-guided anomaly classification",
)
@click.option(
    "--language-segmentation-feature",
    type=click.Choice(["clip", "gem"]),
    default="gem",
    help="Feature used for language-guided anomaly segmentation",
)
@click.option(
    "--vision-feature",
    type=click.Choice(["clip", "gem"]),
    default="gem",
    help="Feature used for vision-guided anomaly classification and segmentation"
    "Note that vision-guided classification and segmentation use the same feature",
)
@click.option(
    "--vision-segmentation-multiplier",  # 乘以视觉引导的分割图以校准其上限值接近 1 的倍数
    type=float,
    default=3.5,
    help="A number multiplied to the vision-guided segmentation map to calibrate its upper bound value to be around 1",
)
@click.option(
    "--vision-segmentation-weight",
    type=click.FloatRange(0.0, 1.0),
    default=0.85,
    help="Weighting w given to the vision-guided segmentation map"
    "Only used when segmentation_mode='both'"
    "Segmentations are merged by: (1-w) * language_segmentation + w * vision_segmentation",
)
@click.option(
    "--use-query-img-in-vision-memory-bank/--no-use-query-img-in-vision-memory-bank",
    type=bool,
    default=False,
    help="Whether to use the query image patch embeddings to build the memory bank for vision-guided anomaly "
    "classification and segmentation."
    "Only used when classification_mode or segmentation_mode is set to 'vision' or 'both'",
)
@click.option(
    "--classification-img-size",
    type=int,
    default=240,
    help="Input image size of classification model",
)
@click.option(
    "--segmentation-img-sizes",
    type=str,
    # default="240,448,896",
    default="240,448,896",
    help="Input image sizes of segmentation models",
)
@click.option(
    "--eval-img-size",
    type=int,
    default=448,
    help="Image size used for evaluation and visualisation",
)
@click.option("--square/--no-square", type=bool, default=True)
@click.option(
    "--text-model-type",
    type=click.Choice(
        ["average", "softmax", "max", "lr", "mlp", "knn", "rf", "xgboost", "gmm"]
    ),
    default="average",
)
@click.option(
    "--shots",
    type=int,
    default=1,
    help="Number of reference images for few-shot detection. "
    "Only used when classification_mode or segmentation_mode is set to 'vision' or 'both'",
)
@click.option(
    "--seed",
    type=int,
    default=111,
    help="Random seed for sampling the few-shot reference images",
)
@click.option(
    "--normalize-segmentations/--no-normalize-segmentations", type=bool, default=False
)
@click.option("--save-visualization/--no-save-visualization", type=bool, default=True)
@click.option("--save-segmentation/--no-save-segmentation", type=bool, default=True)
def main(
    dataset_name: str,
    dataset_source: str,
    experiment_name: str,
    model_name: str,
    pretrained: str,
    classification_mode: str,
    segmentation_mode: str,
    language_classification_feature: str,
    language_segmentation_feature: str,
    vision_feature: str,
    vision_segmentation_multiplier: float,
    vision_segmentation_weight: float,
    use_query_img_in_vision_memory_bank: bool,
    classification_img_size: int,
    segmentation_img_sizes: str,
    eval_img_size: int,
    square: bool,
    text_model_type: str,
    shots: int,
    seed: int,
    normalize_segmentations: bool,
    save_visualization: bool,
    save_segmentation: bool,
):
    print("Start processing...")
    # 创建一个名为 result 的目录，用于保存实验结果。
    # parents=True 表示如果父目录不存在，也会自动创建。
    # exist_ok=True 表示如果目录已经存在，不会抛出异常。
    result_destination = Path(experiment_name) / "result"
    result_destination.mkdir(parents=True, exist_ok=True)
    # 如果 save_visualization 为 True，则创建一个名为 images 的目录，用于保存可视化结果
    if save_visualization:
        image_destination = Path(experiment_name) / "images"
        image_destination.mkdir(parents=True, exist_ok=True)
    # 如果 save_segmentation 为 True，则创建一个名为 segmentations 的目录，用于保存分割结果
    if save_segmentation:
        seg_destination = Path(experiment_name) / "segmentations"
        seg_destination.mkdir(parents=True, exist_ok=True)
    # 定义用于异常分类的文本提示文件路径。use_classname_in_prompt_classification 表示是否在提示中使用类名
    text_prompt_path = "/home/hy/wsc/BMVC-FADE-main/prompts"
    # Text prompts used for anomaly classification
    prompt_paths_classification = [
        f"{text_prompt_path}/winclip_prompt.json",
        # f"{text_prompt_path}/chatgpt3.5_prompt6_add.json",
        # f"{text_prompt_path}/chatgpt3.5_prompt_add2.json",
    ]
    use_classname_in_prompt_classification = True
    # 定义用于异常分割的文本提示文件路径。use_classname_in_prompt_segmentation 表示是否在提示中使用类名
    # Text prompts used for anomaly segmentation
    prompt_paths_segmentation = [
        f"{text_prompt_path}/winclip_prompt.json",

        # f"{text_prompt_path}/manual_prompt.json",
        # f"{text_prompt_path}/manual_prompt_with_classname.json",
        # f"{text_prompt_path}/manual_prompt_with_classname2.json",
        # f"{text_prompt_path}/winclip_prompt_aug_size_position.json",
        # f"{text_prompt_path}/flexible_medical_prompts.json",
        # f"{text_prompt_path}/april_gan.json",
        f"{text_prompt_path}/chatgpt3.5_prompt1.json",
        f"{text_prompt_path}/chatgpt3.5_prompt2.json",
        f"{text_prompt_path}/chatgpt3.5_prompt3.json",
        f"{text_prompt_path}/chatgpt3.5_prompt4.json",
        f"{text_prompt_path}/chatgpt3.5_prompt5.json",
        # f"{text_prompt_path}/chatgpt3.5_prompt6_add.json",
        # f"{text_prompt_path}/chatgpt3.5_prompt_add2.json",
    ]
    use_classname_in_prompt_segmentation = False

    model_cache_dir = "models"
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    # 分割图像大小
    segmentation_img_sizes = [int(i) for i in segmentation_img_sizes.split(",")]
    # 将所有参数和设置保存到一个配置字典中，以便后续使用
    config = {
        "dataset_name": dataset_name,
        "dataset_root": dataset_source,
        "experiment_name": experiment_name,
        "model_name": model_name,
        "pretrained": pretrained,
        "model_cache_dir": model_cache_dir,
        "classification_mode": classification_mode,
        "segmentation_mode": segmentation_mode,
        "language_classification_feature": language_classification_feature,
        "language_segmentation_feature": language_segmentation_feature,
        "vision_feature": vision_feature,
        "vision_segmentation_multiplier": vision_segmentation_multiplier,
        "vision_segmentation_weight": vision_segmentation_weight,
        "use_query_img_in_vision_memory_bank": use_query_img_in_vision_memory_bank,
        "classification_img_size": classification_img_size,
        "segmentation_img_sizes": segmentation_img_sizes,
        "eval_img_size": eval_img_size,
        "square": square,
        "prompt_paths_classification": prompt_paths_classification,
        "use_classname_in_prompt_classification": use_classname_in_prompt_classification,
        "prompt_paths_segmentation": prompt_paths_segmentation,
        "use_classname_in_prompt_segmentation": use_classname_in_prompt_segmentation,
        "text_model_type": text_model_type,
        "shots": shots,
        "seed": seed,
        "normalize_segmentations": normalize_segmentations,
        "save_visualization": save_visualization,
        "save_segmentation": save_segmentation,
        "device": device,
    }
    # 将配置字典保存为 YAML 文件
    with open(result_destination / "config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    # init model gem.create_gem_model：这是一个函数调用，用于创建一个 GEM 模型。GEM（Generalized Embedding Model）可能是一个用于特征提取或异常检测的模型
    from model import gem
    gem_model = gem.create_gem_model(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    # 一个空列表，用于收集实验结果
    result_collect = []
    # 一个 defaultdict，用于存储绘图数据。defaultdict(list) 会自动为不存在的键创建一个空列表，方便后续数据的追加
    plot_data = defaultdict(list)
    # load_classnames：这是一个函数调用，用于加载数据集中类的名称。dataset_name：指定数据集的名称，例如 "mvtec" 或 "visa"
    classnames = load_classnames(dataset_name)

    for classname in classnames:
        print(f"Processing '{classname}'")
        # load image dataset
        # load_dataset：加载数据集，参数包括数据集名称、数据源路径、类名、图像大小、是否裁剪为正方形和数据集分割方式（测试集）
        dataset = load_dataset(
            dataset_name=dataset_name,
            dataset_source=dataset_source,
            classname=classname,
            resize=list(
                {classification_img_size, *segmentation_img_sizes, eval_img_size}
            ),
            square=square,
            split=DatasetSplit.TEST,
        )
        #  DataLoader：创建数据加载器，设置批量大小、是否打乱、工作线程数、预取因子和内存锁定
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
        )

        # Build text model using prompts for language-guided anomaly detection
        # uild_text_model：构建文本模型，参数包括 GEM 模型、提示路径、类名和文本模型类型。classname：类名处理，根据配置决定是否替换下划线
        if classification_mode == "language" or classification_mode == "both":
            classification_text_model = build_text_model(
                gem_model=gem_model,
                prompt_paths=prompt_paths_classification,
                classname=(
                    classname.replace("_", " ")
                    if use_classname_in_prompt_classification
                    else "object"
                ),
                text_model_type=text_model_type,
            )
        if segmentation_mode == "language" or segmentation_mode == "both":
            segmentation_text_model = build_text_model(
                gem_model=gem_model,
                prompt_paths=prompt_paths_segmentation,
                classname=(
                    classname.replace("_", " ")
                    if use_classname_in_prompt_segmentation
                    else "object"
                ),
                text_model_type=text_model_type,
            )

        # Build image model using reference images for vision-guided anomaly detection
        # ref_dataset：加载参考数据集，用于提取参考图像的嵌入。extract_ref_patch_embeddings：提取参考图像的嵌入，参数包括参考数据集、GEM 模型、视觉特征、样本数量、随机种子和设备
        ref_patch_embeddings = None
        if (
            classification_mode == "vision"
            or classification_mode == "both"
            or segmentation_mode == "vision"
            or segmentation_mode == "both"
        ) and shots > 0:
            ref_dataset = load_dataset(
                dataset_name=dataset_name,
                dataset_source=dataset_source,
                classname=classname,
                resize=segmentation_img_sizes,
                square=square,
                split=DatasetSplit.TRAIN,
            )
            ref_patch_embeddings = extract_ref_patch_embeddings(
                ref_dataset,
                gem_model,
                vision_feature,
                shots,
                seed,
                device,
            )

        anomaly_scores = []
        ground_truth_scores = []
        anomaly_segmentations = []
        ground_truth_segmentations = []
        # 数据加载和处理  tqdm：进度条，显示数据加载进度。plot_data：收集数据集中的关键信息，用于后续绘图
        for data in tqdm(dataloader):
            for key in [
                "classname",
                "anomaly",
                "is_anomaly",
                "image_name",
                "image_path",
                "mask_path",
            ]:
                plot_data[key].append(
                    data[key].tolist()
                    if isinstance(data[key], torch.Tensor)
                    else data[key]
                )

            # Extract image embeddings 提取图像嵌入
            # img_sizes：图像大小列表，包括分类图像大小和分割图像大小。
            img_sizes = list({classification_img_size, *segmentation_img_sizes})
            # multiscale_images：多尺度图像字典。
            multiscale_images = {sz: data["image"][sz] for sz in img_sizes}
            #  ----------------------------------------------------
            # # 打印 multiscale_images 中的值
            # print("Multiscale Images:")
            # for img_size, images_tensor in multiscale_images.items():
            #     print(f"Image Size: {img_size}, Shape: {images_tensor.shape}")
            #     --------------------------------------------------------------------
            # extract_image_embeddings：提取图像嵌入，参数包括多尺度图像、GEM 模型和设备
            image_embeddings = extract_image_embeddings_add(
                multiscale_images, gem_model, device
            )
# # -------------------------------------------------------------------
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             clip_inplanted_model = CLIP_Inplanted(gem_model, features=[6, 12, 18, 24]).to(device)
#             clip_inplanted_model.eval()
#             image_embeddings = clip_inplanted_model(multiscale_images, device)
#             features=[4, 8, 12, 16, 20],
#             gem_clip_inplanted = GEM_CLIP_Inplanted(gem_model, features)
#             image_embeddings = gem_clip_inplanted(multiscale_images, device)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # gem_model.eval()
            # clip_inplanted_model = CLIP_Inplanted(gem_model, features=[6, 12, 18, 24]).to(device)
            # clip_inplanted_model.eval()
            # # 将输入数据移动到与模型相同的设备
            # for img_size in multiscale_images:
            #     multiscale_images[img_size] = multiscale_images[img_size].to(device)
            # # 使用 clip_inplanted_model 提取图像嵌入
            # features = {}
            # with torch.no_grad():  # 不需要计算梯度
            #     multiscale_outputs = clip_inplanted_model(multiscale_images)
            #
            # for img_size, (pooled, seg_patch_tokens, det_patch_tokens) in multiscale_outputs.items():
            #     # 归一化特征
            #     pooled_gem = F.normalize(pooled, dim=-1).detach().cpu().numpy()
            #     pooled_clip = F.normalize(pooled, dim=-1).detach().cpu().numpy()
            #
            #     # 组织特征字典
            #     features[img_size] = {
            #         "gem": {
            #             "cls": pooled_gem,
            #             "patch": [tokens.detach().cpu().numpy() for tokens in seg_patch_tokens],
            #         },
            #         "clip": {
            #             "cls": pooled_clip,
            #             "patch": [tokens.detach().cpu().numpy() for tokens in det_patch_tokens],
            #         },
            #     }
            # # 在这里可以继续处理 features 字典
            # image_embeddings = features
#------------------------------------------------------------------------------------
            # Language-guided anomaly classification 语言引导的异常分类
            language_guided_scores = None
            if classification_mode == "language" or classification_mode == "both":
                # predict_classification：预测分类分数，参数包括文本模型、图像嵌入、图像大小和特征类型
                language_guided_scores = predict_classification(
                    text_model=classification_text_model,
                    image_embeddings=image_embeddings,
                    img_size=classification_img_size,
                    feature_type=language_classification_feature,
                )

            # Language-guided anomaly segmentation 语言引导的异常分割
            language_guided_maps = None
            if segmentation_mode == "language" or segmentation_mode == "both":
                # predict_segmentation：预测分割图，参数包括模型、图像嵌入、图像大小、特征类型、补丁大小和分割模式
                language_guided_maps = predict_segmentation(
                    model=segmentation_text_model,
                    image_embeddings=image_embeddings,
                    img_sizes=segmentation_img_sizes,
                    feature_type=language_segmentation_feature,
                    patch_size=gem_model.model.visual.patch_size,
                    segmentation_mode="language",
                )

            # Vision-guided anomaly segmentation and classification 视觉引导的异常分割和分类
            vision_guided_scores = None
            vision_guided_maps = None
            if (
                classification_mode == "vision"
                or classification_mode == "both"
                or segmentation_mode == "vision"
                or segmentation_mode == "both"
            ):
                # Build image models using patch embeddings from reference and/or query images
                # extract_query_patch_embeddings：提取查询图像的嵌入
                query_patch_embeddings = None
                if use_query_img_in_vision_memory_bank:
                    query_patch_embeddings = extract_query_patch_embeddings(
                        image_embeddings, segmentation_img_sizes, vision_feature
                    )
                #     combine_patch_embeddings：结合参考图像和查询图像的嵌入
                if ref_patch_embeddings and query_patch_embeddings:
                    train_patch_embeddings = combine_patch_embeddings(
                        ref_patch_embeddings, query_patch_embeddings
                    )
                else:
                    train_patch_embeddings = (
                        ref_patch_embeddings or query_patch_embeddings
                    )
                assert (
                    train_patch_embeddings
                ), "You cannot set shots=0 AND use_query_img_in_vision_memory_bank=False"
                # build_image_models：构建图像模型
                image_models = build_image_models(
                    train_patch_embeddings, use_query_img_in_vision_memory_bank
                )

                # Vision-guided anomaly segmentation .predict_segmentation：预测分割图，参数同上。
                vision_guided_maps = predict_segmentation(
                    model=image_models,
                    image_embeddings=image_embeddings,
                    img_sizes=segmentation_img_sizes,
                    feature_type=vision_feature,
                    patch_size=gem_model.model.visual.patch_size,
                    segmentation_mode="vision",
                )
                # 乘以缩放因子 vision_segmentation_multiplier 的目的是为了校准视觉引导的分割图的值域，使其更接近于期望的范围（通常是 [0, 1]）。这是因为不同的模型或特征提取方法可能会导致分割图的值域不同，通过乘以一个适当的缩放因子，可以使分割图的值更加合理，从而提高后续处理（如归一化、可视化等）的效果。
                vision_guided_maps *= vision_segmentation_multiplier

                # Vision-guided anomaly classification .vision_guided_scores：计算视觉引导的分类分数
                if classification_mode == "vision" or classification_mode == "both":
                    vision_guided_scores = np.max(vision_guided_maps, axis=(1, 2))

            # Final classification scores  最终分类分数
            # scores：根据分类模式选择最终分类分数
            scores = None
            if classification_mode != "none":
                if classification_mode == "language":
                    scores = language_guided_scores
                elif classification_mode == "vision":
                    scores = vision_guided_scores
                elif classification_mode == "both":
                    al = 0.50
                    scores = (language_guided_scores + vision_guided_scores) / 2
                    # scores = (1 - al) * language_guided_scores + al * vision_guided_scores
                #     np.clip：将分数裁剪到 [0, 1] 范围内
                scores = np.clip(scores, 0, 1)
                # plot_data：收集分类分数和真实标签
                plot_data["image_anomaly_score"].append(scores.tolist())
                anomaly_scores.append(scores)
                ground_truth_scores.append(data["is_anomaly"])

            # Final segmentation maps 最终分割图
            # segmentations：根据分割模式选择最终分割图
            segmentations = None
            if segmentation_mode != "none":
                if segmentation_mode == "language":
                    segmentations = language_guided_maps
                elif segmentation_mode == "vision":
                    segmentations = vision_guided_maps
                elif segmentation_mode == "both":
                    segmentations = (
                        (1.0 - vision_segmentation_weight) * language_guided_maps
                        + vision_segmentation_weight * vision_guided_maps
                    )

                # Post-processing segmentation maps
                # min_max_normalization：对分割图进行归一化
                if normalize_segmentations:
                    segmentations = min_max_normalization(segmentations)
                #     np.clip：将分割图裁剪到 [0, 1] 范围内
                segmentations = np.clip(segmentations, 0, 1)
                segmentations = (segmentations * 255).astype("uint8")
                # save_segmentation：保存分割图
                if save_segmentation:
                    for seg, img_name in zip(segmentations, data["image_name"]):
                        img = Image.fromarray(seg).convert("RGB")
                        save_path = seg_destination / (img_name + ".png")
                        save_path.parent.mkdir(exist_ok=True, parents=True)
                        img.save(save_path)

                # Resize segmentation for evaluation and visualisation
                segmentations = np.array(
                    [
                        cv2.resize(seg, (eval_img_size, eval_img_size))
                        for seg in segmentations
                    ]
                )

                if save_visualization:
                    plot_data["vis_path"].append(
                        # plot_segmentation_images：保存可视化图像
                        plot_segmentation_images(
                            image_destination=image_destination,
                            image_names=data["image_name"],
                            images=data["image"][eval_img_size],
                            segmentations=segmentations,
                            anomaly_scores=scores,
                            masks=data["mask"][eval_img_size],
                            image_transform=lambda x: undo_transform(x, unorm=True),
                            mask_transform=lambda x: undo_transform(x, unorm=False),
                        )
                    )
                # anomaly_segmentations 和 ground_truth_segmentations：收集分割图和真实标签
                anomaly_segmentations.append(segmentations)
                ground_truth_segmentations.append(
                    data["mask"][eval_img_size][:, 0, :, :]
                )

        # Evaluations 计算各类评估指标（如 AUROC），返回一个包含评估结果的字典
        if classification_mode != "none":
            anomaly_scores = np.concatenate(anomaly_scores)
            ground_truth_scores = np.concatenate(ground_truth_scores)
        if segmentation_mode != "none":
            anomaly_segmentations = np.concatenate(anomaly_segmentations)
            ground_truth_segmentations = np.concatenate(ground_truth_segmentations)
        object_results = evaluation(
            ground_truth_scores,
            anomaly_scores,
            ground_truth_segmentations,
            anomaly_segmentations,
        )





        result_collect.append({"object_name": classname, **object_results})
        print(f"Object: {classname}")
        # 这表示对整张图像进行分类时的 AUROC 值。通常，这种评估是在图像级别上进行的，即每张图像被标记为正常或异常。
        # 用途：用来评估模型在区分整张图像是否包含异常情况的能力。较高的 AUROC 值表示模型能够较好地区分正常和异常图像。
        print(f"Full image AUROC: {object_results['full_image_auroc']:.2f}")
        # 这表示对图像中每个像素进行分类时的 AUROC 值。每个像素被预测为正常或异常。
        # 用途：用来评估模型在像素级别上检测异常的能力。这对于需要精确定位异常区域的应用非常重要，例如医学影像分析、工业缺陷检测等。
        print(f"Full pixel AUROC: {object_results['full_pixel_auroc']:.2f}")
        # 这专门针对异常像素的 AUROC 值。它关注的是模型在识别异常像素方面的性能。
        # 用途：用于评估模型在定位和识别图像中异常部分的能力。对于那些只关心异常区域的应用，这个指标尤其重要。
        print(f"Anomaly pixel AUROC: {object_results['anomaly_pixel_auroc']:.2f}")
        print("\n")
    # 获取当前时间并格式化为字符串
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 添加时间戳到文件名中
    # evaluation_results_filename = f"evaluation_results_{current_time}.csv"
    # plot_results_filename = f"plot_results_{current_time}.csv"
    # compute_and_store_final_results 函数：计算并存储最终的评估结果，返回一个数据表
    results_dt = compute_and_store_final_results(result_collect)
    results_dt.to_csv(result_destination / "evaluation_results.csv", index=False)
    # 保存 evaluation_results.csv 文件
    # results_dt = compute_and_store_final_results(result_collect)
    # results_dt.to_csv(result_destination / evaluation_results_filename, index=False)

    # 处理 plot_data 并保存 plot_results.csv 文件
    # Save results
    plot_data = {key: np.concatenate(plot_data[key]).tolist() for key in plot_data}
    plot_data = pd.DataFrame(plot_data)
    if "vis_path" not in plot_data:
        plot_data["vis_path"] = None

    plot_data.to_csv(result_destination / "plot_results.csv", index=False)



if __name__ == "__main__":
    main()
