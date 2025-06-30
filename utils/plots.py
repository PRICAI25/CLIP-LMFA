from pathlib import Path

import cv2
import matplotlib
import numpy as np
# matplotlib.use("Agg") 是 Matplotlib 库中用于设置后端的一个命令。
# 后端决定了 Matplotlib 如何与操作系统交互，以及如何渲染图形。
# Agg 后端是一个非交互式的后端，主要用于生成图像文件，而不是在屏幕上显示图形。这对于服务器环境或没有图形界面的环境中特别有用
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_plot(fig, save_path):
    # 调用 save_path.parent.mkdir(exist_ok=True, parents=True) 检查 save_path 的父目录是否存在，如果不存在则创建该目录。exist_ok=True 参数确保如果目录已经存在不会抛出异常，parents=True 参数确保所有必要的上级目录都会被创建
    save_path.parent.mkdir(exist_ok=True, parents=True)
    # 调用 fig.savefig(save_path, format="png") 将图形对象 fig 保存为 PNG 格式的文件，路径为 save_path。
    fig.savefig(save_path, format="png")

# 定义了一个名为 plot_segmentation_images 的函数，用于生成异常分割图像
def plot_segmentation_images(
    image_destination: Path,
    image_names,
    images,
    segmentations,
    anomaly_scores=None,
    masks=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
):
    """Generate anomaly segmentation images.

    Args:
        image_names: List[str] List of image names.
        images: List[np.ndarray] List of images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        masks: [List[np.ndarray]] List of ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
    """
    vis_paths = []
    if anomaly_scores is None:
        # 异常分数处理：如果 anomaly_scores 为 None，则为每个图像设置一个默认的 nan 值
        anomaly_scores = [np.nan for _ in range(len(image_names))]
    #     遍历图像列表：使用 zip 函数同时遍历 image_names, images, masks, anomaly_scores, segmentations。
    for image_name, image, mask, anomaly_score, segmentation in zip(
        image_names, images, masks, anomaly_scores, segmentations
    ):
        # 图像和掩码变换：对图像和掩码应用可选的变换函数
        image = image_transform(image)
        mask = mask_transform(mask)
        # 热力图生成：使用 cv2 库将分割结果转换为热力图，并与原始图像叠加
        heatmap = cv2.cvtColor(
            cv2.applyColorMap(segmentation, cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        superimposed = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        # 绘图展示：使用 matplotlib 创建一个包含四个子图的绘图窗口，分别显示原始图像、真实掩码、预测结果和叠加图像
        f, axes = plt.subplots(1, 4)
        axes[0].imshow(image)
        axes[0].set_title(image_name)
        axes[0].axis("off")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("GT")
        axes[1].axis("off")
        axes[2].imshow(segmentation, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title(f"Prediction\nImage-level score = {anomaly_score:.4f}")
        axes[2].axis("off")
        axes[3].imshow(superimposed)
        axes[3].set_title("Prediction overlaid")
        axes[3].axis("off")
        f.set_size_inches(3 * 4, 3)
        f.tight_layout()
        # 保存图像：调整布局并保存生成的图像到指定路径，记录保存路径
        save_plot(f, image_destination / (image_name + ".png"))
        plt.close()
        vis_paths.append(image_destination / (image_name + ".png"))
    return vis_paths
