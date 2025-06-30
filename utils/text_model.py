from pathlib import Path
from typing import List, Union

import gem
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from utils.adapter import ImageHead, TextHead
from utils.embeddings import extract_all_text_embeddings
import torch

class TextModel:
    pass


class AverageTextModel(TextModel):
    """
    Compute a mean text embedding for each normal and abnormal group separately.
    Then compute cosine similarity between image embedding and each mean text embedding.
    Do a softmax to get the probs.
    分别计算正常组和异常组的平均文本嵌入。
    然后计算图像嵌入与每个平均文本嵌入之间的余弦相似度。
    最后通过softmax计算概率。
    """
    # 定义了一个名为 fit 的方法，分别计算正常组和异常组的平均文本嵌入
    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.mean_text_embbeddings = np.stack(
            [
                np.mean(text_embeddings[text_labels == 0], axis=0),
                np.mean(text_embeddings[text_labels == 1], axis=0),
            ]
        )
    # 定义了一个名为 predict_proba 的方法，用于计算图像嵌入向量与预定义的文本嵌入向量之间的相似度，并返回每个图像属于各个类别的概率
    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        # 乘以100.0是为了放大模型的输出值，从而提高不同类别的得分差距，使模型的预测更加确定和稳定。这在实际应用中可以提高模型的性能和鲁棒性
        # #
        # with torch.no_grad():
        #     image_head = ImageHead(feature_dim = 640, out_dim = 640)
        #     text_head = TextHead(feature_dim = 640, out_dim = 640)
        #     image_embeddings, t_ = image_head(image_embeddings, self.mean_text_embbeddings)
        #     text_model = text_head(self.mean_text_embbeddings)
        #     text_model = 0.9 * text_model + 0.1 * t_
        #
        # logits = 100.0 * image_embeddings @ text_model.T

        logits = 100.0 * image_embeddings @ self.mean_text_embbeddings.T
        return softmax(logits, axis=1)


class SoftmaxTextModel(TextModel):
    """
    Compute cosine similarity between image embedding and each prompt in the prompt ensemble.
    Then do a softmax across all prompts.
    Add up probs for normal and abnormal group separately
    计算图像嵌入与提示集合中每个提示之间的余弦相似度。
    然后对所有提示进行softmax操作。
    分别累加正常组和异常组的概率。
    """

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.text_embeddings = text_embeddings
        self.text_labels = text_labels.astype(bool)

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        logits = 100.0 * image_embeddings @ self.text_embeddings.T
        probs = softmax(logits, axis=1)
        # 计算正常和异常概率 ~self.text_labels 表示不在 text_labels 中的索引。self.text_labels 表示在 text_labels 中的索引。sum(axis=1) 表示对每一行求和
        normal_probs = probs[:, ~self.text_labels].sum(axis=1)
        abnormal_probs = probs[:, self.text_labels].sum(axis=1)
        # np.stack 将 normal_probs 和 abnormal_probs 按列堆叠。axis=-1 表示按最后一维堆叠
        return np.stack([normal_probs, abnormal_probs], axis=-1)


class MaxTextModel(TextModel):
    """
    For each image embedding, find its nearest text embedding for each normal and abnormal group by max cosine similarity.
    Use the cosine similarity wrt the nearest text embedding for each normal and abnormal group.
    Do a softmax to get the probs
    对于每个图像嵌入，分别找到其在正常组和异常组中最相似的文本嵌入（通过最大余弦相似度）。
    使用相对于每个正常组和异常组中最相似的文本嵌入的余弦相似度。
    通过softmax计算概率。
    """

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.text_embeddings = text_embeddings
        self.text_labels = text_labels.astype(bool)

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        logits = 100.0 * image_embeddings @ self.text_embeddings.T
        normal_max = logits[:, ~self.text_labels].max(axis=1)
        abnormal_max = logits[:, self.text_labels].max(axis=1)
        return softmax(
            np.stack([normal_max, abnormal_max], axis=-1),
            axis=-1,
        )


class SupervisedModel(TextModel):
    """
    Train a binary supervised model on the text embeddings. Use the trained model to do inference on image embeddings.
    训练一个基于文本嵌入的二分类监督模型。使用训练好的模型对图像嵌入进行推理。
    """

    def __init__(self, model_type):
        # 如果 model_type 是 "lr"，则实例化 LogisticRegression 模型
        if model_type == "lr":
            self.model = LogisticRegression(
                solver="liblinear", penalty="l2", n_jobs=-1, random_state=42
            )
        elif model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=640,
                max_iter=200,
                early_stopping=True,
                random_state=42,
                verbose=True,
            )
        elif model_type == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="cosine", n_jobs=-1
            )
        elif model_type == "rf":
            self.model = RandomForestClassifier(n_jobs=-1, random_state=42)
        elif model_type == "xgboost":
            self.model = XGBClassifier(
                objective="binary:logistic", n_jobs=-1, random_state=42
            )
        else:
            raise f"Unknown {model_type=}"

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        self.model.fit(text_embeddings, text_labels)

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(image_embeddings)

# GmmModel 类用于构建一个基于高斯混合模型（GMM）的文本分类模型。该模型可以用于区分正常文本和异常文本，并且可以对图像嵌入进行推理，预测其属于正常组或异常组的概率
# gmmModel 类封装了从数据预处理（PCA降维）、模型训练（GMM聚类）到推理（计算概率）的完整流程
class GmmModel(TextModel):
    def __init__(
        self, gmm_components, covariance_type, run_pca=True, pca_components=0.9
    ):
        # PCA（主成分分析）：如果 run_pca 为 True，则使用指定的 pca_components 进行降维；否则，使用 FunctionTransformer 作为占位符。
        normal_pca = (
            PCA(n_components=pca_components, random_state=42)
            if run_pca
            else FunctionTransformer()
        )
        abnormal_pca = (
            PCA(n_components=pca_components, random_state=42)
            if run_pca
            else FunctionTransformer()
        )
        # GMM（高斯混合模型）：使用指定的 gmm_components 和 covariance_type 进行聚类。
        normal_gmm = GaussianMixture(
            n_components=gmm_components,
            covariance_type=covariance_type,
            random_state=42,
        )
        abnormal_gmm = GaussianMixture(
            n_components=gmm_components,
            covariance_type=covariance_type,
            random_state=42,
        )
        # 分别创建 normal_pipe 和 abnormal_pipe，每个管道包含 PCA 和 GMM 两个步骤
        self.normal_pipe = Pipeline([("pca", normal_pca), ("gmm", normal_gmm)])
        self.abnormal_pipe = Pipeline([("pca", abnormal_pca), ("gmm", abnormal_gmm)])

    def fit(self, text_embeddings: np.ndarray, text_labels: np.ndarray):
        # 将 text_labels 转换为布尔数组 mask。使用 mask 的反向选择正常样本，训练 normal_pipe。使用 mask 选择异常样本，训练 abnormal_pipe。
        mask = np.array(text_labels, dtype=bool)
        self.normal_pipe.fit(text_embeddings[np.logical_not(mask)])
        self.abnormal_pipe.fit(text_embeddings[mask])

    def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
        normal_logprobs = self.normal_pipe.score_samples(image_embeddings)
        abnormal_logprobs = self.abnormal_pipe.score_samples(image_embeddings)
        logprobs = np.stack([normal_logprobs, abnormal_logprobs], axis=-1)
        return softmax(logprobs, axis=-1)


def get_text_model(model_type):
    if model_type == "average":
        return AverageTextModel()
    elif model_type == "softmax":
        return SoftmaxTextModel()
    elif model_type == "max":
        return MaxTextModel()
    elif model_type in ["lr", "mlp", "knn", "rf", "xgboost"]:
        return SupervisedModel(model_type)
    elif model_type == "gmm":
        return GmmModel(
            gmm_components=10,
            covariance_type="spherical",
            run_pca=True,
            pca_components=0.9,
        )
    else:
        raise f"Unknown {model_type=}"


def build_text_model(
    gem_model: gem.gem_wrapper.GEMWrapper,
    prompt_paths: List[Union[Path, str]],
    classname: str,
    text_model_type: str,
) -> TextModel:    # -> TextModel：表示 build_text_model 函数的返回值类型是 TextModel。
    # Text embeddings 提取文本嵌入和标签
    text_embeddings, text_labels = extract_all_text_embeddings(
        prompt_paths,
        gem_model,
        classname=classname,
    )
    # Text model  获取文本模型
    text_model = get_text_model(model_type=text_model_type)
    # 训练文本模型
    text_model.fit(text_embeddings, text_labels)

    return text_model


