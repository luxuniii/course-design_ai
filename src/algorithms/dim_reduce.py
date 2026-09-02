import numpy as np
from sklearn.decomposition import PCA
from db_api import get_train_dataset

_pca_model = None

def train_pca(n_components=2):
    """训练PCA降维模型"""
    X, _ = get_train_dataset()
    pca = PCA(n_components=n_components)
    pca.fit(X)
    global _pca_model
    _pca_model = pca
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "n_components": n_components
    }

def reduce_single(batch_id):
    """对单批次数据降维，返回二维坐标"""
    from db_api import get_single_batch_features
    if _pca_model is None:
        train_pca()
    X = get_single_batch_features(batch_id)
    res = _pca_model.transform(X)[0]
    return float(res[0]), float(res[1])
