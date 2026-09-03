from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from db_api import get_train_dataset

_rf_model = None
_metrics = {}

def train_rf():
    """训练随机森林质量分类模型（不平衡数据加强版）"""
    X, y = get_train_dataset()

    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("测试集失效样本数：", int(y_test.sum()), "/", len(y_test))

    # 对训练集做SMOTE过采样，人工扩充失效样本
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # 模型加强对失效类的权重
    model = RandomForestClassifier(
        n_estimators=150,
        class_weight={0: 1, 1: 10},  # 给失效类10倍权重
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train_bal, y_train_bal)

    # 取失效类概率
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 进一步降低阈值，工业场景优先保召回
    threshold = 0.15
    y_pred = (y_proba >= threshold).astype(int)

    print("失效概率最大值：", round(y_proba.max(), 4))
    print("当前阈值：", threshold)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    global _rf_model, _metrics
    _rf_model = model
    _metrics = metrics
    return metrics


def predict_quality(batch_id):
    """预测单批次质量标签与置信度"""
    from db_api import get_single_batch_features
    if _rf_model is None:
        train_rf()
    X = get_single_batch_features(batch_id)

    threshold = 0.15  # 和训练阈值完全一致
    fail_proba = _rf_model.predict_proba(X)[0][1]
    label = 1 if fail_proba >= threshold else 0
    confidence = float(fail_proba if label == 1 else 1 - fail_proba)

    return label, confidence

