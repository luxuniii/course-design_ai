from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from db_api import get_train_dataset

_rf_model = None
_metrics = {}

# 分类阈值：工业质量预警优先保证召回率（宁可误报、不可漏报）
# 调低阈值 -> 召回率上升、准确率下降；调高阈值 -> 反之
_THRESHOLD = 0.20


def train_rf():
    """
    训练随机森林质量分类模型

    评估方式：从训练集中按 8:2 随机划分出独立验证子集（stratify 保持失效比例），
    模型只用训练部分拟合，指标只在模型从未见过的验证子集上计算，
    因此得到的准确率/召回率反映真实泛化能力，而不是训练集自评估的假象。
    """
    X, y = get_train_dataset()

    # 8:2 随机划分训练/验证子集（内存中完成，不改动数据库原始数据）
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. SMOTE 对训练部分过采样，缓解失效样本极少导致的类别不平衡
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    # 2. 训练模型（限制深度/叶子，避免对 423 维噪声特征过拟合）
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_bal, y_bal)

    # 3. 在独立验证子集上评估，使用自定义阈值
    y_proba = model.predict_proba(X_eval)[:, 1]  # 取失效类概率
    y_pred = (y_proba >= _THRESHOLD).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "recall": float(recall_score(y_eval, y_pred)),
        "f1": float(f1_score(y_eval, y_pred)),
        "threshold": _THRESHOLD,
    }

    print("验证子集标签分布：", dict(y_eval.value_counts()))
    print("混淆矩阵（[[TN, FP], [FN, TP]]）：\n", confusion_matrix(y_eval, y_pred))
    print("阈值:", _THRESHOLD, " 准确率:", metrics["accuracy"], " 召回率:", metrics["recall"], " F1:", metrics["f1"])

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
    # 与训练评估使用完全相同的阈值
    fail_proba = _rf_model.predict_proba(X)[0][1]
    label = 1 if fail_proba >= _THRESHOLD else 0
    confidence = float(fail_proba if label == 1 else 1 - fail_proba)
    return label, confidence
