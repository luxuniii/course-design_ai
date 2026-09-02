from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from db_api import get_train_dataset

_rf_model = None
_metrics = {}

def train_rf():
    """训练随机森林质量分类模型"""
    X, y = get_train_dataset()
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X, y)
    
    # 计算训练集指标
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred))
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
    label = int(_rf_model.predict(X)[0])
    confidence = float(_rf_model.predict_proba(X).max())
    return label, confidence
