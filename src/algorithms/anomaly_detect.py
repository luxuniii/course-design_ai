from sklearn.ensemble import IsolationForest
from db_api import get_train_dataset

_if_model = None

def train_if():
    """训练孤立森林异常检测模型"""
    X, y = get_train_dataset()
    # 仅用合格样本训练
    X_normal = X[y == 0]
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_normal)
    global _if_model
    _if_model = model
    return {"status": "success", "train_samples": len(X_normal)}

def predict_anomaly(batch_id):
    """预测单批次异常得分，得分越高风险越高"""
    from db_api import get_single_batch_features
    if _if_model is None:
        train_if()
    X = get_single_batch_features(batch_id)
    # 转换为0-1风险分，越高越异常
    score = _if_model.decision_function(X)[0]
    risk_score = float((-score + 0.5) * 100)
    risk_score = max(0, min(100, risk_score))
    
    if risk_score < 30:
        level = "低风险"
    elif risk_score < 70:
        level = "中风险"
    else:
        level = "高风险"
    return risk_score, level
