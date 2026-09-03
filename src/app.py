import os
import sys
# 将当前文件所在目录加入模块搜索路径，确保能正常导入同目录下的 db_api 和 algorithms
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# 导入数据库接口与算法模块
from db_api import (
    get_all_batches, get_batch_by_id, get_sensor_by_batch,
    save_algorithm_result, get_all_results
)
from algorithms import train_pca, reduce_single
from algorithms import train_if, predict_anomaly
from algorithms import train_rf, predict_quality

# ===================== 初始化 Flask 应用 =====================
app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)
CORS(app)

# 开发环境配置：禁用模板缓存，修改页面自动生效
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ===================== 页面路由 =====================
@app.route('/')
def index():
    """系统首页"""
    return render_template('index.html')

@app.route('/batch/list')
def batch_list():
    """批次列表页"""
    return render_template('batch_list.html')

@app.route('/batch/<int:batch_id>')
def batch_detail(batch_id):
    """批次详情页"""
    return render_template('batch_detail.html', batch_id=batch_id)

# ===================== 批次相关 API =====================
@app.route('/api/batch/list', methods=['GET'])
def api_batch_list():
    """获取全部生产批次列表"""
    try:
        batches = get_all_batches()
        return jsonify({
            "code": 200,
            "msg": "success",
            "data": batches
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500

@app.route('/api/batch/<int:batch_id>', methods=['GET'])
def api_batch_detail(batch_id):
    """获取单个批次详情"""
    try:
        batch = get_batch_by_id(batch_id)
        if not batch:
            return jsonify({"code": 404, "msg": "批次不存在", "data": None}), 404
        return jsonify({"code": 200, "msg": "success", "data": batch})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500

@app.route('/api/batch/<int:batch_id>/sensor', methods=['GET'])
def api_batch_sensor(batch_id):
    """获取指定批次的传感器数据"""
    try:
        sensors = get_sensor_by_batch(batch_id)
        return jsonify({
            "code": 200,
            "msg": "success",
            "data": sensors
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500

# ===================== 算法相关 API =====================
@app.route('/api/model/train', methods=['POST'])
def api_model_train():
    """全量训练三个算法模型"""
    try:
        pca_res = train_pca()
        if_res = train_if()
        rf_res = train_rf()
        
        return jsonify({
            "code": 200,
            "msg": "模型训练完成",
            "data": {
                "pca": pca_res,
                "anomaly_detect": if_res,
                "quality_predict": rf_res
            }
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500

@app.route('/api/model/predict/<int:batch_id>', methods=['POST'])
def api_model_predict(batch_id):
    """对指定批次执行全流程智能分析"""
    try:
        # 1. PCA降维
        pca_x, pca_y = reduce_single(batch_id)
        # 2. 异常检测
        anomaly_score, risk_level = predict_anomaly(batch_id)
        # 3. 质量预测
        predict_label, confidence = predict_quality(batch_id)
        
        # 保存结果到数据库
        save_algorithm_result(
            batch_id=batch_id,
            pca_x=pca_x,
            pca_y=pca_y,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            predict_label=predict_label,
            confidence=confidence
        )
        
        return jsonify({
            "code": 200,
            "msg": "分析完成",
            "data": {
                "batch_id": batch_id,
                "pca": {"x": pca_x, "y": pca_y},
                "anomaly": {"score": anomaly_score, "level": risk_level},
                "quality": {"label": predict_label, "confidence": confidence}
            }
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500

@app.route('/api/model/history', methods=['GET'])
def api_model_history():
    """获取所有批次的算法分析历史结果"""
    try:
        results = get_all_results()
        return jsonify({
            "code": 200,
            "msg": "success",
            "data": results
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500

# ===================== 启动服务（必须放在文件最末尾） =====================
if __name__ == '__main__':
    # 确保模板和静态资源目录存在
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    print("=" * 50)
    print("半导体产线智能质量预警系统 后端服务启动")
    print("访问地址：http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)


