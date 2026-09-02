import os
import sqlite3
import pandas as pd

# 数据库路径：基于当前文件位置自动定位，不受运行目录影响
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'data',
    'quality_system.db'
)


def _get_connection():
    """内部工具函数：获取数据库连接，调用方需负责关闭"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库文件不存在：{DB_PATH}，请先执行 init_db.py 与 import_to_db.py")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 查询结果支持通过字段名访问
    return conn


# ------------------------------
# 1. 生产批次查询接口
# ------------------------------

def get_all_batches():
    """
    获取全部生产批次列表
    返回：list[dict]，每个元素包含批次基础信息
    """
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT batch_id, batch_code, produce_time, final_result, split_type
            FROM production_batch
            ORDER BY batch_id
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_batch_by_id(batch_id):
    """
    根据ID查询单个批次详情
    参数：batch_id(int) 批次主键
    返回：dict 批次信息，不存在返回None
    """
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT batch_id, batch_code, produce_time, final_result, split_type
            FROM production_batch
            WHERE batch_id = ?
        """, (batch_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_failed_batches():
    """查询所有失效批次列表"""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT batch_id, batch_code, produce_time
            FROM production_batch
            WHERE final_result = 1
            ORDER BY batch_id
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# ------------------------------
# 2. 传感器数据查询接口
# ------------------------------

def get_sensor_by_batch(batch_id):
    """
    查询指定批次的全部传感器测量数据
    返回：list[dict]，包含sensor_no、sensor_value
    """
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sensor_no, sensor_value
            FROM sensor_measurement
            WHERE batch_id = ?
            ORDER BY sensor_no
        """, (batch_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_train_dataset():
    """
    获取训练集完整数据（算法训练专用）
    返回：
        X(pd.DataFrame)：每行一个批次，每列一个传感器特征
        y(pd.Series)：对应批次的质量标签（0合格/1失效）
    """
    conn = _get_connection()
    try:
        query = """
            SELECT s.batch_id, s.sensor_no, s.sensor_value, p.final_result
            FROM sensor_measurement s
            JOIN production_batch p ON s.batch_id = p.batch_id
            WHERE p.split_type = 'train'
        """
        df = pd.read_sql(query, conn)

        # 长表转宽表：行=批次，列=传感器编号，值=测量值
        X = df.pivot(index='batch_id', columns='sensor_no', values='sensor_value')
        y = df.groupby('batch_id')['final_result'].first()
        return X, y
    finally:
        conn.close()


def get_single_batch_features(batch_id):
    """
    获取单个批次的特征向量（用于单样本预测推理）
    返回：pd.DataFrame 单行特征，列顺序与训练集一致
    """
    conn = _get_connection()
    try:
        query = """
            SELECT sensor_no, sensor_value
            FROM sensor_measurement
            WHERE batch_id = ?
            ORDER BY sensor_no
        """
        df = pd.read_sql(query, conn, params=(batch_id,))
        # 转置为单行特征矩阵
        X = df.set_index('sensor_no').T
        X.index = [batch_id]
        return X
    finally:
        conn.close()


# ------------------------------
# 3. 算法结果读写接口
# ------------------------------

def save_algorithm_result(batch_id, pca_x, pca_y, anomaly_score, risk_level, predict_label, confidence):
    """
    保存单次智能分析的全部算法结果（降维、异常检测、质量预测）
    参数全部对应 algorithm_result 表字段
    """
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO algorithm_result
            (batch_id, pca_x, pca_y, anomaly_score, risk_level, predict_label, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (batch_id, pca_x, pca_y, anomaly_score, risk_level, predict_label, confidence))
        conn.commit()
        return cursor.lastrowid  # 返回新记录ID
    finally:
        conn.close()


def get_latest_result(batch_id):
    """
    查询指定批次最新的一次算法分析结果
    返回：dict 结果信息，不存在返回None
    """
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT result_id, batch_id, pca_x, pca_y, anomaly_score,
                   risk_level, predict_label, confidence, create_time
            FROM algorithm_result
            WHERE batch_id = ?
            ORDER BY create_time DESC
            LIMIT 1
        """, (batch_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_results():
    """获取所有批次的最新算法结果列表，关联批次编号"""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.result_id, a.batch_id, p.batch_code, a.risk_level,
                   a.predict_label, a.confidence, a.create_time
            FROM algorithm_result a
            JOIN production_batch p ON a.batch_id = p.batch_id
            GROUP BY a.batch_id
            HAVING MAX(a.create_time)
            ORDER BY a.batch_id
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
