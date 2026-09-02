import sqlite3
import os

def main():
    db_path = 'data/quality_system.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 创建生产批次主表
    create_batch_table = """
    CREATE TABLE IF NOT EXISTS production_batch (
        batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_code TEXT UNIQUE NOT NULL,
        produce_time TEXT,
        final_result INTEGER NOT NULL,
        split_type TEXT NOT NULL
    );
    """

    # 2. 创建传感器测量数据表
    create_sensor_table = """
    CREATE TABLE IF NOT EXISTS sensor_measurement (
        measure_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        sensor_no INTEGER NOT NULL,
        sensor_value REAL,
        FOREIGN KEY (batch_id) REFERENCES production_batch(batch_id)
    );
    """

    # 3. 创建算法结果表（统一存储三个算法模块输出）
    create_result_table = """
    CREATE TABLE IF NOT EXISTS algorithm_result (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        pca_x REAL,
        pca_y REAL,
        anomaly_score REAL,
        risk_level TEXT,
        predict_label INTEGER,
        confidence REAL,
        create_time TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (batch_id) REFERENCES production_batch(batch_id)
    );
    """

    # 创建索引，提升查询性能
    create_sensor_index = "CREATE INDEX IF NOT EXISTS idx_sensor_batch ON sensor_measurement(batch_id);"
    create_result_index = "CREATE INDEX IF NOT EXISTS idx_result_batch ON algorithm_result(batch_id);"

    # 执行建表
    cursor.execute(create_batch_table)
    cursor.execute(create_sensor_table)
    cursor.execute(create_result_table)
    cursor.execute(create_sensor_index)
    cursor.execute(create_result_index)

    conn.commit()
    conn.close()
    print(f"数据库初始化完成，文件路径：{db_path}")

if __name__ == '__main__':
    main()
