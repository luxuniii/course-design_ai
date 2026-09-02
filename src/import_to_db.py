import sqlite3
import pandas as pd
import os

def main():
    db_path = 'data/quality_system.db'
    processed_path = 'data/processed/secom_processed.csv'

    if not os.path.exists(processed_path):
        raise FileNotFoundError("请先运行data_preprocess.py生成处理后数据集")

    df = pd.read_csv(processed_path)
    feature_cols = [col for col in df.columns if col not in ['batch_id', 'label', 'split']]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("正在导入生产批次数据...")
    batch_data = []
    for idx, row in df.iterrows():
        batch_data.append((
            f"BATCH_{int(row['batch_id']):04d}",
            None,  # 原始数据无生产时间，留空
            int(row['label']),
            row['split']
        ))

    cursor.executemany(
        "INSERT INTO production_batch (batch_code, produce_time, final_result, split_type) VALUES (?, ?, ?, ?)",
        batch_data
    )
    conn.commit()

    print("正在导入传感器测量数据...")
    sensor_data = []
    for idx, row in df.iterrows():
        batch_id = idx + 1  # 自增主键从1开始
        for sensor_no, col in enumerate(feature_cols):
            sensor_data.append((batch_id, sensor_no, row[col]))

    cursor.executemany(
        "INSERT INTO sensor_measurement (batch_id, sensor_no, sensor_value) VALUES (?, ?, ?)",
        sensor_data
    )
    conn.commit()

    # 统计导入结果
    cursor.execute("SELECT COUNT(*) FROM production_batch")
    batch_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sensor_measurement")
    sensor_count = cursor.fetchone()[0]

    conn.close()
    print(f"导入完成！")
    print(f"生产批次：{batch_count} 条")
    print(f"传感器记录：{sensor_count} 条")

if __name__ == '__main__':
    main()
