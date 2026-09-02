import sqlite3

def main():
    db_path = 'data/quality_system.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=== 数据库测试开始 ===")

    # 测试1：统计总批次
    cursor.execute("SELECT COUNT(*), SUM(final_result) FROM production_batch")
    total, fail = cursor.fetchone()
    print(f"测试1 - 批次统计：总批次{total}，失效批次{fail}，良率{(1-fail/total):.2%}")

    # 测试2：查询单批次传感器数据
    cursor.execute("SELECT COUNT(*) FROM sensor_measurement WHERE batch_id = 1")
    sensor_num = cursor.fetchone()[0]
    print(f"测试2 - 单批次传感器数：{sensor_num} 个")

    # 测试3：查询算法结果表结构
    cursor.execute("PRAGMA table_info(algorithm_result)")
    cols = [col[1] for col in cursor.fetchall()]
    print(f"测试3 - 算法结果表字段：{cols}")

    # 测试4：按数据集划分查询
    cursor.execute("SELECT split_type, COUNT(*) FROM production_batch GROUP BY split_type")
    splits = cursor.fetchall()
    print("测试4 - 数据集划分：")
    for s, cnt in splits:
        print(f"  {s}: {cnt} 条")

    conn.close()
    print("=== 全部测试通过 ===")

if __name__ == '__main__':
    main()
