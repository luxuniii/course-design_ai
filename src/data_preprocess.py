import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    # ===================== 路径配置（适配你的文件命名） =====================
    raw_data_path = 'data/raw/secom.data'          # 传感器主数据文件
    raw_label_path = 'data/raw/secom_labels.data'   # 标签文件
    processed_path = 'data/processed/secom_processed.csv'
    report_path = 'data/docs/data_statistic.md'

    # ===================== 1. 文件存在性校验 =====================
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"找不到主数据文件：{raw_data_path}\n"
            "请将 secom.data 放入 data/raw/ 目录下"
        )
    if not os.path.exists(raw_label_path):
        raise FileNotFoundError(
            f"找不到标签文件：{raw_label_path}\n"
            "请将 secom_labels.data 放入 data/raw/ 目录下"
        )

    # 确保输出目录存在
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # ===================== 2. 加载原始数据与标签 =====================
    print("[1/6] 正在加载原始数据集...")

    # 主数据：空格分隔，无表头，缺失值标记为 NaN
    data = pd.read_csv(
        raw_data_path,
        sep=r'\s+',
        header=None,
        na_values=['NaN'],
        engine='python'
    )

    # 标签文件：每行一个标签，原始值 -1=合格，1=失效
    labels = pd.read_csv(
        raw_label_path,
        sep=r'\s+',
        header=None,
        usecols=[0],
        names=['label']
    )

    # 标签转换：-1 → 0（合格），1 → 1（失效）
    labels['label'] = labels['label'].map({-1: 0, 1: 1})

    # 合并特征与标签
    df = pd.concat([data, labels], axis=1)

    # 清洗前统计
    raw_samples = df.shape[0]
    raw_features = df.shape[1] - 1
    raw_missing_rate = df.isnull().sum().sum() / (raw_samples * raw_features)
    raw_fail_rate = df['label'].mean()

    # ===================== 3. 缺失值处理 =====================
    print("[2/6] 正在处理缺失值...")

    # 按列统计缺失率，缺失率 > 30% 的特征列直接剔除
    missing_ratio = df.isnull().mean()
    drop_missing_cols = missing_ratio[missing_ratio > 0.3].index.tolist()
    df = df.drop(columns=drop_missing_cols)

    # 剩余缺失值用中位数填充（工业数据更稳健，不受异常值影响）
    df = df.fillna(df.median(numeric_only=True))

    # ===================== 4. 异常值处理（3σ 准则） =====================
    print("[3/6] 正在处理异常值...")

    feature_cols = [col for col in df.columns if col != 'label']
    for col in feature_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std == 0:
            continue
        # 超出 3σ 范围的数值替换为该列中位数
        df[col] = np.where(
            (df[col] < col_mean - 3 * col_std) | (df[col] > col_mean + 3 * col_std),
            df[col].median(),
            df[col]
        )

    # ===================== 5. 无效特征过滤 + 标准化 =====================
    print("[4/6] 正在过滤无效特征并标准化...")

    # 剔除方差接近 0 的无意义传感器特征
    var_threshold = 1e-6
    variances = df[feature_cols].var()
    low_var_cols = variances[variances < var_threshold].index.tolist()
    df = df.drop(columns=low_var_cols)

    # 更新特征列列表
    feature_cols = [col for col in df.columns if col != 'label']

    # Z-score 标准化，消除量纲差异
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # ===================== 6. 数据集划分（7:2:1） =====================
    print("[5/6] 正在划分数据集...")

    total = len(df)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)

    df['split'] = 'train'
    df.loc[train_end:val_end, 'split'] = 'val'
    df.loc[val_end:, 'split'] = 'test'

    # 保存处理后数据集
    df.to_csv(processed_path, index_label='batch_id')

    # ===================== 7. 生成数据质量统计报告 =====================
    print("[6/6] 正在生成数据质量统计报告...")

    clean_samples = df.shape[0]
    clean_features = len(feature_cols)
    clean_missing_rate = df[feature_cols].isnull().sum().sum() / (clean_samples * clean_features)

    report = f"""# 数据质量统计报告
## 一、清洗前概况
- 总样本数：{raw_samples}
- 原始特征数：{raw_features}
- 整体缺失率：{raw_missing_rate:.4%}
- 失效样本占比：{raw_fail_rate:.4%}

## 二、清洗后概况
- 有效样本数：{clean_samples}
- 保留特征数：{clean_features}
- 清洗后缺失率：{clean_missing_rate:.4%}
- 剔除高缺失列数：{len(drop_missing_cols)}
- 剔除低方差列数：{len(low_var_cols)}

## 三、数据集划分（按批次顺序 7:2:1）
- 训练集（train）：{train_end} 条
- 验证集（val）：{val_end - train_end} 条
- 测试集（test）：{total - val_end} 条
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # ===================== 完成提示 =====================
    print("\n" + "=" * 50)
    print("✅ 数据预处理全部完成！")
    print(f"处理后数据集：{processed_path}")
    print(f"质量统计报告：{report_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
