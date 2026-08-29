
# 半导体产线智能质量预警B/S系统

## 项目简介
本项目为制造智能技术课程设计作品，基于半导体晶圆制造真实工业数据集，构建B/S架构的智能质量预警系统。集成高维特征降维、工业异常检测、质量分类预测三项核心智能技术，实现生产批次质量提前预判与工艺异常预警，辅助产线质量管控。

## 技术栈
- 后端：Flask
- 数据库：SQLite
- 前端：HTML + Bootstrap + JavaScript
- 算法库：pandas、numpy、scikit-learn
- 开发模式：vibe coding（规格驱动开发 + AI辅助编码）
- 版本管理：Git

## 目录结构
```

course-design_ai/
├── data/                     # 项目数据总目录
│   ├── raw/                  # 原始数据集（未修改）
│   ├── processed/            # 预处理完成的数据集
│   └── docs/                 # 数据相关文档（来源说明、数据字典、统计报告）
├── prompt/                   # AI 交互记录全量存档
│   └── phase3_data/          # 第三阶段数据准备阶段 prompt 记录
│       └── backup_before_compress/  # 上下文压缩前备份
├── src/                      # 源码目录
│   ├── algorithms/           # 核心算法模块
│   ├── test/                 # 自动化单元测试
│   ├── data_preprocess.py    # 数据预处理脚本
│   ├── init_db.py            # 数据库初始化建表脚本
│   ├── import_to_db.py       # 数据批量导入脚本
│   ├── db_api.py             # 数据库交互封装层
│   └── app.py                # 后端服务主入口
├── process_log/              # 过程档案（打卡记录、AI 出错修正记录）
├── project_rules.md          # vibe coding 项目全局规则
├── requirements.txt          # Python 依赖清单
└── README.md                 # 项目说明文档



## 数据资源
### 数据集来源
本项目采用 UCI 公开工业数据集 SECOM（半导体制造工艺传感器数据集），官方永久链接：
https://archive.ics.uci.edu/ml/datasets/SECOM

数据集采集自真实半导体晶圆生产线，包含590路工艺传感器测量数据，对应1567个生产批次的质量合格/失效标签，具备真实工业数据的缺失值、高维特征、类别不平衡等特性。

### 数据说明
- `data/raw/`：存放原始数据集文件，保持官方原始状态，未做任何修改
- `data/processed/`：存放预处理完成的数据集，可通过脚本一键复现
- `data/docs/`：包含数据集来源说明、数据字典、数据质量统计报告

## 运行步骤
### 环境准备
安装Python 3.8及以上版本，执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

### 1. 数据预处理

执行原始数据清洗、缺失值处理、异常值过滤、特征标准化与数据集划分：


python src/data_preprocess.py


处理后数据输出至 `data/processed/secom_processed.csv`，同时生成数据质量统计报告。

### 2. 数据库初始化

创建数据库表结构：

```
python src/init_db.py
```

将处理后的数据批量导入数据库：

```
python src/import_to_db.py
```

### 3. 数据验收测试

验证数据库读写功能与数据完整性：

```
python src/test/test_db.py
```

## 过程档案说明

项目全程采用 vibe coding 开发模式，所有与 AI 工具的交互记录均存档于 `prompt/` 目录，按开发阶段分目录保存。

- 每次上下文压缩前均导出完整备份，存入对应阶段的 `backup_before_compress/` 目录
- 记录包含任务 prompt、AI 输出、人工修正说明三部分
- 各阶段完成后同步更新，确保开发过程完整可追溯

## 开发规范

- 全程使用 Git 进行版本管理，小步提交，提交说明语义清晰
- 每个功能模块配套自动化单元测试
- AI 生成代码均经过人工业务逻辑审查与修正
- 所有数据库操作通过封装层调用，不直接编写 SQL

```

