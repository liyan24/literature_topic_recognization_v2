# Literature Topic Recognition V2

#### 项目介绍
基于 Streamlit 框架开发的文献主题识别系统，是对原 literature_topic_recognition 项目的重构版本。本项目提供了友好的 Web 界面，用于进行文献主题的自动识别与分类。

#### 技术架构
- 前端框架：Streamlit
- 后端语言：Python
- 主要功能：文献主题识别、文本分类
- 核心依赖：scikit-learn, transformers, pandas

#### 安装步骤

1. 克隆项目到本地
bash
git clone https://github.com/your-username/literature-topic-recognition-v2.git
cd literature-topic-recognition-v2
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行项目
```bash
streamlit run app.py    


4. 在浏览器中访问 `http://localhost:8501`

#### 主要功能

- 文献主题自动识别
- 批量文献处理
- 主题分类可视化
- 结果导出功能

#### 项目结构
literature-topic-recognition-v2/
├── app.py # 主程序入口
├── requirements.txt # 项目依赖
├── models/ # 模型文件
├── utils/ # 工具函数
└── data/ # 数据文件