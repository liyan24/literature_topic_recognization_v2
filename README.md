# Literature Topic Recognition V2

#### 项目介绍

基于 Streamlit 框架开发的文献主题识别系统，是对原 literature_topic_recognition 项目的重构版本。本项目提供了友好的 Web 界面，用于进行文献主题的自动识别与分类。

#### 更新日志
- 2024-12-18 增加了术语抽取部分，增加了进度条及错误处理，如果中间出错，将在输出文件位置放置空字符串，在程序运行结束后将错误的输入行号和输入信息也提供下载。
- 2024-12-18 增加了主题揭示部分，增加了获取标签结果之后的下载功能，可以将揭示标签结果下载为txt文件。
- 2024-12-23 增加了.env文件，用户配置可以使用.env文件，不需要在代码或者前台中每次配置。

#### 背景
![背景](image/README/background.png)

#### 软件效果
![软件效果](image/README/software_effect.png)

#### 方法路线
![方法路线](image/README/method_route.png)


#### 技术架构

- 前端框架：Streamlit
- 后端语言：Python
- 主要功能：文献主题识别、文本分类

#### 主要功能

- **术语抽取**：从文献标题和摘要中自动提取科技术语
- **聚类及可视化**：基于术语共现关系进行聚类分析，生成可视化网络图
- **主题标签揭示**：为聚类结果自动生成主题标签，揭示研究热点
- **批量文献处理**：支持批量处理大量文献数据
- **结果导出功能**：支持将处理结果导出为多种格式

#### 处理步骤

1. **术语抽取**
   - 上传CSV格式的文献数据（包含标题和摘要）
   - 配置大模型API参数（API Key、Base URL、模型名称）
   - 系统自动从文献中提取科技术语
   - 导出提取结果，包含错误处理记录

2. **聚类及可视化**
   - 上传术语列表文件
   - 配置聚类参数（节点个数、关系个数、迭代参数等）
   - 系统自动构建术语共现矩阵，进行聚类分析
   - 生成术语关系网络可视化图
   - 导出聚类结果和节点关系数据

3. **主题标签揭示**
   - 上传聚类结果文件
   - 配置大模型API参数
   - 系统自动为每个聚类生成主题标签
   - 上传节点关系数据，生成带标签的可视化图表
   - 导出主题标签结果

#### 安装步骤

1. 克隆项目到本地
```bash
   git clone https://gitee.com/liyan24/literature_topic_recognition_v2.git
   cd literature-topic-recognition-v2

```

2. 安装依赖(python3.12)
```bash
pip install -r requirements.txt
```

或者：

```bash
pip install streamlit
pip install streamlit-echarts
pip install langchain
pip install langchain_openai
pip install networkx
```

建议使用国内镜像源安装，以提高安装速度。如：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt
```

以及
```bash
pip install streamlit -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install streamlit-echarts -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install langchain_openai -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install networkx -i https://pypi.tuna.tsinghua.edu.cn/simple/
```


3. 运行项目
```bash
streamlit run app.py  
```

4. 在浏览器中访问 `http://localhost:8501`（初次运行streamlit，需要填写邮箱）
    (1) 第一步输入的样例文件位于data\test.csv，需要配置相关大模型进行术语抽取（基于prompt，会消耗较多token），直接使用其他关键词列表可以跳过此步骤。
    (2) 第二步输入的样例文件位于data\keywords.txt。输出文件有cluster.txt及graph_data.json

#### 项目结构
literature-topic-recognition-v2/
├── app.py # 主程序入口
├── requirements.txt # 项目依赖
├── modules/ # 核心功能模块
│   ├── clustering.py # 聚类分析模块
│   ├── term_extraction.py # 术语抽取模块
│   ├── topic_labeling.py # 主题标签模块
│   └── community_vos.py # 社区检测模块
├── utils/ # 工具函数
├── image/ # 图片文件
└── data/ # 数据文件
