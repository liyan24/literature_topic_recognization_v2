# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from modules.term_extraction import TermExtractor
from modules.clustering import Clusterer
from modules.topic_labeling import TopicLabeler
from modules import homepage
from streamlit_echarts import st_echarts
from utils.time_utils import format_time, estimate_remaining_time
import urllib.parse
import time
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'step1_checkpoint.jsonl')

# 第一步输入文件的标题列、摘要列可能使用的列名
TITLE_COLUMN_CANDIDATES = (
    'title', '标题', '题名', '篇名', '课题名称', '项目名称', '论文题目', '论文标题', 'title(s)', 'titles'
)
ABSTRACT_COLUMN_CANDIDATES = (
    'abstract', '摘要', '内容摘要', 'abstracts', 'abstract(s)'
)


def clean_dataframe(df):
    """清除单元格中的换行符、制表符及多余空白字符"""
    df = df.dropna(axis=1, how='all')
    for col in df.columns:
        df[col] = (df[col].astype('string')
                   .str.replace(r'[\r\n\t]+', ' ', regex=True)
                   .str.replace(r'\s+', ' ', regex=True)
                   .str.strip())
    return df


def read_literature_file(uploaded_file):
    """读取第一步的文献数据文件，支持 CSV 与 Excel 格式"""
    filename = uploaded_file.name.lower()
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file, sheet_name=0)
        # 处理表头不在首行、存在空白列的情况（如 data/step1-excel.xlsx）：
        # 重新按无表头读取，删除全空行/列后，将第一行非空数据提升为表头
        if all(str(c).startswith('Unnamed') for c in df.columns):
            raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
            raw = raw.dropna(axis=1, how='all').dropna(axis=0, how='all')
            raw.columns = raw.iloc[0].astype('string').str.strip()
            df = raw.iloc[1:].reset_index(drop=True)
    else:
        raise ValueError('仅支持 CSV 或 Excel（.xlsx / .xls）文件')
    return clean_dataframe(df)


def normalize_literature_columns(df):
    """将输入数据统一为 title / abstract 两列"""
    lower_map = {c: str(c).strip().lower() for c in df.columns}
    title_col = next((c for c, l in lower_map.items() if l in TITLE_COLUMN_CANDIDATES), None)
    abstract_col = next((c for c, l in lower_map.items() if l in ABSTRACT_COLUMN_CANDIDATES), None)

    if title_col is None:
        remaining = [c for c in df.columns if c != abstract_col]
        if not remaining:
            raise ValueError('未在文件中找到标题列')
        title_col = remaining[0]
        if abstract_col is None and len(remaining) > 1:
            abstract_col = remaining[1]

    df = df.rename(columns={title_col: 'title'})
    if abstract_col is not None and abstract_col != title_col:
        df = df.rename(columns={abstract_col: 'abstract'})
    if 'abstract' not in df.columns:
        df['abstract'] = ''

    df = df[['title', 'abstract']].copy()
    df['title'] = df['title'].fillna('').str.strip()
    df['abstract'] = df['abstract'].fillna('').str.strip()
    # 去掉与列名重复的行（防止表头行混入数据）
    df = df[~df['title'].isin(TITLE_COLUMN_CANDIDATES + ABSTRACT_COLUMN_CANDIDATES)]
    df = df[df['title'] != ''].reset_index(drop=True)
    return df


def sample_download_button(filename, label=None):
    """提供 data 目录下样例文件的下载按钮"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        data = f.read()
    st.download_button(
        label=label or f"下载样例：{filename}",
        data=data,
        file_name=filename,
        key=f"sample_dl_{filename}"
    )


def render_api_key_settings():
    """侧边栏大模型 API 参数设置（第一步与第三步共用）"""
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv("LLM_API_KEY", "")
    message_container = st.empty()
    with st.form("api_key_form", clear_on_submit=True):
        user_api_key = st.text_input(
            "请输入您的 API Key",
            type="password",
            placeholder="输入您的 API Key 或使用默认值",
            value=""  # 不显示任何值，确保安全
        )
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("保存 API Key", use_container_width=True)
        with col2:
            reset_submitted = st.form_submit_button("重置为默认值", use_container_width=True)
        if submitted:
            if user_api_key:
                st.session_state.api_key = user_api_key
                message_container.success("API Key 已安全保存！")
            else:
                message_container.warning("请输入有效的 API Key")
        if reset_submitted:
            default_key = os.getenv("LLM_API_KEY", "")
            if default_key:
                st.session_state.api_key = default_key
                message_container.success("已重置为默认 API Key！")
            else:
                message_container.warning("未找到默认 API Key，请检查环境变量配置")

    api_key = st.session_state.api_key
    base_url = st.text_input("Base URL", value=os.getenv("LLM_BASE_URL"))
    model_name = st.text_input("模型名称", value=os.getenv("LLM_MODEL_NAME"))
    return api_key, base_url, model_name


def step_term_extraction():
    st.header("术语抽取")

    with st.expander("📋 数据格式说明 / 样例下载 / 操作说明", expanded=False):
        st.markdown(
            """
            **输入数据格式说明**
            - 支持 **CSV** 与 **Excel（.xlsx / .xls）** 文件。
            - 文件需包含一列**标题**（支持的列名：`title` / `标题` / `题名` / `篇名` / `课题名称` / `论文题目` 等），
              以及可选的一列**摘要**（支持的列名：`abstract` / `摘要`）。
            - 程序会自动清除单元格中的换行符、制表符及多余空白字符，无需手动预处理。
            - Excel 表格的表头建议放在首行；如存在全空列会自动忽略。
            """
        )
        st.markdown("**数据样例下载**")
        col1, col2 = st.columns(2)
        with col1:
            sample_download_button('test.csv', '下载样例：test.csv（CSV 格式）')
        with col2:
            sample_download_button('step1-excel.xlsx', '下载样例：step1-excel.xlsx（Excel 格式）')
        st.markdown(
            """
            **操作说明**
            1. 上传文献数据文件（CSV 或 Excel），页面会显示清洗后的数据预览。
            2. 在左侧边栏配置大模型 API Key、Base URL 和模型名称（也可在 `.env` 文件中预设），
               并可按需调整**并发线程数**（默认 5，过大可能触发接口限流）。
            3. 点击左侧边栏的 **RUN** 开始抽取，页面会显示处理进度、已用时间与预计剩余时间。
            4. 处理完成后，下载**提取结果**（txt 文件，可直接作为第二步的输入）；
               处理失败的记录可单独下载核对。
            5. 开启**断点续传**后，处理进度会自动保存；中断后重新上传同一文件并点击 RUN，
               会自动跳过已处理的记录（依据记录序号与标题匹配），继续处理剩余部分。
            """
        )

    uploaded_file = st.file_uploader("上传文献数据文件（CSV / Excel）", type=['csv', 'xlsx', 'xls'])

    # 侧边栏参数设置
    with st.sidebar:
        st.subheader("参数设置")
        api_key, base_url, model_name = render_api_key_settings()
        max_workers = st.number_input("并发线程数", min_value=1, max_value=32, value=5,
                                      help="同时调用大模型 API 的线程数，过大可能触发接口限流")
        enable_resume = st.checkbox("断点续传", value=True,
                                    help="检测到上次未完成的进度时，自动跳过已处理的记录")
        if st.button("清除断点记录"):
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)
                st.success("断点记录已清除")
            else:
                st.info("暂无断点记录")
        start_button = st.button("RUN")

    if uploaded_file is not None:
        try:
            df = normalize_literature_columns(read_literature_file(uploaded_file))
            st.write("数据预览（已清除换行符与多余空白）：")
            st.dataframe(df.head(10))
            total_records = len(df)
            st.write(f"共读取 **{total_records}** 条记录")
        except Exception as e:
            st.error(f"文件解析失败: {str(e)}")
            df = None

        if start_button and df is not None:
            # 创建显示容器
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            time_placeholder = st.empty()
            estimate_placeholder = st.empty()

            try:
                # 创建进度显示
                with progress_placeholder.container():
                    progress_bar = st.progress(0)

                # 记录开始时间
                start_time = time.time()
                total_records = len(df)

                # 创建抽取器并处理
                extractor = TermExtractor(
                    api_key,
                    base_url,
                    model_name,
                )

                def update_progress(done_count, total):
                    progress_bar.progress(done_count / total)
                    status_placeholder.text(f"正在处理: {done_count}/{total}")

                    # 计算已用时间
                    elapsed_time = time.time() - start_time
                    time_placeholder.text(f"已用时间: {format_time(elapsed_time)}")

                    # 估算剩余时间
                    remaining_time = estimate_remaining_time(elapsed_time, done_count, total)
                    estimate_placeholder.text(f"预计剩余时间: {remaining_time}")

                # 加载断点：仅当记录序号与标题都匹配时才跳过，防止换文件后误跳
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                checkpoint = {}
                if enable_resume and os.path.exists(CHECKPOINT_PATH):
                    with open(CHECKPOINT_PATH, encoding='utf-8') as cf:
                        for line in cf:
                            if line.strip():
                                record = json.loads(line)
                                checkpoint[record['index']] = record
                    matched = {i for i in df.index
                               if i in checkpoint and checkpoint[i].get('title') == df.loc[i, 'title']}
                    checkpoint = {i: checkpoint[i] for i in matched}
                    if checkpoint:
                        st.info(f"断点续传：检测到 {len(checkpoint)} 条已完成记录，将跳过这些记录")

                remaining_df = df[~df.index.isin(checkpoint.keys())]

                if len(remaining_df) > 0:
                    new_results = extractor.process_dataframe(
                        remaining_df,
                        max_workers=max_workers,
                        progress_callback=lambda d, t: update_progress(d + len(checkpoint), total_records)
                    )
                    for idx, terms in zip(remaining_df.index, new_results):
                        checkpoint[idx] = {'index': int(idx), 'title': df.loc[idx, 'title'], 'terms': terms}

                # 保存断点
                with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as cf:
                    for idx in df.index:
                        if idx in checkpoint:
                            cf.write(json.dumps(checkpoint[idx], ensure_ascii=False) + '\n')

                results = [checkpoint.get(i, {}).get('terms', '') for i in df.index]
                error_indices = extractor.error_indices

                update_progress(total_records, total_records)

                # 显示处理结果
                if len(error_indices) > 0:
                    st.error(f"处理过程中出现 {len(error_indices)} 条记录错误")

                    # 获取出错记录
                    error_df = extractor.get_error_records_df()

                    # 下载出错记录
                    st.download_button(
                        "下载出错记录",
                        error_df.to_csv(index=False).encode('utf-8'),
                        "error_records.csv",
                        "text/csv"
                    )

                # 过滤掉空值记录（出错的记录）
                valid_terms = [r for r in results if r.strip()]

                # 显示有效结果
                st.write(f"已处理 {total_records} 条记录，失败 {len(error_indices)} 条记录")
                st.dataframe(pd.DataFrame({'terms': valid_terms}))

                # 下载成功处理的结果（txt，每行一条，可直接作为第二步输入）
                st.download_button(
                    "下载提取结果",
                    '\n'.join(valid_terms).encode('utf-8'),
                    "extracted_terms.txt",
                    "text/plain"
                )

            except Exception as e:
                st.error(f"程序发生错误: {str(e)}")
                progress_placeholder.empty()


def step_clustering():
    st.header("聚类及可视化")

    with st.expander("📋 数据格式说明 / 样例下载 / 操作说明", expanded=False):
        st.markdown(
            """
            **输入数据格式说明**
            - 上传**第一步下载的提取结果文件**（txt）：每行一条文献的术语，术语之间用**英文分号 `;`** 分隔。
            - 若已有现成的关键词列表（同样为每行一条、分号分隔的 txt），可直接上传，跳过第一步。
            """
        )
        st.markdown("**数据样例下载**")
        col1, col2 = st.columns(2)
        with col1:
            sample_download_button('keywords.txt', '下载样例：keywords.txt（关键词列表）')
        with col2:
            sample_download_button('extracted_terms.txt', '下载样例：extracted_terms.txt（第一步输出样式）')
        st.markdown(
            """
            **操作说明**
            1. 上传术语文件。
            2. 在左侧边栏配置聚类参数（默认参数通常即可获得较好效果）。
            3. 点击 **RUN**，等待聚类与网络图生成。
            4. 下载**节点关系数据（graph_data.json）**与**聚类结果（cluster.txt）**，供第三步使用。
            """
        )

    terms_file = st.file_uploader("上传术语列表", type=['txt'])

    # 侧边栏参数设置
    with st.sidebar:
        st.subheader("参数设置")
        st.subheader("聚类参数")

        # 第一行
        col1, col2 = st.columns(2)
        with col1:
            num_nodes = st.number_input("节点个数", value=300)
            decay_steps = st.number_input("衰减步数", value=1)
            initial_step = st.number_input("初始步长", value=1.0)
            step_convergence = st.number_input("步长收敛", value=0.001, format="%.3f")

        with col2:
            num_relations = st.number_input("关系个数", value=500)
            max_iter = st.number_input("最大迭代次数", value=100)
            step_decay = st.number_input("步长衰减", value=0.75)
            random_seed = st.number_input("随机种子", value=0)

        start_button = st.button("RUN")

    # 主界面显示结果
    if start_button and terms_file is not None:
        # 创建进度显示容器
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        try:
            with progress_placeholder.container():
                progress_bar = st.progress(0)

            # 参数预检：统计唯一术语数，避免节点个数超过实际术语数
            preview_lines = []
            for line in terms_file:
                t = line.decode('utf-8-sig').strip()
                if t and t.lower() != 'terms':
                    preview_lines.append(t)
            terms_file.seek(0)
            unique_terms = {x.strip() for line in preview_lines for x in line.split(';') if x.strip()}
            if not unique_terms:
                raise ValueError('术语文件为空或格式不正确，请检查文件内容')
            if num_nodes > len(unique_terms):
                st.warning(f"节点个数（{int(num_nodes)}）大于唯一术语数（{len(unique_terms)}），"
                           f"已自动调整为 {len(unique_terms)}")
                num_nodes = len(unique_terms)

            # 更新状态
            status_placeholder.text("正在初始化聚类器...")
            progress_bar.progress(20)

            clusterer = Clusterer(
                num_nodes=num_nodes,
                num_relations=num_relations,
                random_seed=random_seed,
                decay_steps=decay_steps,
                max_iter=max_iter,
                initial_step=initial_step,
                step_decay=step_decay,
                step_convergence=step_convergence
            )

            status_placeholder.text("正在处理数据并进行聚类...")
            progress_bar.progress(40)

            graph_data, cluster_results_str, visualization = clusterer.process(terms_file)

            status_placeholder.text("正在生成可视化...")
            progress_bar.progress(80)

            # 对数据进行编码
            encoded_graph_data = urllib.parse.quote(graph_data)
            encoded_cluster_results = urllib.parse.quote(cluster_results_str)

            # 处理完成
            progress_bar.progress(100)
            status_placeholder.text("处理完成！")
            time.sleep(1)

            # 清除进度显示
            progress_placeholder.empty()
            status_placeholder.empty()

            # 显示下载按钮
            col1, col2 = st.columns(2)
            with col1:
                href = f'data:application/json;charset=utf-8,{encoded_graph_data}'
                st.markdown(f'<a href="{href}" download="graph_data.json">下载节点关系数据</a>', unsafe_allow_html=True)

            with col2:
                href = f'data:text/plain;charset=utf-8,{encoded_cluster_results}'
                st.markdown(f'<a href="{href}" download="cluster.txt">下载聚类结果</a>', unsafe_allow_html=True)

            # 显示图表
            st.write("关系图可视化：")
            st_echarts(visualization, height="800px")

        except Exception as e:
            status_placeholder.error(f"处理过程中发生错误: {str(e)}")
            progress_placeholder.empty()


def step_topic_labeling():
    st.header("主题标签揭示")

    # 初始化 session state
    if 'topics_text' not in st.session_state:
        st.session_state.topics_text = None
    if 'topics' not in st.session_state:
        st.session_state.topics = None

    with st.expander("📋 数据格式说明 / 样例下载 / 操作说明", expanded=False):
        st.markdown(
            """
            **输入数据格式说明**
            - **聚类结果文件（cluster.txt）**：第二步的输出，每行格式为
              `cluster_x<TAB>词1;词2;词3;...`（制表符分隔聚类编号与术语列表）。
            - **节点关系数据（graph_data.json）**：第二步的输出，包含网络图的节点与关系。
            """
        )
        st.markdown("**数据样例下载**")
        col1, col2 = st.columns(2)
        with col1:
            sample_download_button('cluster.txt', '下载样例：cluster.txt（聚类结果）')
        with col2:
            sample_download_button('graph_data.json', '下载样例：graph_data.json（节点关系数据）')
        st.markdown(
            """
            **操作说明**
            1. 上传聚类结果文件，点击 **获取主题标签**，等待大模型生成（结果流式显示）。
            2. 生成后可下载主题结果文件；如有需要，可在文本框中编辑主题标签（用分号分隔）。
            3. 上传第二步的节点关系数据，点击 **生成图表**，查看带主题标签的网络可视化。
            """
        )

    # 侧边栏参数设置
    with st.sidebar:
        st.subheader("参数设置")
        api_key, base_url, model_name = render_api_key_settings()

    # 第一步：上传聚类文件并获取主题
    cluster_file = st.file_uploader("上传聚类结果文件", type=['txt'])
    get_topics_button = st.button("获取主题标签")

    if get_topics_button and cluster_file is not None:
        # 处理主题标签 - 使用流式输出
        labeler = TopicLabeler(api_key, base_url, model_name)

        # 创建流式输出的占位符
        output_placeholder = st.empty()
        status_placeholder = st.empty()

        status_placeholder.text("正在生成主题标签...")

        # 初始化完整文本
        full_text = ""

        try:
            # 使用流式输出
            for chunk in labeler.label_topics_stream(cluster_file):
                full_text += chunk
                # 实时更新显示
                output_placeholder.text(full_text)

            # 处理完成后存储结果
            st.session_state.topics_text = full_text
            # 提取主题标签
            st.session_state.topics = ';'.join([match.group(1) for match in re.finditer(r'《(.*?)》', full_text)])
            # 同步刷新标签编辑框，避免显示旧内容
            st.session_state.topics_edit_area = st.session_state.topics

            status_placeholder.text("主题标签生成完成！")

        except Exception as e:
            status_placeholder.error(f"生成主题标签时发生错误: {str(e)}")
            output_placeholder.empty()

    # 显示主题标签结果（如果存在）
    if st.session_state.topics_text:
        st.write("原始主题标签结果：\n", st.session_state.topics_text)
        st.download_button(
            "下载主题结果",
            st.session_state.topics_text.encode('utf-8'),
            "topics_text.txt",
            "text/plain"
        )
        # 第二步：允许编辑主题标签并上传图表数据
        st.subheader("生成可视化图表")
        # 通过 key 绑定 session_state，新生成标签时同步刷新编辑框内容
        if 'topics_edit_area' not in st.session_state:
            st.session_state.topics_edit_area = st.session_state.topics if st.session_state.topics else ''
        topics_input = st.text_area("编辑主题标签（用分号分隔）", key='topics_edit_area')
        graph_file = st.file_uploader("上传节点关系数据", type=['json'])
        visualize_button = st.button("生成图表")

        if visualize_button and graph_file is not None:
            # 处理主题标签列表
            topic_list = [topic.strip() for topic in topics_input.split(';') if topic.strip()]

            # 读取并处理图表数据
            graph_data = json.load(graph_file)

            # 创建图表配置
            visualization = {
                'title': {'text': '术语关系网络'},
                'tooltip': {},
                'legend': {'data': topic_list},
                'series': [{
                    'type': 'graph',
                    "layout": "none",
                    "symbolSize": 10,
                    "circular": {
                        "rotateLabel": False
                    },
                    "force": {
                        "repulsion": 50,
                        "gravity": 0.2,
                        "edgeLength": 30,
                        "friction": 0.6,
                        "layoutAnimation": True
                    },
                    "label": {
                        "show": True,
                        "position": "inside",
                        "margin": 8,
                        "valueAnimation": False
                    },
                    "lineStyle": {
                        "show": True,
                        "width": 0.5,
                        "opacity": 0.7,
                        "curveness": 0.3,
                        "type": "solid"
                    },
                    "roam": True,
                    "draggable": False,
                    "focusNodeAdjacency": True,
                    'data': graph_data['nodes'],
                    'links': graph_data['links'],
                    'categories': [{'name': topic} for topic in topic_list],
                    'labelLayout': {
                        'hideOverlap': True
                    }
                }]
            }

            # 显示图表
            st.write("关系图可视化：")
            st_echarts(visualization, height="800px")


def main():
    st.set_page_config(page_title="科技文献聚类分析系统", layout="wide")
    st.markdown("""
    <head>
        <meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' 'unsafe-eval' *; style-src 'self' 'unsafe-inline' *; script-src 'self' 'unsafe-inline' 'unsafe-eval' *;">
    </head>
""", unsafe_allow_html=True)
    st.title("科技文献聚类与主题标签揭示软件")

    # 在侧边栏创建步骤选择
    step = st.sidebar.radio("选择步骤", ["首页", "术语抽取", "聚类及可视化", "主题标签揭示"])

    if step == "首页":
        homepage.render()
    elif step == "术语抽取":
        step_term_extraction()
    elif step == "聚类及可视化":
        step_clustering()
    else:
        step_topic_labeling()


if __name__ == "__main__":
    main()
