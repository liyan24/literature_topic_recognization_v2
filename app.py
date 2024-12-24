# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from modules.term_extraction import TermExtractor
from modules.clustering import Clusterer
from modules.topic_labeling import TopicLabeler
from streamlit_echarts import st_echarts
from utils.time_utils import format_time, estimate_remaining_time
import urllib.parse
import time
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    st.set_page_config(page_title="科技文献聚类分析系统", layout="wide")
    
    st.title("科技文献聚类与主题标签揭示软件")
    
    # 在侧边栏创建步骤选择
    step = st.sidebar.radio("选择步骤", ["术语抽取", "聚类及可视化", "主题标签揭示"])
    
    # 术语抽取部分
    if step == "术语抽取":
        st.header("术语抽取")
        
        # 主界面文件上传和提示词模板
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])

        # 侧边栏参数设置
        with st.sidebar:
            st.subheader("参数设置")
            api_key = st.text_input("API Key", type="password", value=os.getenv("LLM_API_KEY"))
            base_url = st.text_input("Base URL", value=os.getenv("LLM_BASE_URL"))
            model_name = st.text_input("模型名称", value=os.getenv("LLM_MODEL_NAME"))
            col1, col2 = st.columns(2)
            with col1:
                start_button = st.button("RUN")
        
        results = []
        error_indices = []

        # 主界面显示结果
        if start_button and uploaded_file is not None:
            # 创建显示容器
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            time_placeholder = st.empty()
            estimate_placeholder = st.empty()
            
            try:
                # 读取CSV文件并获取总记录数
                df = pd.read_csv(uploaded_file)
                total_records = len(df)
                
                # 创建进度显示
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                
                # 记录开始时间
                start_time = time.time()
                
                # 创建抽取器并处理
                extractor = TermExtractor(
                    api_key, 
                    base_url, 
                    model_name,
                )
                
                def update_progress(i):
                    current = i + 1
                    progress_bar.progress(current / total_records)
                    status_placeholder.text(f"正在处理: {current}/{total_records}")
                    
                    # 计算已用时间
                    elapsed_time = time.time() - start_time
                    time_placeholder.text(f"已用时间: {format_time(elapsed_time)}")
                    
                    # 估算剩余时间
                    remaining_time = estimate_remaining_time(elapsed_time, current, total_records)
                    estimate_placeholder.text(f"预计剩余时间: {remaining_time}")
                    
                
                for i, row in df.iterrows():
                    update_progress(i)
                    try:
                        response = extractor.extract_terms_single(row)  # 假设有单条处理的方法
                        results.append(response.strip())
                    except Exception as e:
                        error_indices.append(i)
                        extractor.log_error(i, row, str(e))
                        results.append("")
                
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
                valid_results = pd.DataFrame({'terms': results})
                
                # 显示有效结果
                st.write(f"已处理{total_records} 条记录，失败 {len(error_indices)} 条记录")
                st.dataframe(valid_results)
                
                # 下载成功处理的结果
                st.download_button(
                    "下载提取结果",
                    valid_results.to_csv(index=False).encode('utf-8'),
                    "extracted_terms.txt",
                    "text/plain"
                )
                
            except Exception as e:
                st.error(f"程序发生错误: {str(e)}")
                progress_placeholder.empty()
            

    # 聚类及可视化部分
    elif step == "聚类及可视化":
        st.header("聚类及可视化")
        
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

    # 主题标签揭示部分
    else:
        st.header("主题标签揭示")
        
        # 初始化 session state
        if 'topics_text' not in st.session_state:
            st.session_state.topics_text = None
        if 'topics' not in st.session_state:
            st.session_state.topics = None
            
        # 侧边栏参数设置
        with st.sidebar:
            st.subheader("参数设置")
            api_key = st.text_input("API Key", type="password", value=os.getenv("LLM_API_KEY"))
            base_url = st.text_input("Base URL", value=os.getenv("LLM_BASE_URL"))
            model_name = st.text_input("模型名称", value=os.getenv("LLM_MODEL_NAME"))

        # 第一步：上传聚类文件并获取主题
        cluster_file = st.file_uploader("上传聚类结果文件", type=['txt'])
        get_topics_button = st.button("获取主题标签")

        if get_topics_button and cluster_file is not None:
            # 处理主题标签
            labeler = TopicLabeler(api_key, base_url, model_name)
            st.session_state.topics_text = labeler.label_topics(cluster_file)
            # 提取主题标签
            st.session_state.topics = ';'.join([match.group(1) for match in re.finditer(r'《(.*?)》', st.session_state.topics_text)])
            

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
            topics_input = st.text_area("编辑主题标签（用分号分隔）", st.session_state.topics)
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
                        'roam': True,
                        'labelLayout': {
                            'hideOverlap': True
                        }
                    }]
                }
            
                # 显示图表
                st.write("关系图可视化：")
                st_echarts(visualization, height="800px")

if __name__ == "__main__":
    main() 