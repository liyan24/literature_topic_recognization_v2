import streamlit as st
import pandas as pd
from modules.term_extraction import TermExtractor
from modules.clustering import Clusterer
from modules.topic_labeling import TopicLabeler
from streamlit_echarts import st_echarts
import urllib.parse
import time
import json
import re

st.set_page_config(page_title="科技文献聚类分析系统", layout="wide")

def main():
    st.title("科技文献聚类与主题标签揭示软件")
    
    # 在侧边栏创建步骤选择
    step = st.sidebar.radio("选择步骤", ["术语抽取", "聚类及可视化", "主题标签揭示"])
    
    # 术语抽取部分
    if step == "术语抽取":
        st.header("术语抽取")
        
        # 主界面文件上传和提示词模板
        uploaded_file = st.file_uploader("上传文献CSV文件", type=['csv'])

        # 侧边栏参数设置
        with st.sidebar:
            st.subheader("参数设置")
            api_key = st.text_input("API Key", type="password", value="")
            base_url = st.text_input("Base URL", value="https://api.deepseek.com")
            model_name = st.text_input("模型名称", value="deepseek-chat")
            start_button = st.button("RUN")
        
        # 主界面显示结果
        if start_button and uploaded_file is not None:
            extractor = TermExtractor(api_key, base_url, model_name)
            results = extractor.extract_terms(uploaded_file)
            st.dataframe(results)
            st.download_button(
                "下载提取结果",
                results.to_csv(index=False, header=False).encode('utf-8'),
                "extracted_terms.txt", 
                "text/plain"
            )

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
            api_key = st.text_input("API Key", type="password", value="")
            base_url = st.text_input("Base URL", value="https://api.deepseek.com")
            model_name = st.text_input("模型名称", value="deepseek-chat")

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
            st.write("原始主题标签结果：", st.session_state.topics_text)
            
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