import streamlit as st
import pandas as pd
from modules.term_extraction import TermExtractor
from modules.clustering import Clusterer
from modules.topic_labeling import TopicLabeler
from streamlit_echarts import st_echarts
import urllib.parse
import time
import json

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
            api_key = st.text_input("API Key", type="password", value="sk-ba6af59586a1441ca7ebb6ffbb0a75c8")
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
                results.to_csv(index=False).encode('utf-8'),
                "extracted_terms.csv",
                "text/csv"
            )

    # 聚类及可视化部分
    elif step == "聚类及可视化":
        st.header("聚类及可视化")
        
        terms_file = st.file_uploader("上传术语列表", type=['txt'])
        
        # 侧边栏参数设置 - 移除图例设置
        with st.sidebar:
            st.subheader("参数设置")
            st.subheader("聚类参数")
            num_nodes = st.number_input("节点个数", value=300)
            num_relations = st.number_input("关系个数", value=500)
            random_seed = st.number_input("随机种子", value=0)
            decay_steps = st.number_input("衰减步数", value=1)
            max_iter = st.number_input("最大迭代次数", value=100)
            initial_step = st.number_input("初始步长", value=1.0)
            step_decay = st.number_input("步长衰减", value=0.75)
            step_convergence = st.number_input("步长收敛", value=0.001)
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
                
                # 先进行聚类处理，使用临时的图例名称
                graph_data, cluster_results_str, visualization = clusterer.process(terms_file)
                
                # 解析聚类结果获取类别数量
                num_clusters = len(visualization['series'][0]['categories'])
                
                # 设置默认图例名称
                default_legend_names = [f"类别{i+1}" for i in range(num_clusters)]
                
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
                
                # 添加图例设置区域
                st.subheader("图例设置")
                legend_names = st.text_area(
                    "输入图例名称（每行一个）",
                    value="\n".join(default_legend_names),
                    help=f"当前聚类数量为{num_clusters}，请输入对应数量的图例名称"
                )
                update_legend = st.button("更新图例")
                
                # 创建图表容器
                chart_container = st.empty()
                
                # 显示初始图表
                with chart_container:
                    st.write("关系图可视化：")
                    st_echarts(visualization, height="800px")
                
                # 当点击更新图例按钮时
                if update_legend:
                    new_legend_list = [name.strip() for name in legend_names.split('\n') if name.strip()]
                    new_visualization = clusterer.update_visualization(visualization, new_legend_list)
                    with chart_container:
                        st.write("关系图可视化：")
                        st_echarts(new_visualization, height="800px")
                
            except Exception as e:
                status_placeholder.error(f"处理过程中发生错误: {str(e)}")
                progress_placeholder.empty()

    # 主题标签揭示部分
    else:
        st.header("主题标签揭示")
        
        # 移到主界面的文件上传
        cluster_file = st.file_uploader("上传聚类结果文件", type=['txt'])
        
        # 侧边栏参数设置
        with st.sidebar:
            st.subheader("参数设置")
            api_key = st.text_input("API Key", type="password", value="sk-ba6af59586a1441ca7ebb6ffbb0a75c8")
            base_url = st.text_input("Base URL", value="https://api.deepseek.com")
            model_name = st.text_input("模型名称", value="deepseek-chat")
            start_button = st.button("RUN")
        
        # 主界面显示结果
        if start_button and cluster_file is not None:
            labeler = TopicLabeler(api_key, base_url, model_name)
            topics = labeler.label_topics(cluster_file)
            st.dataframe(topics)
            st.download_button(
                "下载主题标签结果",
                topics.to_csv(index=False).encode('utf-8'),
                "topic_labels.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main() 