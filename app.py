import streamlit as st
import pandas as pd
from modules.term_extraction import TermExtractor
from modules.clustering import Clusterer
from modules.topic_labeling import TopicLabeler

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
        st.header("步骤二：聚类及可视化")
        
        # 移到主界面的文件上传
        terms_file = st.file_uploader("上传术语列表", type=['csv'])
        
        # 侧边栏参数设置
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
            
            graph_data, cluster_results, visualization = clusterer.process(terms_file)
            st.write("关系图可视化：")
            st.echarts_chart(visualization)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "下载节点关系数据",
                    graph_data,
                    "graph_data.json",
                    "application/json"
                )
            with col2:
                st.download_button(
                    "下载聚类结果",
                    cluster_results,
                    "cluster.txt",
                    "text/plain"
                )

    # 主题标签揭示部分
    else:
        st.header("步骤三：主题标签揭示")
        
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