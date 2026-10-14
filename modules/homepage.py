# -*- coding: utf-8 -*-
"""首页模块：独立维护项目介绍页面，不依赖 README.md。"""
import os

import streamlit as st

IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'image', 'README')

_CSS = """
<style>
.ltr-hero {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 60%, #3b6bc4 100%);
    padding: 2.6rem 2rem 2.2rem;
    border-radius: 14px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 14px rgba(30, 60, 114, 0.25);
}
.ltr-hero h1 { color: #ffffff; margin: 0; font-size: 2.1rem; letter-spacing: 1px; }
.ltr-hero p  { color: #dbe6ff; margin: 0.9rem auto 0; font-size: 1.02rem; max-width: 46em; line-height: 1.7; }
.ltr-section-title {
    font-size: 1.35rem; font-weight: 700; margin: 1.6rem 0 0.9rem;
    color: #1e3c72; border-left: 5px solid #2a5298; padding-left: 0.6rem;
}
.ltr-card {
    background: #ffffff; border: 1px solid #e6ecf5; border-radius: 12px;
    padding: 1.1rem 1.05rem 1.2rem; box-shadow: 0 2px 8px rgba(30, 60, 114, 0.08);
    height: 100%; box-sizing: border-box;
}
.ltr-card .icon { font-size: 1.6rem; }
.ltr-card h3 { color: #1e3c72; font-size: 1.02rem; margin: 0.45rem 0 0.5rem; }
.ltr-card p  { color: #4a5568; font-size: 0.88rem; margin: 0; line-height: 1.65; }
.ltr-step {
    background: #f7faff; border: 1px solid #dbe6f7; border-radius: 12px;
    padding: 1.1rem 1.05rem 1.2rem; height: 100%; box-sizing: border-box;
}
.ltr-step-num {
    display: inline-block; min-width: 2rem; height: 2rem; line-height: 2rem;
    border-radius: 50%; background: #2a5298; color: #ffffff; text-align: center;
    font-weight: 700; margin-bottom: 0.5rem; padding: 0 0.25rem;
}
.ltr-step h3 { color: #1e3c72; font-size: 1rem; margin: 0.3rem 0 0.5rem; }
.ltr-step p  { color: #4a5568; font-size: 0.86rem; margin: 0; line-height: 1.65; }
.ltr-step .io { color: #718096; font-size: 0.8rem; margin-top: 0.5rem; }
.ltr-arrow { color: #2a5298; font-size: 1.5rem; font-weight: 700; text-align: center; padding-top: 3.2rem; }
.ltr-note {
    background: #eef4ff; border: 1px dashed #9db8e8; border-radius: 10px;
    color: #33507e; font-size: 0.88rem; padding: 0.7rem 1rem; margin: 0.4rem 0 1rem; line-height: 1.6;
}
</style>
"""

_FEATURES = [
    ("🧠", "术语抽取", "基于大模型从文献标题、摘要中自动提取科技术语，支持 CSV / Excel 批量处理，"
                     "多线程并发并带限流重试与断点续传。"),
    ("🕸️", "聚类及可视化", "基于术语共现关系构建相似度矩阵并聚类，生成可交互的术语关系网络图，"
                        "节点个数、迭代参数等均可调节。"),
    ("🏷️", "主题标签揭示", "为每个聚类自动生成主题标签，概括类别、揭示领域研究热点，"
                       "并支持生成带标签的网络可视化图表。"),
    ("🗂️", "结果导出与追溯", "各步骤结果均可下载（txt / json），处理失败的记录单独导出，"
                        "便于核对与复用。"),
]

_STEPS = [
    ("1", "术语抽取",
     "上传 CSV / Excel 文献数据，配置大模型参数后运行，自动提取每条文献的科技术语。",
     "输入：文献数据（含标题、可选摘要）　→　输出：extracted_terms.txt"),
    ("2", "聚类及可视化",
     "上传第一步的术语文件（或现有关键词列表），构建共现矩阵并聚类，生成网络图。",
     "输入：extracted_terms.txt　→　输出：cluster.txt、graph_data.json"),
    ("3", "主题标签揭示",
     "上传第二步的聚类结果生成主题标签，再上传节点关系数据生成带标签的可视化图表。",
     "输入：cluster.txt + graph_data.json　→　输出：主题标签结果"),
]

_IMAGE_CAPTIONS = [
    ('background.png', '研究背景'),
    ('method_route.png', '方法路线'),
    ('software_effect.png', '软件效果'),
]


def render():
    """渲染首页"""
    st.markdown(_CSS, unsafe_allow_html=True)

    # 顶部横幅
    st.markdown(
        """
        <div class="ltr-hero">
            <h1>科技文献聚类与主题标签揭示软件</h1>
            <p>基于大模型术语抽取与共现聚类的文献主题识别系统：三步流水线完成
            “术语抽取 → 聚类可视化 → 主题揭示”，帮助快速发现研究热点。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 主要功能
    st.markdown('<div class="ltr-section-title">主要功能</div>', unsafe_allow_html=True)
    cols = st.columns(len(_FEATURES))
    for col, (icon, title, desc) in zip(cols, _FEATURES):
        with col:
            st.markdown(
                f'<div class="ltr-card"><div class="icon">{icon}</div>'
                f'<h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    # 使用步骤
    st.markdown('<div class="ltr-section-title">使用步骤</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ltr-note">三步流水线：每一步的输出即为下一步的输入。'
        '各步骤页面内均提供<b>数据样例下载</b>、<b>数据格式说明</b>与<b>操作说明</b>，'
        '可在左侧边栏切换步骤。</div>',
        unsafe_allow_html=True,
    )

    step_cols = st.columns(5)
    for idx, col in enumerate(step_cols):
        with col:
            if idx % 2 == 0:
                step = _STEPS[idx // 2]
                st.markdown(
                    f'<div class="ltr-step"><span class="ltr-step-num">{step[0]}</span>'
                    f'<h3>{step[1]}</h3><p>{step[2]}</p>'
                    f'<p class="io">{step[3]}</p></div>',
                    unsafe_allow_html=True,
                )
            elif idx < 4:
                st.markdown('<div class="ltr-arrow">➜</div>', unsafe_allow_html=True)

    # 项目图示
    st.markdown('<div class="ltr-section-title">项目图示</div>', unsafe_allow_html=True)
    img_cols = st.columns(len(_IMAGE_CAPTIONS))
    for col, (img_name, caption) in zip(img_cols, _IMAGE_CAPTIONS):
        img_path = os.path.join(IMAGE_DIR, img_name)
        with col:
            if os.path.exists(img_path):
                st.image(img_path, caption=caption, use_container_width=True)
            else:
                st.info(f'缺少图片：{img_name}')
