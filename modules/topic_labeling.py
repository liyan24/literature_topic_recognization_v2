import pandas as pd
from utils.llm_utils import LLMClient

class TopicLabeler:
    def __init__(self, api_key, base_url, model_name):
        self.llm_client = LLMClient(api_key, base_url, model_name)
    
    def label_topics(self, cluster_file):
        # 读取聚类结果
        clusters = {}
        # 修改为直接读取上传的文件对象
        content = cluster_file.read().decode('utf-8')
        for line in content.split('\n'):
            if line.strip():  # 跳过空行
                cluster_id, terms = line.strip().split('\t')
                clusters[cluster_id] = terms
        
        # 存储标签结果
        labels = []
        cluster_ids = []
        term_lists = []
        
        prompt = "你是一位科技情报信息的分析专家\n你的任务是：\n（1）通过给定的一系列的科技文献的聚类结果及类别下的词，概括每个类的类别。\n（2）根据所有类别的结果，概括这个领域的研究热点是什么。\n输入格式为：cluster_x:主题词1、主题词2、...\n输出格式为：cluster_x的主题标签是《主题标签》\n以下是类别及类别下的词：\n\n"

        # 对每个聚类进行处理
        for cluster_id, terms in clusters.items():
            prompt += f"{cluster_id}:{terms}\n"
            
        label = self.llm_client.get_completion(prompt)
        return label 