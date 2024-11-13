import pandas as pd
from utils.llm_utils import LLMClient

class TopicLabeler:
    def __init__(self, api_key, base_url, model_name):
        self.llm_client = LLMClient(api_key, base_url, model_name)
    
    def label_topics(self, cluster_file, prompt_template):
        # 读取聚类结果
        clusters = {}
        with open(cluster_file, 'r', encoding='utf-8') as f:
            for line in f:
                cluster_id, terms = line.strip().split('\t')
                clusters[cluster_id] = terms
        
        # 存储标签结果
        labels = []
        cluster_ids = []
        term_lists = []
        
        # 对每个聚类进行处理
        for cluster_id, terms in clusters.items():
            prompt = prompt_template.format(terms=terms)
            label = self.llm_client.get_completion(prompt)
            
            labels.append(label.strip())
            cluster_ids.append(cluster_id)
            term_lists.append(terms)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'cluster_id': cluster_ids,
            'topic_label': labels,
            'terms': term_lists
        })
        
        return results 