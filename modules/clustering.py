import pandas as pd
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import json
from streamlit_echarts import st_echarts

class Clusterer:
    def __init__(self, num_nodes=300, num_relations=500, random_seed=0,
                 decay_steps=1, max_iter=100, initial_step=1.0,
                 step_decay=0.75, step_convergence=0.001):
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.random_seed = random_seed
        self.decay_steps = decay_steps
        self.max_iter = max_iter
        self.initial_step = initial_step
        self.step_decay = step_decay
        self.step_convergence = step_convergence
    
    def build_cooccurrence_network(self, terms_df):
        # 构建术语共现网络
        G = nx.Graph()
        term_counts = {}
        
        for terms in terms_df['terms']:
            term_list = terms.split(';')
            # 统计术语频率
            for term in term_list:
                term = term.strip()
                term_counts[term] = term_counts.get(term, 0) + 1
            
            # 构建共现关系
            for i in range(len(term_list)):
                for j in range(i + 1, len(term_list)):
                    term1, term2 = term_list[i].strip(), term_list[j].strip()
                    if G.has_edge(term1, term2):
                        G[term1][term2]['weight'] += 1
                    else:
                        G.add_edge(term1, term2, weight=1)
        
        return G, term_counts
    
    def filter_network(self, G, term_counts):
        # 选择高频节点
        top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:self.num_nodes]
        selected_terms = set(term for term, _ in top_terms)
        
        # 过滤子图
        H = G.subgraph(selected_terms).copy()
        
        # 选择权重最大的边
        edges = sorted(H.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        selected_edges = edges[:self.num_relations]
        
        # 构建最终网络
        final_network = nx.Graph()
        for u, v, data in selected_edges:
            final_network.add_edge(u, v, weight=data['weight'])
        
        return final_network
    
    def community_detection(self, G):
        # 使用Louvain算法进行社区检测
        communities = nx.community.louvain_communities(G)
        
        # 将结果转换为字典格式
        cluster_results = {}
        for i, community in enumerate(communities):
            cluster_results[f"cluster_{i}"] = list(community)
        
        return cluster_results
    
    def layout_network(self, G):
        # 使用MDS进行布局
        adj_matrix = nx.adjacency_matrix(G).todense()
        mds = MDS(n_components=2, random_state=self.random_seed)
        positions = mds.fit_transform(adj_matrix)
        
        # 将布局结果转换为字典
        pos = {node: positions[i].tolist() for i, node in enumerate(G.nodes())}
        return pos
    
    def create_visualization_data(self, G, pos, cluster_results, legend_names):
        nodes = []
        edges = []
        
        # 为每个聚类分配颜色
        cluster_colors = {}
        for i, cluster in enumerate(cluster_results.values()):
            for node in cluster:
                cluster_colors[node] = i
        
        # 添加节点
        for node in G.nodes():
            nodes.append({
                'name': node,
                'x': pos[node][0],
                'y': pos[node][1],
                'category': cluster_colors[node]
            })
        
        # 添加边
        for u, v, data in G.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'value': data['weight']
            })
        
        # 使用自定义的图例名称
        num_clusters = len(cluster_results)
        categories = [{'name': legend_names[i]} for i in range(min(num_clusters, len(legend_names)))]
        
        # 创建echarts配置
        option = {
            'title': {'text': '术语关系网络'},
            'tooltip': {},
            'legend': {'data': [cat['name'] for cat in categories]},
            'series': [{
                'type': 'graph',
                'layout': 'none',
                'data': nodes,
                'links': edges,
                'categories': categories,
                'roam': True,
                'label': {'show': True},
                'force': {'repulsion': 100}
            }]
        }
        
        return option
    
    def process(self, terms_file, legend_names=None):
        # 如果没有提供图例名称，使用默认的
        if legend_names is None:
            legend_names = [f'类别{i}' for i in range(100)]  # 设置足够大的默认值
            
        # 读取术语文件 - 修改为读取txt文件
        terms_data = []
        for line in terms_file:
            terms = line.decode('utf-8').strip()
            if terms:  # 忽略空行
                terms_data.append({'terms': terms})
        terms_df = pd.DataFrame(terms_data)
        
        # 构建网络
        G, term_counts = self.build_cooccurrence_network(terms_df)
        
        # 过滤网络
        G = self.filter_network(G, term_counts)
        
        # 社区检测
        cluster_results = self.community_detection(G)
        
        # 计算布局
        pos = self.layout_network(G)
        
        # 创建可视化数据
        visualization = self.create_visualization_data(G, pos, cluster_results, legend_names)
        
        # 准备输出数据
        graph_data = {
            'nodes': list(G.nodes()),
            'edges': [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        }
        
        # 将结果转换为字符串
        graph_data_str = json.dumps(graph_data, ensure_ascii=False)
        cluster_results_str = '\n'.join([f"{k}\t{';'.join(v)}" for k, v in cluster_results.items()])
        
        return graph_data_str, cluster_results_str, visualization 
    
    def update_visualization(self, visualization, new_legend_names):
        """更新可视化配置中的图例名称"""
        # 创建新的可视化配置（深拷贝避免修改原始数据）
        new_visualization = json.loads(json.dumps(visualization))
        
        # 更新类别名称
        num_categories = len(new_visualization['series'][0]['categories'])
        for i in range(min(num_categories, len(new_legend_names))):
            new_visualization['series'][0]['categories'][i]['name'] = new_legend_names[i]
        
        # 更新图例数据
        new_visualization['legend']['data'] = new_legend_names[:num_categories]
        
        return new_visualization