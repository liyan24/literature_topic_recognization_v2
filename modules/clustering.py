import pandas as pd
import numpy as np
import networkx as nx
from modules import community_vos as community
import json
from streamlit_echarts import st_echarts
from sklearn import preprocessing
from collections import defaultdict

class Clusterer:
    def __init__(self, num_nodes=300, num_relations=500, random_seed=0,
                 decay_steps=1, max_iter=100, initial_step=1.0,
                 step_decay=0.75, step_convergence=0.001):
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.decay_steps = decay_steps
        self.max_iter = max_iter
        self.initial_step = initial_step
        self.step_decay = step_decay
        self.step_convergence = step_convergence
    
    def get_matrix(self, terms_df):
        """Convert terms to co-occurrence matrix with frequency counts"""
        # Count term frequencies
        dict_word_freq = defaultdict(int)
        for terms in terms_df['terms']:
            term_list = terms.split(';')
            for term in term_list:
                if term:
                    dict_word_freq[term.strip()] += 1
        
        # Select top frequency terms
        sorted_dict = sorted(dict_word_freq.items(), key=lambda item: item[1], reverse=True)
        freq = sorted_dict[self.num_nodes - 1][1] if len(dict_word_freq) >= self.num_nodes else sorted_dict[-1][1]
        word_list = [item[0] for item in sorted_dict if item[1] >= freq]
        freqs = [item[1] for item in sorted_dict if item[1] >= freq]
        
        # Build co-occurrence matrix
        dict_k_w = defaultdict(int)
        matrix = pd.DataFrame(np.zeros((len(word_list), len(word_list))), columns=word_list, index=word_list)
        
        for terms in terms_df['terms']:
            term_list = terms.split(';')
            term_list = list(set(term_list) & set(word_list))
            for i in range(len(term_list) - 1):
                for j in range(i + 1, len(term_list)):
                    key = frozenset([term_list[i].strip(), term_list[j].strip()])
                    dict_k_w[key] += 1
        
        for key, value in dict_k_w.items():
            key = list(key)
            matrix.loc[key[0], key[1]] = value
            matrix.loc[key[1], key[0]] = value
            
        # Remove isolated nodes and normalize
        matrix = matrix.loc[(matrix.sum(axis=1) != 0), (matrix.sum(axis=0) != 0)]
        series_a = matrix.sum(axis=1)
        for index in matrix.index:
            for col in matrix.columns:
                matrix.loc[index, col] = matrix.loc[index, col] / (series_a[index] * series_a[col])
                
        labels = dict(enumerate(matrix.index))
        freqs = [dict_word_freq[term] for term in matrix.index]
        
        return matrix, labels, freqs

    def keep_norm(self, init_data, k=1):
        """Normalize the coordinate data by calculating pairwise distances and scaling
        
        Args:
            init_data: Input coordinate array of shape (n_samples, n_features)
            k: Normalization factor, defaults to 1
            
        Returns:
            Normalized coordinate array
        """
        # Calculate pairwise distances using vectorized operations
        diffs = init_data[:, np.newaxis] - init_data  # Shape: (n,n,2)
        norms = np.linalg.norm(diffs, axis=2)  # Shape: (n,n)
        
        # Sum upper triangle only since matrix is symmetric
        total = np.sum(np.triu(norms)) / k
        
        return init_data / total if total != 0 else init_data

    def train_model(self, matrix):
        """Train model to get node coordinates"""
        # 使用numpy数组替代pandas操作以提高性能
        matrix_values = matrix.values
        
        # 初始化坐标
        np.random.seed(self.random_seed)
        init_data = np.random.rand(matrix.shape[0], 2)
        n_pairs = init_data.shape[0] * (init_data.shape[0] - 1) / 2.0
        init_data = self.keep_norm(init_data, n_pairs)

        # 预计算矩阵行和，避免重复计算
        row_sums = matrix_values.sum(axis=1)
        
        for k in range(self.max_iter):
            # 计算学习率
            new_rate = self.initial_step * self.step_decay ** (k / self.decay_steps)
            rate = max(new_rate, 0.0001)
            
            # 向量化计算，替代循环
            tmp_data = np.dot(matrix_values, init_data) / row_sums[:, np.newaxis]
            init_data = init_data - rate * (init_data - tmp_data)
            
            # 标准化
            init_data = self.keep_norm(init_data, n_pairs)
            
            if rate <= self.step_convergence:
                break
                
        return init_data
    
    def create_visualization_data(self, matrix, coordinate, labels, freqs):
        """Create visualization data"""
        nodes = []
        links = []
        norm_freqs = preprocessing.scale(freqs)
        size_freqs = preprocessing.minmax_scale(freqs)

        # Create NetworkX graph from matrix
        G = nx.Graph()
        total = matrix.shape[0]
        for i in range(total - 1):
            for j in range(i + 1, total):
                if matrix.iloc[i, j]:
                    G.add_edge(labels[i], labels[j], weight=matrix.iloc[i, j])

        # Detect communities
        clusters = community.best_partition(G, lamda=0.001)
        categories = set(clusters.values())

        # Create node data
        for k, v in labels.items():
            nodes.append({
                "x": coordinate[k][0],
                "y": coordinate[k][1],
                "id": v,
                "name": v,
                "category": clusters[v],
                "symbolSize": 15 + 10 * round(norm_freqs[k], 2),
                "label": {
                    "fontSize": 10 + 10 * round(size_freqs[k], 2)
                }
            })

        # Create edge data
        for u, v, data in G.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "value": data['weight']
            })

        # Sort links by weight and limit the number
        links = sorted(links, key=lambda item: item['value'], reverse=True)[:self.num_relations]
        categories = [{"name": f"类别{i}"} for i in categories]

        # Create echarts configuration
        option = {
            'title': {'text': '术语关系网络'},
            'tooltip': {},
            'legend': {'data': [cat['name'] for cat in categories]},
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
                'data': nodes,
                'links': links,
                'categories': categories,
                'roam': True,
                'labelLayout': {
                    'hideOverlap': True
                }
            }]
        }
        
        return option, clusters,nodes,links
    
    def process(self, terms_file):
        """Process the input terms file and generate visualization data"""
        # Read terms data
        terms_data = []
        for line in terms_file:
            terms = line.decode('utf-8').strip()
            if terms:  # Skip empty lines
                terms_data.append({'terms': terms})
        terms_df = pd.DataFrame(terms_data)
        
        # Generate matrix and coordinates
        matrix, labels, freqs = self.get_matrix(terms_df)
        coordinate = self.train_model(matrix)
        
        # Create visualization data
        matrix = matrix / 10.0  # Scale down weights for visualization
        visualization, clusters,nodes,links = self.create_visualization_data(matrix, coordinate, labels, freqs)
        
        # Prepare output data
        graph_data = {
            'nodes': nodes,
            'links': links
        }
        

        
        # Convert results to strings
        graph_data_str = json.dumps(graph_data, ensure_ascii=False)
        
        # Create cluster results
        cluster_results = {f"cluster_{v}": [] for v in set(clusters.values())}
        for node, cluster_id in clusters.items():
            cluster_results[f"cluster_{cluster_id}"].append(node)
        cluster_results_str = '\n'.join([f"{k}\t{';'.join(v)}" for k, v in cluster_results.items()])
        
        return graph_data_str, cluster_results_str, visualization