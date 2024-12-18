import pandas as pd
from utils.llm_utils import LLMClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class TermExtractor:
    def __init__(self, api_key, base_url, model_name):
        self.llm_client = LLMClient(api_key, base_url, model_name).client
        self.error_records = []  # 存储出错的记录
        self.extracted_terms = []  # 存储已处理的结果
        self.error_indices = []  # 存储出错的行索引
    
    def extract_terms_single(self, row):
        """处理单条记录"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_template()),
            ("user", f"标题：{row['title']}\n摘要：{row['abstract']}")
        ])
        chain = prompt | self.llm_client | StrOutputParser()
        response = chain.invoke({"title": row["title"], "abstract": row["abstract"]})
        return response.strip()

    def log_error(self, index, row, error):
        """记录错误信息"""
        self.error_indices.append(index)
        self.error_records.append({
            'index': index,
            'title': row["title"],
            'abstract': row["abstract"],
            'error': error
        })

    def extract_terms(self, df, progress_callback=None):
        self.extracted_terms = []
        self.error_indices = []
        self.error_records = []
        
        prompt = ChatPromptTemplate.from_messages([("system", self.get_template()),("user","标题：{title}\n摘要：{abstract}")])
        chain = prompt | self.llm_client | StrOutputParser()

        for i, row in df.iterrows():
            # 检查停止标志
                
            try:
                response = chain.invoke({"title": row["title"], "abstract": row["abstract"]})
                self.extracted_terms.append(response.strip())
                
                if progress_callback:
                    progress_callback(i)
                    
            except Exception as e:
                # 记录出错的行
                self.error_indices.append(i)
                self.error_records.append({
                    'index': i,
                    'title': row["title"],
                    'abstract': row["abstract"],
                    'error': str(e)
                })
                # 为保持数组长度一致，添加空值
                self.extracted_terms.append("")
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'terms': self.extracted_terms
        })
        
        return results, self.error_indices
    
    def get_partial_results(self):
        """返回已处理的部分结果"""
        return pd.DataFrame({
            'terms': self.extracted_terms
        }), self.error_indices
    
    def get_error_records_df(self):
        """返回出错记录的DataFrame"""
        return pd.DataFrame(self.error_records)
    
    def get_template(self):
        return '''你是一个科技术语抽取的专业人员，请遵循以下指令来工作。
### 指令：
从给定的论文标题和摘要中提取出与科学技术相关的关键词。
抽取过程中注意以下事项：
1.尽可能的抽取长词。
2.输出严格按照样例的格式。
3.仅参考样例的格式，不受到样例内容的影响。
以下是两个案例：
案例1：
### 输入：
标题： "深度学习在医学影像中的应用"
摘要： "本文全面回顾了深度学习技术在医学影像中的应用。我们讨论了各种架构，如卷积神经网络（CNNs）和生成对抗网络（GANs），以及它们在诊断、治疗规划和预后中的应用。研究还探讨了该领域的挑战和机遇，包括数据隐私、模型可解释性和实时处理。"
### 输出：
深度学习;医学影像;卷积神经网络（CNNs）;生成对抗网络（GANs）;诊断;治疗规划;预后;数据隐私;模型可解释性;实时处理
案例2：
### 输入：
标题： "基于Transformer的图像分割技术研究"
摘要： "本文提出了一基于Transformer架构的图像分割方法。通过自注意力机制，该方法能够有效地捕捉图像中的全局和局部特征。实验结果表明，该方法在分割精度和计算效率方面均优于传统方法。此外，我们还���讨了模型的可扩展性和实时应用能力。"
### 输出：
计算机视觉;图像分割技术;Transformer架构;自注意力机制;分割精度;计算效率;模型可扩展性;实时应用能力

以下是我的输入：
### 输入：
标题： {title}
摘要： {abstract}

### 输出：
    '''