import pandas as pd
from utils.llm_utils import LLMClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class TermExtractor:
    def __init__(self, api_key, base_url, model_name):
        self.llm_client = LLMClient(api_key, base_url, model_name).client
    
    def extract_terms(self, file):
        # 读取CSV文件
        df = pd.read_csv(file)
        
        # 存储提取结果
        extracted_terms = []
        
        prompt = ChatPromptTemplate.from_messages([("system", self.get_template()),("user","标题：{title}\n摘要：{abstract}")])
        chain = prompt | self.llm_client | StrOutputParser()

        # 对每篇文献进行处理
        for _,row in df.iterrows():
            response = chain.invoke({"title": row["title"], "abstract": row["abstract"]})
            extracted_terms.append(response.strip())
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'terms': extracted_terms
        })
        
        return results 
    
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
摘要： "本文提出了一种基于Transformer架构的图像分割方法。通过自注意力机制，该方法能够有效地捕捉图像中的全局和局部特征。实验结果表明，该方法在分割精度和计算效率方面均优于传统方法。此外，我们还探讨了模型的可扩展性和实时应用能力。"
### 输出：
计算机视觉;图像分割技术;Transformer架构;自注意力机制;分割精度;计算效率;模型可扩展性;实时应用能力

以下是我的输入：
### 输入：
标题： {title}
摘要： {abstract}

### 输出：
    '''