from langchain_openai import ChatOpenAI

class LLMClient:
    def __init__(self, api_key, base_url, model_name):
        self.client = ChatOpenAI(
            openai_api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )
    
    def get_completion(self, prompt):
        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            raise Exception(f"调用LLM接口失败: {str(e)}") 