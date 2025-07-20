# llm_huggingface.py
from langchain_core.runnables import Runnable
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceChatLLM(Runnable):
    def __init__(self, model: str = "moonshotai/Kimi-K2-Instruct"):
        self.client = InferenceClient(
            provider="novita",
            api_key=os.environ["HF_TOKEN"]
        )
        self.model = model

    def invoke(self, input_text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": input_text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("LLM Error:", e)
            return "Error generating response."
