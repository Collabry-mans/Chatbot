from ..LLMInterface import LLMInterface
from ..LLMEnum import DocumentTypeEnum
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage
from helpers.config import get_settings
from langgraph.graph import START,MessagesState,StateGraph,END
import logging
import os

settings=get_settings()
os.environ["GOOGLE_API_KEY"] =  settings.GOOGLE_API_KEY

class GIMINIProvider(LLMInterface):
    def __init__(self, api_key: str,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 generation_model_id:int=None,
                 embedding_model_id:str=None,
                 default_generation_temperature: float = 0.1):
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = generation_model_id
        self.embedding_model_id = embedding_model_id
        self.embedding_size = None  
        

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list = None,
                      max_output_tokens: int = None, temperature: float = None):
        
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        try:
            model = ChatGoogleGenerativeAI(
                model=self.generation_model_id,
                temperature=0.7,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            if chat_history:
                chat = model.start_chat(history=chat_history)
                response = chat.send_message(self.process_text(prompt),
                                             generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens})
                return response.text
            else:
                response = model.generate_content(self.process_text(prompt),
                                                  generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens})
                return response.text
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        if not self.embedding_model_id:
            self.logger.error("Embedding model ID is not set.")
            return None

        try:
            model = SentenceTransformer(self.embedding_model_id)
            response = model.encode(text)
            return response
        except Exception as e:
            self.logger.error(f"Error during text embedding: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": self.process_text(prompt)
        }
