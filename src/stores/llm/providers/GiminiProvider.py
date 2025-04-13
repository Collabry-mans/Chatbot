from ..LLMInterface import LLMInterface
from ..LLMEnum import DocumentTypeEnum
import google.generativeai as genai
import logging

class GIMINIProvider(LLMInterface):
    def __init__(self, api_key: str,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = "models/gemini-pro"
        self.embedding_model_id = "models/embedding-001"
        self.embedding_size = None  
        
        genai.configure(api_key=self.api_key)

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
            model = genai.GenerativeModel(self.generation_model_id)

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

        task_type = "retrieval_document"
        if document_type == DocumentTypeEnum.QUERY:
            task_type = "retrieval_query"

        try:
            model = genai.EmbeddingModel(self.embedding_model_id)
            response = model.embed_content(
                content=self.process_text(text),
                task_type=task_type
            )
            return response.embedding
        except Exception as e:
            self.logger.error(f"Error during text embedding: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": self.process_text(prompt)
        }
