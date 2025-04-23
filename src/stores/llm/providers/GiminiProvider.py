from ..LLMInterface import LLMInterface
from ..LLMEnum import DocumentTypeEnum
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage,HumanMessage
from langchain_core.messages import SystemMessage
from typing import Optional, Dict, List, Any
from langgraph.graph import Graph, END
from .Prompt import collabry_prompt
from langgraph.prebuilt import ToolNode
from ...ChatHistoryManager import ChatHistoryManager
from helpers.config import get_settings
from ..GenerationScheme.GenerationScheme import GenerationConfig
from langgraph.graph import START,MessagesState,StateGraph,END
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os

settings=get_settings()
os.environ["GOOGLE_API_KEY"] =  settings.GOOGLE_API_KEY

class GIMINIProvider(LLMInterface):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text generator with configuration.
        
        Args:
            config: Dictionary containing:
                - generation_model_id: ID of the generation model
                - default_max_tokens: Default maximum tokens (default: 2048)
                - default_temperature: Default temperature (default: 0.7)
                - embedding_model_id: ID of the embedding model
        """
        self.generation_model_id = config.get("generation_model_id")
        self.default_config = GenerationConfig(
            max_output_tokens=config.get("default_max_tokens", 2048),
            temperature=config.get("default_temperature", 0.7)
        )
        self.embedding_model_id = config.get("embedding_model_id")
        self.logger = logging.getLogger(__name__)
        self._workflow = self._build_workflow()

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    
    def _initialize_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the language model"""
        config = state.get("config", self.default_config)
        
        model = ChatGoogleGenerativeAI(
            model=self.generation_model_id,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            timeout=None,
            max_retries=2,
        )
        
        return {"model": model, "config": config}
    
    def _prepare_history(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare chat history for the model"""
        chat_history = state.get("chat_history", [])
        
        messages = [
            AIMessage(content=msg["content"]) if msg["role"] == "ai" else
            SystemMessage(content=msg["content"]) if msg["role"] == "system" else
            HumanMessage(content=msg["content"])
            for msg in chat_history
        ]
        
        return {"prepared_messages": messages}
    
    def _format_context(self, documents: List) -> str:
        """Convert retrieved docs into numbered citations"""
        if not documents:
            return ""
        
        context_lines = ["**Relevant Context:**"]
        for i, doc in enumerate(documents, 1):
            context_lines.append(
                f"[context{i}] {doc})"
            )
        return "\n".join(context_lines)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response from the model"""
        model = state["model"]
        messages = state["prepared_messages"]
        prompt = state["prompt"]
        config = state["config"]
        context = state.get("context", "")

        # Format the full prompt with context
        full_prompt = f"{context}\n\nQuestion: {prompt}" if context else prompt
    
        # Add the current prompt to messages
        messages.append(HumanMessage(content=full_prompt))
        messages.append(collabry_prompt)
        
        # Generate response
        response = model.invoke(
            messages,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
                "top_p": config.top_p,
                "top_k": config.top_k
            }
        )
        
        return {"response": response.content}

    def _update_history(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update chat history if manager is available"""
        if "chat_history_manager" in state and "user_id" in state:
            manager = state["chat_history_manager"]
            user_id = state["user_id"]
            prompt = state["prompt"]
            response = state["response"]
            
            manager.add_message(user_id, "user", prompt)
            manager.add_message(user_id, "ai", response)
        
        return {"status": "completed"}   
     
    def _handle_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors in the workflow"""
        error = state.get("error")
        self.logger.error(f"Error in generation workflow: {error}")
        return {"response": None, "status": "failed"}
    

    def _build_workflow(self) -> Graph:
        """Build the LangGraph workflow"""
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("initialize_model", self._initialize_model)
        workflow.add_node("prepare_history", self._prepare_history)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("update_history", self._update_history)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define edges
        workflow.add_edge("initialize_model", "prepare_history")
        workflow.add_edge("prepare_history", "generate_response")
        workflow.add_edge("generate_response", "update_history")
        workflow.add_edge("update_history", END)
        
        # Set entry point
        workflow.set_entry_point("initialize_model")
        
        return workflow.compile()
    
    def generate_text(
        self,
        prompt: str,
        user_id: str,
        context:List=None,
        chat_history_manager: Optional[ChatHistoryManager] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> Optional[str]:
        """
        Generate text using the configured language model.
        
        Args:
            prompt: The input prompt/text
            user_id: ID of the user making the request
            chat_history_manager: Optional chat history manager
            generation_config: Optional generation configuration
            
        Returns:
            Generated text or None if generation fails
        """
        if not self.generation_model_id:
            self.logger.error("Generation model ID is not set.")
            return None

        try:
            # Prepare input state
            state = {
                "prompt": self._process_text(prompt),
                "user_id": user_id,
                "config": generation_config or self.default_config,
                "chat_history": [],
            }
            if context:
                state["context"]=self._format_context(context)
            
            if chat_history_manager:
                state["chat_history_manager"] = chat_history_manager
                state["chat_history"] = chat_history_manager.get_conversation(
                    user_id=user_id,
                    max_messages=10
                )
            
            # Execute workflow
            result = self._workflow.invoke(state)
            
            return result.get("response")
        
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return None

    def _process_text(self, text: str) -> str:
        """Pre-process input text before generation"""
        return text.strip()
                

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




config = {
    "generation_model_id": "gemini-pro",
    "default_max_tokens": 1024,
    "default_temperature": 0.7,
    "logger": logging.getLogger(__name__)
}

# Initialize
generator = GIMINIProvider(config)
history_manager = ChatHistoryManager()

# Example usage
response = generator.generate_text(
    prompt="Tell me about LangChain",
    user_id="user123",
    chat_history_manager=history_manager,
    generation_config=GenerationConfig(
        temperature=0.5,
        max_output_tokens=512
    )
)

print(response)