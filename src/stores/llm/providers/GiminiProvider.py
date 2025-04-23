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
        self.embedding_model_id = config.get("embedding_model_id")

        self.llm = ChatGoogleGenerativeAI(
            model=self.generation_model_id,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def call_llm(self,state: MessagesState):
        """Execute the LLM with current state"""
        chain = collabry_prompt | self.llm
        result = chain.invoke(state["messages"])
        return {"messages": [AIMessage(content=result.content, name="Collabry chatbot")]}

    def create_graph(self):
        """Create the LangGraph workflow without checkpointer"""
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("collabry", self.call_llm)
        workflow.add_edge(START, "collabry")
        workflow.add_edge("collabry", END)
        return workflow.compile()


    def generate_text(
        self,
        prompt: str,
        user_id: str,
        context:List=None,
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
            history_manager = ChatHistoryManager()
    
            # Load existing history
            history = history_manager.get_conversation(user_id=user_id)
            messages = [
                HumanMessage(content=msg["content"]) if msg["role"] == "human" 
                else AIMessage(content=msg["content"])
                for msg in history
            ]
            if context:
                    context_str = "\n\n".join(
                        f"[Context {i+1}]: {text[:500]}{'...' if len(text) > 500 else ''}"
                        for i, text in enumerate(context)
                    )
            query = f"Context:\n{context_str}\n\nQuestion: {prompt}"
            # Add current message
            messages.append(HumanMessage(content=query))
            
            # Create and execute graph
            graph = self.create_graph()
            res = graph.invoke({"messages": messages})
            
            # Save updated history
            new_history = [
                {"role": "human" if isinstance(m, HumanMessage) else "ai", 
                "content": m.content}
                for m in messages + [res["messages"][-1]]]
            history_manager.add_message(user_id=user_id, message=new_history)
            
            return res["messages"][-1].content
        
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

