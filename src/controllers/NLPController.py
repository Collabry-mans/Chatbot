from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnum import DocumentTypeEnum
from typing import List
import json

class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client):
        super().__init__()
        self.collection_name=self.app_settings.VECTOR_DB_COLLECTION_NAME
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self):
        return self.vectordb_client.delete_collection(collection_name=self.collection_name)
    
    def get_vector_db_collection_info(self):
        collection_info = self.vectordb_client.get_collection_info(collection_name=self.collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        

        # step2: manage items
        texts = [ c["chunk_text"] for c in chunks ]
        metadata = [ c["chunk_metadata"] for c in  chunks]
        vectors = self.embedding_client.embed_text(text=texts, 
                                            document_type=DocumentTypeEnum.DOCUMENT.value)
        
        # step3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=self.collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        _ = self.vectordb_client.insert_many(
            collection_name=self.collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            doc_ids=chunks_ids,
        )

        return True
    def search_vector_db_collection(self, text: str, limit: int = 10):


        vector=[]
        # step2: get text embedding vector
        vector = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if len(vector) == 0:
            return False

        # step3: do semantic search
        results = self.vectordb_client.search_by_vector(
            collection_name=self.collection_name,
            vector=vector,
            limit=limit
        )

        if not results:
            return False
        
        context=[]
        for hits in results[0]:
   
            context.append(hits["entity"]['text'])

        return context
    
    def delete_file_from_vectorDB_by_ID(self,project_id:str):
        ans=self.vectordb_client.delete_document_by_id(
            collection_name=self.collection_name,
            doc_id=project_id
        )
        return ans
    
    def get_chatbot_answer(self,prompt,user_id,context,chat_history_manager):
        ans=self.generation_client.generate_text(
            prompt=prompt,
            user_id=user_id,
            context=context,
        )
        return ans