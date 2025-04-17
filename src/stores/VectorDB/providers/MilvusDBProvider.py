from pymilvus import Collection, connections,MilvusClient,DataType
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnum import DistanceMethodEnums
import logging
from typing import List
import json

class MilvusDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str,token:str):

        self.client = None
        self.db_path = db_path
        self.distance_method = None
        self.token=token
        if distance_method not in [DistanceMethodEnums.COSINE.value, 
                                   DistanceMethodEnums.DOT.value,
                                   DistanceMethodEnums.L2.value]:
            raise ValueError(f"Invalid distance method: {distance_method}")
        self.distance_method = distance_method
        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = MilvusClient(
        uri=self.db_path,
        token=self.token
         )
        
    def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        res = self.client.list_collections()
        if collection_name in res:
            return True
        else:
            return False
    
    def list_all_collections(self) -> List:
        return self.client.list_collections()
    
    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.describe_collection(
            collection_name=collection_name
        )

    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            self.client.drop_collection(
                collection_name=collection_name
            )
            return True
        return False
        
    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            schema=MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="metadata",datatype=DataType.JSON)
            schema.add_field(field_name="record_id",datatype=DataType.VARCHAR)
            schema.add_field(field_name="doc_id",datatype=DataType.VARCHAR)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=embedding_size)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1050)
            index_params = self.client.prepare_index_params()

            index_params.add_index(
                field_name="record_id",
                index_type="AUTOINDEX"
            )

            index_params.add_index(
                field_name="vector", 
                index_type="IVF_FLAT",
                metric_type=self.distance_method
            )
            self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
             )
            return True
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         doc_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        try:
            data=[
                {
                    "doc_id":doc_id,
                    "metadata":json.dumps(metadata),
                    "text":text,
                    "vector":vector
                }
            ]
            res=self.client.insert(
                collection_name=collection_name,
                data=data
            )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True
    
    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadata: list = None,
                    doc_ids: list = None, batch_size: int = 50):

        if metadata is None:
            metadata = [None] * len(texts)

        if doc_ids is None:
            return False

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_ids = doc_ids[i:batch_end]

            # Build data as list of dictionaries
            batch_data = [
                {
                    "doc_id": batch_ids[j],
                    "vector": batch_vectors[j],
                    "text": batch_texts[j],
                    "metadata": batch_metadata[j]
                }
                for j in range(len(batch_texts))
            ]

            try:
                self.client.insert(collection_name=collection_name, data=batch_data)
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        return True
    def delete_document_by_id(self,collection_name:str,doc_id:str):
        expre=f"doc_id=={doc_id}"
        results=self.client.delete(
            collection_name=collection_name,
            filter=expre)
        if results["delete_cnt"]==0:
            return False
        else:
            return True
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):

        return self.client.search(
            collection_name=collection_name,
            data=[vector],
            anns_field="vector",
            limit=limit,
            search_params=self.distance_method
        )