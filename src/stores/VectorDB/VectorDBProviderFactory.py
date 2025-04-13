from .providers import MilvusDBProvider
from VectorDBEnum import VectorDBEnums

class VectorDBProviderFactory:
    def __init__(self,config):
        self.config=config

    def create(self,provider:str):
        if provider==VectorDBEnums.MILVUS.value:
            return MilvusDBProvider(
                db_path=self.config.VECTOR_DB_PATH,
                token=self.config.VECTOR_DB_TOKEN,
                distance_method=None
            )