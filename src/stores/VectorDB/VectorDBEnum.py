from enum import Enum

class VectorDBEnums(Enum):
    MILVUS="MILVUS"

class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"