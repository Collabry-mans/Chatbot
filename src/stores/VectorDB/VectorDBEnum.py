from enum import Enum

class VectorDBEnums(Enum):
    MILVUS="MILVUS"

class DistanceMethodEnums(Enum):
    COSINE = "COSINE"
    DOT = "DOT"
    L2="l2"