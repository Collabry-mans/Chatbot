from enum import Enum

class ResponseSignal(Enum):
    FILE_VALIDATED_SUCCESS="file_validate_successfully"
    FILE_TYPE_NOT_SUPPORTED="file_type_not_supported"
    FILE_SIZE_EXCEEDED="file_size_exceeded"
    FILE_UPLOADED_SUCCESS="file_uploaded_success"
    FILE_UPLOADED_FAILED="file_uploaded_failed"

    PROCESSING_FAILD="processing_faild"
    PROCESSING_SUCCESS="prosessing_success"

    NO_FILES_ERROR="not_found_files"
    FILE_ID_ERROR="no_file_found_with_this_id"
    PROJECT_NOT_FOUND_ERROR="project_not_found"
    INSERT_INTO_VECTORDB_ERROR="insert_into_vector_db_error"
    INSERT_INTO_VECTORDB_success="insert_into_vector_db_success"
    VECTORDB_COLLECTION_RETRIEVED="vectordb_collection_retrieved"
    VECTORDB_SEARCH_ERROR="vectordb_search_error"
    VECTORDB_SEARCH_SUCCESS="vectordb_search_success"