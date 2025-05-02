from fastapi import FastAPI,APIRouter,Depends,UploadFile,status,Request
from fastapi.responses import JSONResponse
from .schemes.data import processRequest
from helpers.config import get_settings,Settings
from models.enums import ResponseSignal
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemes import DataChunk,Asset
from models.AssetModel import AssetModel
from models.enums.AssetTypeEnum import AssetTypeEnum
from controllers import DataController,ProjectController,ProcessController,NLPController
import aiofiles
import logging
import os

logger=logging.getLogger("uvicorn.error")


data_router=APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1","data"]
)

@data_router.post("/upload_and_process/{project_id}")
async def upload_and_process_data(
    request: Request,
    project_id: str,
    file: UploadFile,
    chunk_size: int = 1000,
    overlap_size: int = 50,
    app_settings: Settings = Depends(get_settings)
):

    # Validate file
    data_controller = DataController()
    is_valid, signal = data_controller.validate_uploaded_file(file=file)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.FILE_UPLOADED_FAILED.value}
        )

    # Create project file path
    file_path, file_id = data_controller.generate_unique_filepath(
        original_file_name=file.filename,
        project_id=project_id
    )

    try:
        # Save file to disk
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.FILE_UPLOADED_FAILED.value}
        )

    # Process file
    process_controller = ProcessController(project_id=project_id)
    file_content = process_controller.get_file_content(file_id=file_id)
    if file_content is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROCESSING_FAILD.value}
        )

    file_chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id=file_id,
        chunk_size=chunk_size,
        overlap=overlap_size
    )
    if not file_chunks:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROCESSING_FAILD.value}
        )

    # Prepare response chunks
    chunks_data = [
        {
            "chunk_text": chunk.page_content,
            "chunk_metadata": chunk.metadata,
            "chunk_order": i + 1
        }
        for i, chunk in enumerate(file_chunks)
    ]
    chunks=[
        chunk.page_content 
        for i,chunk in enumerate(file_chunks)
    ]
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.PROCESSING_SUCCESS.value,
            "file_id": file_id,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
    )

@data_router.post("/upload_process_index/{project_id}")
async def upload_process_and_index(
    request: Request,
    project_id: str,
    file: UploadFile,
    chunk_size: int = 500,
    overlap_size: int = 50,
    do_reset: int = 0,  # 1 means reset, 0 means append
    app_settings: Settings = Depends(get_settings)
):
    

    # Step 2: Validate File
    data_controller = DataController()
    is_valid, signal = data_controller.validate_uploaded_file(file=file)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.FILE_UPLOADED_FAILED.value}
        )

    # Step 3: Save File Temporarily
    file_path, file_id = data_controller.generate_unique_filepath(
        original_file_name=file.filename,
        project_id=project_id
    )
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.FILE_UPLOADED_FAILED.value}
        )

    # Step 4: Process File Content into Chunks
    process_controller = ProcessController(project_id=project_id)
    file_content = process_controller.get_file_content(file_id=file_id)
    if file_content is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROCESSING_FAILD.value}
        )

    file_chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id=file_id,
        chunk_size=chunk_size,
        overlap=overlap_size
    )
    file_chunks = [
        {
            "chunk_text": chunk.page_content,
            "chunk_metadata": chunk.metadata,
            "chunk_order": i + 1
        }
        for i, chunk in enumerate(file_chunks)
    ]
    if not file_chunks:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROCESSING_FAILD.value}
        )

    # Step 5: Directly Index Into Vector DB
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client
    )

    chunks_ids = [project_id for _ in  range(len(file_chunks))]
    
    is_inserted = nlp_controller.index_into_vector_db(

        chunks=file_chunks,
        do_reset=do_reset,
        chunks_ids=chunks_ids
    )

    if not is_inserted:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_success.value,
            "file_id": file_id,
            "total_chunks": len(file_chunks)
        }
    )
