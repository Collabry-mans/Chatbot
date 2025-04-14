from fastapi import FastAPI
from routes import base,data
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import Settings,get_settings
from stores.llm.LLMProvierFactory import LLMProviderFactory
from stores.VectorDB.VectorDBProviderFactory import VectorDBProviderFactory
app=FastAPI()

async def startup_span():
    settings=get_settings()
    app.mongo_conn=AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client=app.mongo_conn[settings.MONGODB_DATABASE]

    llm_provier_factory=LLMProviderFactory(settings)
    vectordb_provider_factory=VectorDBProviderFactory(settings)
    app.generation_client=llm_provier_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    app.embedding_client=llm_provier_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    app.vectordb_client=VectorDBProviderFactory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()

async def shutdown_span():
    app.mongo_conn.close()
    app.vectordb_client.disconnect()


app.router.lifespan.on_startup.append(startup_span)
app.router.lifespan.on_shutdown.append(shutdown_span)

app.include_router(base.base_router)
app.include_router(data.data_router)


