from fastapi import FastAPI,APIRouter,Depends
from helpers.config import get_settings,Settings
import os
base_router=APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)

@base_router.get("/")
async def welcome(app_settings:Settings =Depends(get_settings)):
    app_setting=get_settings()

    app_name=app_setting.APP_NAME
    app_version=app_setting.APP_VERSION
    return {
        "app_name":app_name,
        "app_version":app_version
        
    }
