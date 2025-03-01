from .BaseDataModel import BaseDataModel
from .enums.DataBaseEnum import DataBaseEnum
from .db_schemes.project import Project

class ProjectModel(BaseDataModel):
    def __init__(self,db_client):
        super().__init__(db_client)
        self.collection=self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]

    async def create_project(self,project:Project):
        result=await self.collection.insert_one(project.dict(by_alias=True, exclude_unset=True))
        project._id=result.inserted_id
        return project
    
    async def get_project_or_create_one(self,project_id:str):
        record=await self.collection.find_one({"project_id":project_id})
        if record:
            return Project(**record)
        else:
            project=Project(project_id=project_id)
            project= await self.create_project(project)
            return project
 
    async def get_all_projects(self,page:int=1,page_size:int=10):   
        total_documents=await self.collection.count_documents({})
        total_pages=total_documents//page_size
        if total_documents%page_size>0:
            total_pages+=1

        cursor=self.collection.find().skip((page-1)*page_size).limit(page_size)
        projects=[]
        async for project in cursor:
            projects.append(Project(**project))

        return projects,total_pages