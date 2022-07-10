from random import randint
from fastapi import APIRouter, HTTPException, UploadFile

from .documents import router as doc_router
from ...settings import s3, db

router = APIRouter(prefix="/projects")
router.include_router(doc_router)



@router.post("/", status_code=201)
async def create_project(file: UploadFile):
	if not file:
		raise HTTPException(400, detail="file doesn't exists")
		
	fileKey = str(randint(1000, 9999)) + file.filename

	s3.upload_fileobj(file.file, 'classify', fileKey)

	project_id = db['projects'].insert_one({
		'file_id': fileKey,
	}).inserted_id
	
	return {"status": "success", "payload": {"id": str(project_id)}}


@router.get("/{id}")
async def get_project(id: int):
   raise HTTPException(status_code=404, detail="not found")
