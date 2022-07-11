from random import randint
from fastapi import APIRouter, HTTPException, UploadFile

import requests

from app.adapters.ya import yc

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

	r = requests.post('https://datasphere.api.cloud.yandex.net/datasphere/v1/nodes/7fbc3783-bd8a-482e-b1cb-6c78dcd327c7:execute', json={
		'folder_id': 'b1g59nv1s9n6s73toofh',
		'node_id': '7fbc3783-bd8a-482e-b1cb-6c78dcd327c7',
		'input': {
			'project_id': project_id,
			'file_id': fileKey,
		}
	}, headers={
		'Authorization': 'Bearer ' + yc.aim_token
	})

	print(r.status_code)
	
	return {"status": "success", "payload": {"id": str(project_id)}}


@router.get("/{id}")
async def get_project(id: int):
   raise HTTPException(status_code=404, detail="not found")
