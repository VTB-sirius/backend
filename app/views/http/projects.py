from random import randint
from bson import ObjectId
from fastapi import APIRouter, HTTPException, UploadFile

import subprocess

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
	
	subprocess.run("docker", "run", "-it --rm --name dbs_model", "--env-file /.env.docker", "-v /home/simbauser/sirius/models/:/home/jupyter", "deploymodel:latest", "python3", "/home/jupyter/model_dbs.py", project_id, fileKey)
		
	return {"status": "success", "payload": {"id": str(project_id)}}


@router.get("/{_id}")
async def get_project(_id: str, model: str):
	project = db['projects'].find_one({ '_id': ObjectId(_id) })

	if model == 'lda' and ('lda_payload' in project):
		return { "status": "success", "payload": project['lda_payload'] }
	elif model == 'lda':
		return { "status": "pending", "payload": {} }
	elif model == 'dbscan' and ('dbs_payload' in project):
		return { "status": "success", "payload": project['dbs_payload'] }
	elif model == 'dbscan':
		return { "status": "pending", "payload": {} }
	else:
		raise HTTPException(status_code=404, detail="not found")
