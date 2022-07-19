from concurrent.futures import process
from random import randint
from bson import ObjectId
from fastapi import APIRouter, HTTPException, UploadFile

import subprocess
import docker

from app.adapters.ya import yc

from .documents import router as doc_router
from ...settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MONGODB_DATABASE, MONGODB_HOST, MONGODB_PASSWORD, MONGODB_USERNAME, YA_SERVICE_ACC_ID, YA_SERVICE_KEY_ID, s3, db


router = APIRouter(prefix="/projects")
router.include_router(doc_router)

client = docker.from_env()

@router.post("/", status_code=201)
async def create_project(file: UploadFile):
	if not file:
		raise HTTPException(400, detail="file doesn't exists")
		
	fileKey = str(randint(1000, 9999)) + file.filename

	s3.upload_fileobj(file.file, 'classify', fileKey)

	project_id = db['projects'].insert_one({
		'file_id': fileKey,
	}).inserted_id

	envis = {
		'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
		'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,

		'MONGODB_HOST': MONGODB_HOST,
		'MONGODB_DATABASE': MONGODB_DATABASE,
		'MONGODB_USERNAME': MONGODB_USERNAME,
		'MONGODB_PASSWORD': MONGODB_PASSWORD,

		'YA_SERVICE_ACC_ID': YA_SERVICE_ACC_ID,
		'YA_SERVICE_KEY_ID': YA_SERVICE_KEY_ID,
	}

	commandToRun = "python3 /home/jupyter/model_dbs.py " + str(project_id) + " " +  fileKey

	volumes = ['/app/models/:/home/jupyter']
	
	container = client.containers.run("dbs_model", commandToRun, environment=envis, volumes=volumes, remove=True)

	print(container.logs())
		
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
