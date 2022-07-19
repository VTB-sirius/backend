from concurrent.futures import process
from random import randint
from bson import ObjectId
from fastapi import APIRouter, HTTPException, UploadFile

import subprocess
import docker

from app.adapters.ya import yc

from ...settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ENDPOINT, KEY, MONGODB_DATABASE, MONGODB_HOST, MONGODB_PASSWORD, MONGODB_USERNAME, S3_BUCKET, SECRET, SERVICE_NAME, YA_SERVICE_ACC_ID, YA_SERVICE_KEY_ID, s3, db


router = APIRouter(prefix="/projects")

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
		'S3_BUCKET': S3_BUCKET,
		'SERVICE_NAME': SERVICE_NAME,
		'KEY': KEY,
		'SECRET': SECRET,
		'ENDPOINT': ENDPOINT,

		'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
		'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,

		'MONGODB_HOST': MONGODB_HOST,
		'MONGODB_DATABASE': MONGODB_DATABASE,
		'MONGODB_USERNAME': MONGODB_USERNAME,
		'MONGODB_PASSWORD': MONGODB_PASSWORD,

		'YA_SERVICE_ACC_ID': YA_SERVICE_ACC_ID,
		'YA_SERVICE_KEY_ID': YA_SERVICE_KEY_ID,
	}

	commandToRunBert = "python3 /app/model_bert.py " + str(project_id) + " " +  fileKey
	commandToRunLda = "python3 /app/model_lda.py 15 " + str(project_id) + " " +  fileKey
	commandToRunDbs = "python3 /app/model_dbs.py " + str(project_id) + " " +  fileKey
	
	client.containers.run("vtb_models", commandToRunBert, environment=envis, detach=True, remove=True) #bert
	client.containers.run("vtb_models", commandToRunLda, environment=envis, detach=True, remove=True) #lda
	client.containers.run("vtb_models", commandToRunDbs, environment=envis, detach=True, remove=True) #dbscan
		
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
	elif model == 'bert' and ('bert_payload' in project):
		return { "status": "pending", "payload": project['bert_payload'] }
	elif model == 'bert':
		return { "status": "pending", "payload": {} }
	else:
		raise HTTPException(status_code=404, detail="not found")
