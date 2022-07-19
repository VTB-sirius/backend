from concurrent.futures import process
from random import randint
from bson import ObjectId
from fastapi import APIRouter, HTTPException, UploadFile

from app.adapters.ya import yc

from ...settings import db

router = APIRouter(prefix="/documents")

@router.get("/{_id}")
async def get_project(_id: str):
	doc = db['documents'].find_one({ '_id': ObjectId(_id) })
	
	return { 'status': 'success', 'payload': doc }
