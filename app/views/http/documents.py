from concurrent.futures import process
from random import randint
from fastapi import APIRouter, HTTPException, UploadFile

from app.adapters.ya import yc

from ...settings import db

router = APIRouter(prefix="/documents")

@router.get("/{_id}")
async def get_document(_id: str):
	doc = db['documents'].find_one({ '_id': _id })
	
	return { 'status': 'success', 'payload': doc }
