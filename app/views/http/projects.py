from fastapi import APIRouter, File, Form, HTTPException

from .documents import router as doc_router

router = APIRouter(prefix="/projects")
router.include_router(doc_router)
d = []


@router.post("/", status_code=201)
async def create_project(file: bytes = File(), model=Form(), desc=Form()):
    if not file:
        raise HTTPException(400, detail="file doesn't exists")

    d.append((file, model, desc))

    return {"status": "success", "payload": {"id": len(d) - 1}}


@router.get("/{id}")
async def get_project(id: int):
    try:
        return d[id][1:]
    except IndexError:
        raise HTTPException(status_code=404, detail="not found")
