from fastapi import APIRouter

router = APIRouter(prefix="")


@router.get("/{id_}/documents/{doc_id}/")
async def get_doc_by_id(id_: int, doc_id: int, model: str):
    pass

