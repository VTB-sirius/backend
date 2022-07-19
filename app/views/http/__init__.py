from fastapi import APIRouter

from .projects import router as pr_router
from .documents import router as doc_router

router = APIRouter()
router.include_router(pr_router)
router.include_router(doc_router)
