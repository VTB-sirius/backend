from fastapi import APIRouter

from .projects import router as pr_router

router = APIRouter()
router.include_router(pr_router)
