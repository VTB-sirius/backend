from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .views.http import router

app = FastAPI()

app.include_router(router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
