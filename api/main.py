from fastapi import FastAPI

from api.routers import mock

app = FastAPI()

app.include_router(mock.router)
