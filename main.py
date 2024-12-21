from fastapi import FastAPI
from routers.clean_bot import router as clean_bot_router
app = FastAPI()

app.include_router(clean_bot_router)