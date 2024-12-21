from fastapi import APIRouter, Depends
from service.cleanBot.main import CleanBotService
from service.dependencies import get_clean_bot_service

router = APIRouter()

@router.get("/test")
async def filter_text(text: str, service: CleanBotService = Depends(get_clean_bot_service)):
    pipe = service.load_text_filter()
    result = service.filter_text(text, pipe)
    return result