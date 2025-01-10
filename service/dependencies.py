from .cleanBot.main import CleanBotService
from .summaryText.main import SummarizerService
from .embSearch.pydantic_embsearch import VectorizeService

def get_clean_bot_service() -> CleanBotService:
    return CleanBotService()

def get_summarizer_service() -> SummarizerService:
    return SummarizerService()

def get_emb_search_service() -> VectorizeService:
    return VectorizeService()