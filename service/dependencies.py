from .cleanBot.main import CleanBotService
from .summaryText.main import SummarizerService

def get_clean_bot_service() -> CleanBotService:
    return CleanBotService()

def get_summarizer_service() -> SummarizerService:
    return SummarizerService()