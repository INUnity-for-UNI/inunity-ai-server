from fastapi import APIRouter, Depends

from service.cleanBot.main import CleanBotService
from service.dependencies import get_clean_bot_service
from service.dependencies import get_summarizer_service
from service.summaryText.main import SummarizerService
from service.dependencies import get_emb_search_service
from service.embSearch.pydantic_embsearch import VectorizeService, Query

router = APIRouter()

@router.post("/cleanBot")
async def filter_text(text: str, service: CleanBotService = Depends(get_clean_bot_service)):
    pipe = service.load_text_filter()
    result = service.filter_text(text, pipe)
    return result


@router.post("/summaryBot")
async def summary_notice(text: str, service: SummarizerService = Depends(get_summarizer_service)):
    requested_text = service.create_summary_request(text)
    result = service.summarize_notice(requested_text)
    return result

@router.post("/embSearch")
async def emb_search(keyword: str, service: VectorizeService = Depends(get_emb_search_service)):

    # 사용자 검색어 처리
    query = Query(text=keyword)
    query_vectors = query.to_vector(service)
    return query_vectors.tolist()