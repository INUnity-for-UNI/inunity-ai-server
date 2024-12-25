from pydantic import BaseModel, Field, validator
from konlpy.tag import Okt
from fasttext import load_model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


class VectorizeService:
    def __init__(self, model_path: str, okt=None):
        # FastText 모델 로드
        self.fasttext_model = load_model(model_path)
        self.okt = okt if okt else Okt()

    def words2vec(self, text: str):
        # 영어 단어 추출 및 소문자 변환
        eng_text = re.findall(r'[a-zA-Z]+', text)
        eng_text = [e.lower() for e in eng_text]

        # 한국어 명사 추출
        words = self.okt.nouns(text)
        words.extend(eng_text)

        # FastText를 활용해 각 단어를 벡터로 변환
        vectors = [self.fasttext_model.get_word_vector(word) for word in words]
        return np.array(vectors)

    def sentence2vec(self, sentence: str):
        # 검색어를 벡터로 변환
        vectors = self.words2vec(sentence)
        if len(vectors) == 0:  # 벡터가 비어있는 경우
            return np.zeros(300)
        return np.mean(vectors, axis=0)  # 300차원 벡터 반환


# 검색어 데이터 처리
class Query(BaseModel):
    text: str = Field(..., description="사용자가 입력한 검색어")

    @validator("text")
    def validate_text(cls, value):
        if not value or not isinstance(value, str):
            raise ValueError("검색어는 비어 있을 수 없으며 문자열이어야 합니다.")
        return value

    def to_vector(self, vectorizer: VectorizeService):
        # 검색어를 벡터로 변환
        return vectorizer.sentence2vec(self.text)


if __name__ == '__main__':
    # 1. VectorizeService 초기화
    model_path = './models/cc.ko.300.bin'  # FastText 모델 경로
    vectorizer = VectorizeService(model_path=model_path)

    # 2. Query 입력
    query = Query(text='수강신청')

    # 3. 검색어를 벡터로 변환
    query_vector = query.to_vector(vectorizer)

    # 4. 결과 출력
    print("검색어:", query.text)
    print("벡터 (300차원):", query_vector)
