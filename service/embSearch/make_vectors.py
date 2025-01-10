import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils.emb_search import VectorizeService
from utils.load_notice import load_notice


# 메인 코드
if __name__ == '__main__':
    # load notice
    notices = load_notice()

    # VectorizeService 초기화
    model_path = './models/cc.ko.300.bin'
    vectorizer = VectorizeService(model_path=model_path)

    # 공지사항 제목 벡터화
    print('\nStart Vectorization')
    notice_vectors = [vectorizer.sentence2vec(notice) for notice in tqdm(notices)]
    np.save('./DB/notice_vectors.npy', np.array(notice_vectors, dtype='object'))

    # 종료
    print('Vectorization is complete\n')
