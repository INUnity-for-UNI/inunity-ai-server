from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import torch


class CleanBotService:
    def __init__(self):
        pass

    @staticmethod
    def load_text_filter(model_name='smilegate-ai/kor_unsmile', device=0):
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pipe = TextClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
            function_to_apply='sigmoid'
        )
        return pipe

    @staticmethod
    def filter_text(text, pipe, threshold=0.3):
        results = pipe(text)[0]

        # clean 점수 찾기
        clean_score = next(item['score'] for item in results if item['label'] == 'clean')

        # 이진 분류 결과 (clean: True, non-clean: False)
        if clean_score <= threshold:
            filtering = True
        else:
            filtering = False

        # 결과 반환
        return {
            'binary_class': filtering,
            'clean_score': clean_score,
            'filtered_text': text if clean_score >= threshold else "부적절한 내용으로 필터링 되었습니다.",
            'original_text': text,
            'total': results
        }