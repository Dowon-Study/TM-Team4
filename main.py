import pandas as pd
import re
from konlpy.tag import Okt

# 1. 불용어 로딩
stopwords_df = pd.read_csv('korean_stopwords.csv', header=None)
stopwords_set = set(stopwords_df[0].tolist())

# 2. 예시 텍스트
text = """사회적 의식을 갖춘 패션브랜드 아이린피셔(Eileen Fishe)와 캐나다 환경NGO 카노피(Canopy)는 삼림생태계 보존을 강화하고, 삼림을 위태롭게하는 패션산업 현실에 경종을 울리기 위해 합동 캠페인을 선언했다.
작년에 약 7000만그루의 나무가 섬유제작을 위해 절단됐고, 그 수치는 20년뒤 2배로 증가할 것으로 예상되고 있다."""

# 3. 특수문자 및 숫자 제거
text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', text)
text = re.sub(r'\d+', '', text)

# 4. 문장 분리
sentences = text.split('\n')

# 5. 형태소 분석기 (Okt)
okt = Okt()

# 6. 문장별 형태소 분석 + 불용어 제거
processed_sentences = []

for sentence in sentences:
    tokens = okt.morphs(sentence)
    filtered = [token for token in tokens if token not in stopwords_set and len(token) > 1]
    if filtered:
        processed_sentences.append(filtered)

# 7. 출력
for i, tokens in enumerate(processed_sentences[:5]):
    print(f"[{i+1}] {tokens}")