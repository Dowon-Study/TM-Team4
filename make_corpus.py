import pandas as pd
import os

def extract_text_to_corpus(excel_files, output_file="corpus.txt"):
    # 모든 텍스트를 저장할 리스트
    all_text = []
    
    # 각 엑셀 파일 처리
    for file in excel_files:
        if os.path.exists(file):
            # 엑셀 파일 로드
            df = pd.read_excel(file)
            
            # MQ, SQ, UA, SA 열에서 텍스트 추출
            for column in ['MQ', 'SQ', 'UA', 'SA']:
                if column in df.columns:
                    # NaN 값은 빈 문자열로 처리
                    texts = df[column].fillna('').astype(str)
                    all_text.extend(texts)
    
    # 중복 제거 후 텍스트를 파일에 저장
    unique_text = list(dict.fromkeys(all_text))
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in unique_text:
            f.write(text + '\n')
    
    print(f"텍스트가 {output_file}에 저장되었습니다.")

# 사용 예시
excel_files = [
    "F 개폐(7,859)_new.xlsx",
    "D 소매점(14,949)_new.xlsx",
    "H 관광여가욕(4,949)_new.xlsx",
    "I 부동산(8,131)_new.xlsx",
    "G 숙박업(7,113)_new.xlsx",
    "E 생활서비스(11,087)_new.xlsx",
    "C 학원(4,773)_new.xlsx",
    "B 의류(15,826)_new.xlsx",
    "A 음식점(15,726)_new.xlsx"
]
extract_text_to_corpus(excel_files)