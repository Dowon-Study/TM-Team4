from django.http import JsonResponse
from django.views.decorators.http import require_GET
from datetime import datetime
import requests
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import json
from pathlib import Path
from difflib import SequenceMatcher
import csv
from sentence_transformers import SentenceTransformer, util
import hashlib
from .crawler import (convert_to_desktop_url, extract_article_paragraphs, classify_label,)

load_dotenv()  # 환경 변수(.env) 불러오기
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2") # Sentence Transformer 모델 로드 (한 번만 로드)

# 요약 텍스트 생성 (앞 5문장)
def make_paragraph_key(sentences):
    text = " ".join(sentences[:5])
    return text

# 중복 검사 함수 (제목 + 앞 5문장 기준 + 임베딩 벡터 비교)
def check_duplicate_paragraph(paragraph_sentences, title, collected_paragraphs, threshold=0.85, embed_threshold=0.85):
    summary_part = make_paragraph_key(paragraph_sentences)
    summary_part_with_title = title + "\n" + summary_part
    key_hash = hashlib.md5(summary_part.encode('utf-8')).hexdigest()

    # 1. 해시 중복 검사
    if key_hash in collected_paragraphs:
        return True

    # 2. 문자열 유사도 검사 (SequenceMatcher 사용)
    for prev in collected_paragraphs:
        if summary_part_with_title[:30] == prev[:30]:
            return True
        if SequenceMatcher(None, summary_part_with_title, prev).ratio() >= threshold:
            return True

    # 3. 의미 유사도 검사 (임베딩 벡터 사용)
    key_embed = model.encode(summary_part_with_title, convert_to_tensor=True)
    for prev_text in collected_paragraphs:
        prev_embed = model.encode(prev_text, convert_to_tensor=True)
        embed_sim = util.pytorch_cos_sim(key_embed, prev_embed).item()
        if embed_sim >= embed_threshold:
            return True

    collected_paragraphs.append(summary_part_with_title)
    return False

@require_GET  # GET 메소드를 이용해 API 요청
def get_news_by_date(request):
    date = request.GET.get('date')
    query = request.GET.get('query', '뉴스')
    fuzzy = request.GET.get('fuzzy_date', 'false').lower() == 'true'

    if not date:
        return JsonResponse({'error': '날짜를 입력해주세요.'}, status=400)

    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    input_date = datetime.strptime(date, "%Y-%m-%d")
    result_data = []
    collected_paragraphs = []
    start = 1
    total_news_needed = 1000
    valid_news_count = 0

    # CSV 저장 폴더 및 파일 설정
    csv_output_dir = Path(__file__).resolve().parent / "outputs_csv" / date
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = csv_output_dir / f"{date}_{query}.csv"
    write_header = not csv_output_path.exists()

    while valid_news_count < total_news_needed and start <= 1000:
        params = {
            "query": query,
            "display": 100,
            "start": start,
            "sort": "date"
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return JsonResponse({'error': 'API 요청 실패'}, status=500)

        news_data = response.json().get('items', [])
        if not news_data:
            break

        print(f"API 요청: start={start} / 받아온 기사 수: {len(news_data)}")

        for item in news_data:
            pub_date_str = item['pubDate']
            pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")

            # 날짜 필터 (fuzzy 여부에 따라 다르게 작동)
            if not fuzzy:
                if pub_date.date() != input_date.date():
                    continue
            else:
                if abs((pub_date.date() - input_date.date()).days) > 2:
                    continue

            link = item['link']
            if "n.news.naver.com" in link:
                link = convert_to_desktop_url(link)

            if "news.naver.com" not in link:
                print("외부 뉴스 스킵")
                continue

            title = BeautifulSoup(item['title'], "html.parser").get_text()
            paragraph_sentences = extract_article_paragraphs(link)

            print(f"🔗 {link}")
            print(f"📄 본문 요약: {paragraph_sentences[:5]}")  # 앞 5문장 요약

            if isinstance(paragraph_sentences, str):
                print(f"[크롤링 실패] {link} → 이유: {paragraph_sentences}")
                continue

            if len(paragraph_sentences) < 5:
                print("문장 수 부족 → 제외")
                continue

            # 중복 검사
            if check_duplicate_paragraph(paragraph_sentences, title, collected_paragraphs):
                print("유사 본문 스킵")
                continue

            combined_text = "\n".join(paragraph_sentences)
            collected_paragraphs.append(title + "\n" + "\n".join(paragraph_sentences[:2]))
            label = classify_label(combined_text + " " + title)

            result_data.append({
                "title": title,
                "text": combined_text,
                "label": label
            })

            # JSON 저장 로직
            sentence_info = []
            for idx, sent in enumerate(paragraph_sentences, start=1):
                sentence_info.append({
                    "sentenceNo": idx,
                    "sentenceContent": sent,
                    "sentenceSize": len(sent)
                })

            news_id = f"{date}_{query}_{valid_news_count + 1:03}"
            safe_news_id = news_id.replace("/", "-").replace("\\", "-")

            output_data = {
                "sourceDataInfo": {
                    "newsID": f"LC_M09_{safe_news_id}",
                    "newsCategory": label,
                    "newsSubcategory": "",
                    "newsTitle": title,
                    "newsSubTitle": "null",
                    "newsContent": combined_text,
                    "partNum": "P1",
                    "useType": 0,
                    "processType": "D",
                    "processPattern": "11",
                    "processLevel": "하",
                    "sentenceCount": len(sentence_info),
                    "sentenceInfo": sentence_info
                },
                "labeledDataInfo": {
                    "newTitle": title,
                    "clickbaitClass": 0,
                    "referSentenceInfo": [
                        {
                            "sentenceNo": s["sentenceNo"],
                            "referSentenceyn": "N"
                        } for s in sentence_info
                    ]
                }
            }

            output_dir = Path(__file__).resolve().parent / "outputs_json" / date
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{safe_news_id}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            # CSV 저장 로직 (문장 단위)
            with open(csv_output_path, "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["newsID", "title", "label", "sentenceNo", "sentenceContent", "sentenceSize"])
                    write_header = False

                for sentence in sentence_info:
                    writer.writerow([
                        f"LC_M09_{safe_news_id}",
                        title,
                        label,
                        sentence["sentenceNo"],
                        sentence["sentenceContent"],
                        sentence["sentenceSize"]
                    ])

            valid_news_count += 1

            if valid_news_count >= total_news_needed:
                break

        start += 100

    return JsonResponse({"data": result_data}) 