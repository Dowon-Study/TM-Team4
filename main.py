import re
from naver_news_crawler import NaverNewsCrawler

def split_sentences(text):
    # 문장 단위로 나누기 (간단한 정규식 기반)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [{"sentenceNo": i+1, "sentenceContent": s, "sentenceSize": len(s)} for i, s in enumerate(sentences) if s]

def format_json_structure(item, part_num="P1"):
    content = item.get("content", "").strip()
    title = re.sub('<[^<]+?>', '', item.get("title", ""))
    category = item.get("category", "기타")
    sentences = split_sentences(content)

    return {
        "sourceDataInfo": {
            "newsID": item.get("link", "")[-10:],  # 링크 끝부분을 ID로 사용
            "newsCategory": category,
            "newsSubcategory": category,
            "newsTitle": title,
            "newsSubTitle": "null",
            "newsContent": content,
            "partNum": part_num,
            "useType": 0,
            "processType": "D",
            "processPattern": "11",
            "processLevel": "하",
            "sentenceCount": len(sentences),
            "sentenceInfo": sentences
        }
    }

if __name__ == "__main__":
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    
    print("[INFO] Starting main process...")
    crawler = NaverNewsCrawler(client_id, client_secret)
    
    results = crawler.fetch_news(
        query="인공지능",
        max_count=500,
        display=100,
        sort="date",
        start_date="2021-03-01",
        end_date="2025-04-06",
        remove_duplicates=True,
        extract_content=True
    )

    # 원본 데이터 저장
    crawler.save_to_json(results, "news_with_content.json")
    crawler.save_to_csv(results, "news_with_content.csv")

    # JSON 구조 변환 후 저장
    formatted = [format_json_structure(item) for item in results if "content" in item]
    crawler.save_to_json(formatted, "formatted_news.json")

    print("[INFO] Finished main process.")
