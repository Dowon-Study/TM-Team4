from naver_news_crawler import NaverNewsCrawler

if __name__ == "__main__":
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"
    
    print("[INFO] Starting main process...")
    crawler = NaverNewsCrawler(client_id, client_secret)
    
    # 테스트 요청 예시
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

    # JSON 저장
    crawler.save_to_json(results, "news_with_content.json")
    # CSV 저장
    crawler.save_to_csv(results, "news_with_content.csv")

    print("[INFO] Finished main process.")
