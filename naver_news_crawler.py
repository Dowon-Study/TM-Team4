import requests
import json
import csv
import time
import os
from datetime import datetime
from urllib.parse import quote
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수를 로드

class NaverNewsCrawler:
    def __init__(self, client_id, client_secret):
        self.client_id = os.environ.get("CLIENT_ID", "")
        self.client_secret = os.environ.get("CLIENT_SECRET", "")
        
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }

    def fetch_news(self, 
                   query, 
                   max_count=1000, 
                   display=100, 
                   sort='date', 
                   start_date=None, 
                   end_date=None, 
                   remove_duplicates=True,
                   extract_content=False):
        print("[INFO] Starting to fetch news...")
        print(f"[INFO] Query: {query}, Max Count: {max_count}, Display: {display}, Sort: {sort}")

        results = []
        encoded_query = quote(query)
        start = 1
        seen_links = set()

        while start <= max_count:
            url = f"{self.base_url}?query={encoded_query}&display={display}&start={start}&sort={sort}"
            print(f"[INFO] Requesting URL: {url}")
            response = requests.get(url, headers=self.headers)

            if response.status_code != 200:
                print(f"[ERROR] HTTP Status: {response.status_code}, Response: {response.text}")
                break

            data = response.json()
            items = data.get("items", [])
            print(f"[INFO] Fetched {len(items)} items from API.")

            if not items:
                print("[INFO] No more items returned from API.")
                break

            for item in items:
                pub_date_str = item.get("pubDate")
                pub_date_obj = self._parse_pub_date(pub_date_str) if pub_date_str else None

                if not self._check_date_range(pub_date_obj, start_date, end_date):
                    print("[DEBUG] Skipping item because it's outside date range.")
                    continue

                link = item.get("link", "")
                if remove_duplicates and link in seen_links:
                    print("[DEBUG] Skipping item because it's a duplicate link.")
                    continue
                seen_links.add(link)

                if extract_content:
                    if self._is_naver_news_link(link):
                        print(f"[DEBUG] Extracting content from: {link}")
                        content = self._extract_main_content(link)
                        if content.strip() == "":
                            print("[DEBUG] Content extraction failed, skipping item.")
                            continue  # 본문이 없으면 저장 안함
                        item["content"] = content
                    else:
                        print("[DEBUG] Not a Naver link, skipping item.")
                        continue  # 네이버 뉴스 링크가 아니면 저장 안함

                results.append(item)
                print(f"[DEBUG] Appended item to results (total {len(results)})")

            start += display
            if len(items) < display:
                print("[INFO] Items returned less than display count, ending loop.")
                break

            time.sleep(0.5)

        print("[INFO] Finished fetching news.")
        print(f"[INFO] Total fetched results: {len(results)}")
        return results

    def save_to_json(self, data, filename):
        if not data:
            print("[INFO] No data to save as JSON.")
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved data to JSON: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON: {e}")

    def save_to_csv(self, data, filename):
        if not data:
            print("[INFO] No data to save as CSV.")
            return
        try:
            keys = data[0].keys()
            with open(filename, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
            print(f"[INFO] Saved data to CSV: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")

    def _parse_pub_date(self, pub_date_str):
        try:
            return datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
        except ValueError:
            return None

    def _check_date_range(self, date_obj, start_date, end_date):
        if date_obj is None:
            return True
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if date_obj.date() < start_dt.date():
                return False
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if date_obj.date() > end_dt.date():
                return False
        return True

    def _extract_main_content(self, url):
        # 네이버 뉴스 기사 본문에서 텍스트를 추출하는 메소드
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                print(f"[ERROR] Failed to fetch content: {url}, Status Code: {resp.status_code}")
                return ""

            soup = BeautifulSoup(resp.text, "html.parser")

            # 네이버 뉴스 기사 본문은 <article id="dic_area"> 내부에 있음
            article_tag = soup.find("article", {"id": "dic_area"})
            if not article_tag:
                print("[DEBUG] <article id='dic_area'> not found.")
                return ""

            # 모든 하위 태그에서 텍스트만 추출
            text = article_tag.get_text(separator="\n", strip=True)
            if not text:
                print("[DEBUG] Article content is empty.")
                return ""

            return text
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] RequestException during content fetch: {e}")
            return ""


    def _is_naver_news_link(self, url):
        return ("news.naver.com" in url) or ("n.news.naver.com" in url)
