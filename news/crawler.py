import requests
import re
import time
import random
import chardet
from bs4 import BeautifulSoup

# 모바일 url을 pc용 url로 변환
def convert_to_desktop_url(mobile_url):
    match = re.search(r'article/(\d+)/(\d+)', mobile_url)
    if match:
        oid, aid = match.groups()
        return f"https://news.naver.com/main/read.naver?oid={oid}&aid={aid}"
    return mobile_url

# 본문 전체 수집 및 카테고리 추출
def extract_article_paragraphs(url):
    try:
        time.sleep(random.uniform(0.8, 1.2))
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return "본문 요청 실패"

        detected = chardet.detect(response.content)
        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        html = response.content.decode(encoding, errors='replace')
        soup = BeautifulSoup(html, 'html.parser')

        selectors = [
            {'id': 'dic_area'},
            {'id': 'newsct_article'},
            {'class': 'article_body'},
            {'class': 'content'},
        ]

        # 카테고리 추출
        category = "기타"
        category_section = soup.find("div", class_="media_end_categorize")
        if category_section:
            em_tag = category_section.find("em", class_="media_end_categorize_item")
            if em_tag:
                category = em_tag.get_text(strip=True)

        for sel in selectors:
            content_div = soup.find('div', sel)
            if content_div:
                text = content_div.get_text(strip=True, separator=' ')
                text = re.sub(r'\s+', ' ', text).strip()  # 줄바꿈 제거하고 공백 정리

                # 문장 단위로 분리
                sentences = re.split(r'(?<=[.!?]) +', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

                if len(sentences) >= 1:
                    return sentences, category

        return "본문 너무 짧음", category

    except Exception as e:
        return f"본문 크롤링 실패: {e}", category