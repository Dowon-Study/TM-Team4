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

# 카테고리 별 분류
def classify_label(text):
    text = text.lower()

    economy_keywords = ["경제", "주식", "증시", "환율", "물가", "삼성전자", "기업", "코스피", "코스닥", 
                        "무역", "금리", "금융", "재정", "투자", "부동산", "부채", "원자재", "경기", "실업률",
                        "소비", "수출입", "중소기업"]
    
    politics_keywords = ["정치", "정부", "대통령", "국회", "총리", "장관", "선거", "외교", "정당", "국정",
                         "청와대", "의회", "국회의원", "정책", "법안", "행정부", "시위", "입법", "개헌", "지방선거"]
    
    it_keywords = ["ai", "인공지능", "기술", "it", "테크", "빅데이터", "로봇", "메타버스", "클라우드", "5g",
                   "스마트폰", "반도체", "os", "앱", "알고리즘", "개발자", "프로그래밍", "사물인터넷", "코딩"]
    
    entertainment_keywords = ["연예", "배우", "드라마", "가수", "영화", "연예인", "예능", "방송", "뮤직", "소속사",
                              "공연", "무대", "앨범", "팬", "아이돌", "콘서트", "엔터테인먼트", "화보", "연출", "ost"]
    
    sports_keywords = ["스포츠", "축구", "야구", "농구", "배구", "골프", "선수", "감독", "올림픽", "경기",
                       "리그", "승부", "득점", "이적", "국가대표", "월드컵", "피겨", "마라톤", "심판", "우승"]
    
    society_keywords = ["사회", "사건", "사고", "경찰", "법원", "검찰", "범죄", "화재", "사망", "구속", "수사",
                        "교통", "재난", "실종", "노동", "시민", "주민", "시위", "복지", "인권", "소송"]
    
    education_keywords = ["교육", "학교", "대학", "입시", "수능", "시험", "학생", "교사", "교수", "교육청", "장학금",
                          "학원", "온라인 강의", "출결", "자퇴", "졸업", "수업", "성적", "과외", "교과서"]

    if any(word in text for word in economy_keywords):
        return "경제"
    elif any(word in text for word in politics_keywords):
        return "정치"
    elif any(word in text for word in it_keywords):
        return "IT"
    elif any(word in text for word in entertainment_keywords):
        return "연예"
    elif any(word in text for word in sports_keywords):
        return "스포츠"
    elif any(word in text for word in society_keywords):
        return "사회"
    elif any(word in text for word in education_keywords):
        return "교육"
    else:
        return "기타"

    
# 본문 전체 수집
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

        for sel in selectors:
            content_div = soup.find('div', sel)
            if content_div:
                text = content_div.get_text(strip=True, separator=' ')
                text = re.sub(r'\s+', ' ', text).strip()  # 줄바꿈 제거하고 공백 정리

                # 문장 단위로 분리
                sentences = re.split(r'(?<=[.!?]) +', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

                if len(sentences) >= 1:
                    return sentences

        return "본문 너무 짧음"
    except Exception as e:
        return f"본문 크롤링 실패: {e}"