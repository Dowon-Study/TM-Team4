"""
Terms-vs-Daily TF-IDF Analyzer  (noun-only · daily_freq < 10 filter)

변경 요약
─────────
• 불용어 목록: 기존 짧은 버전(회사, 서비스 … 회원원)만 사용
• 후보 필터:  daily_freq ≥ 10  →  **제외**
• sparse-word( daily_freq == 0 ) 별도 표시는 그대로 유지
"""

import re
import logging
from collections import Counter
from typing import List, Optional

import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer


# ─────────────────── LOGGING ───────────────────
logging.basicConfig(
    filename="terms_extractor_debug.log",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class TermsDifficultWordsExtractor:
    def __init__(
        self,
        *,
        min_word_length: int = 2,
        allow_single_char_noun: bool = True,
        debug: bool = True,
    ):
        self.okt = Okt()
        self.min_word_length = min_word_length
        self.allow_single_char_noun = allow_single_char_noun
        self.debug = debug

        # ── 기본+약식 TOS 불용어 ──
        self.stopwords = {
            # 조사·기능어
            "은", "는", "이", "가", "을", "를", "에", "의", "와", "과",
            "도", "로", "으로", "에서", "까지", "부터", "만", "라도",
            "조차", "마저", "에게", "한테", "및", "또는", "그리고",
            "하지만", "그러나", "따라서",
            # TOS에서 잦은 일반 명사
            "회사", "서비스", "회원", "이용", "약관", "고객", "사이트",
            "웹사이트", "본", "당사", "사용", "정보", "관련", "제공",
            "위", "이하", "경우", "때", "내용", "목적", "조항", "제공자",
            "회원원",
            # 정중 어미(명사 태깅)
            "합니다", "합니다.",
        }

    # ── 내부 디버그 ──
    def _dbg(self, msg: str) -> None:
        if self.debug:
            log.debug(msg)

    # ── 전처리: 명사만 남기기 ──
    def preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s가-힣]", " ", text)
        morphs = self.okt.pos(text, stem=True)

        tokens: List[str] = []
        for word, pos in morphs:
            if pos != "Noun":
                continue
            length_ok = len(word) >= self.min_word_length
            if self.allow_single_char_noun and len(word) == 1:
                length_ok = True
            if not length_ok or word in self.stopwords or word.isdigit():
                continue
            tokens.append(word)

        self._dbg(
            f"Preprocessed {len(text)} chars → {len(tokens)} tokens "
            f"(sample: {tokens[:15]})"
        )
        return tokens

    # ── 파일 → 문서 리스트 ──
    def load_corpus_from_file(self, path: str, lines_per_doc: int = 5) -> List[str]:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        docs = [
            " ".join(lines[i : i + lines_per_doc])
            for i in range(0, len(lines), lines_per_doc)
        ]
        self._dbg(f"Loaded {len(lines)} lines → {len(docs)} docs")
        return docs

    # ── 핵심: TF-IDF 추출 ──
    def extract_difficult_words(
        self,
        terms_text: str,
        daily_corpus_path: str,
        *,
        top_n: int = 20,
        min_freq: int = 2,
        lines_per_doc: int = 5,
    ):
        daily_raw = self.load_corpus_from_file(daily_corpus_path, lines_per_doc)
        terms_tok = self.preprocess_text(terms_text)
        daily_tok_list = [self.preprocess_text(doc) for doc in daily_raw]

        docs = [" ".join(terms_tok)] + [" ".join(t) for t in daily_tok_list]

        vec = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w가-힣]+\b", max_features=5000, min_df=1
        )
        tfidf_mat = vec.fit_transform(docs)
        feats = vec.get_feature_names_out()
        terms_tfidf = tfidf_mat[0].toarray().flatten()
        idf = vec.idf_

        terms_cnt = Counter(terms_tok)
        daily_cnt = Counter(tok for lst in daily_tok_list for tok in lst)

        words = []
        for i, w in enumerate(feats):
            tfidf_val = terms_tfidf[i]
            t_freq = terms_cnt.get(w, 0)
            d_freq = daily_cnt.get(w, 0)

            # ★ 필터: 일상 빈도 10 이상이면 제외 ★
            if (
                t_freq >= min_freq
                and tfidf_val > 0
                and d_freq < 10            # ← 핵심 조건
            ):
                words.append((w, tfidf_val, t_freq, idf[i], t_freq, d_freq))

        self._dbg(f"Candidates kept after daily_freq<10: {len(words)}")
        words.sort(key=lambda x: x[1], reverse=True)
        return words[:top_n]

    # ── 실행 & 출력 ──
    def analyze_and_display(
        self,
        terms_file: str,
        corpus_file: str,
        *,
        top_n: int = 20,
        min_freq: int = 2,
        lines_per_doc: int = 5,
    ) -> Optional[pd.DataFrame]:
        terms_text = open(terms_file, encoding="utf-8").read()

        res = self.extract_difficult_words(
            terms_text,
            corpus_file,
            top_n=top_n,
            min_freq=min_freq,
            lines_per_doc=lines_per_doc,
        )
        if not res:
            print("No difficult words found (after daily_freq < 10 filter).")
            return None

        df = pd.DataFrame(
            res, columns=["word", "tfidf", "tf", "idf", "terms_freq", "daily_freq"]
        )
        df["difficulty"] = df["terms_freq"] / (df["daily_freq"] + 1)

        # 메인 테이블
        print("=" * 90)
        print(f"Top {len(df)} Difficult Words  (daily_freq < 10)")
        print("=" * 90)
        print(
            f"{'Rank':<4} {'Word':<20} {'TF-IDF':<9} {'TF':<5} "
            f"{'IDF':<7} {'TermsFreq':<10} {'DailyFreq':<9} {'Difficulty':<10}"
        )
        print("-" * 90)
        for i, r in df.iterrows():
            print(
                f"{i+1:<4} {r.word:<20} {r.tfidf:<9.4f} {r.tf:<5} "
                f"{r.idf:<7.4f} {r.terms_freq:<10} {r.daily_freq:<9} "
                f"{r.difficulty:<10.2f}"
            )

        # sparse spotlight(daily_freq == 0)
        sparse = df[df["daily_freq"] == 0]
        if not sparse.empty:
            print("\n► Sparse words (absent from daily corpus):")
            for w in sparse["word"]:
                print("  •", w)

        return df


# ── Driver ──
def main() -> None:
    ext = TermsDifficultWordsExtractor(
        min_word_length=4, allow_single_char_noun=False, debug=True
    )
    ext.analyze_and_display(
        terms_file="term.txt",
        corpus_file="corpus.txt",
        top_n=15,
        min_freq=1,
        lines_per_doc=5,
    )

    print("\n" + "=" * 90)
    print("Analysis completed.  See 'terms_extractor_debug.log' for details.")


if __name__ == "__main__":
    main()
