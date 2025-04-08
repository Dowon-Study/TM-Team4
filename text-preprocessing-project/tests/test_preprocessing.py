import unittest
from src.preprocessing.korean_preprocessing import clean_korean_text, remove_korean_stopwords, preprocess_sentence_korean
from src.preprocessing.spacing import correct_spacing
from src.preprocessing.soynlp_utils import tokenize_with_soynlp

class TestKoreanPreprocessing(unittest.TestCase):

    def test_clean_korean_text(self):
        text = "안녕하세요! 반갑습니다. 1234"
        cleaned = clean_korean_text(text)
        self.assertEqual(cleaned, "안녕하세요 반갑습니다 1234")

    def test_remove_korean_stopwords(self):
        tokens = ['이', '것', '안녕하세요', '반갑습니다']
        filtered = remove_korean_stopwords(tokens)
        self.assertEqual(filtered, ['안녕하세요', '반갑습니다'])

    def test_preprocess_sentence_korean(self):
        sentence = "이것은 테스트 문장입니다."
        processed = preprocess_sentence_korean(sentence)
        self.assertIn("테스트", processed)
        self.assertIn("문장", processed)

    def test_correct_spacing(self):
        text = "안녕하세요.저는학생입니다."
        corrected = correct_spacing(text)
        self.assertEqual(corrected, "안녕하세요. 저는 학생입니다.")

    def test_tokenize_with_soynlp(self):
        text = "안녕하세요. 자연어 처리를 배우고 있습니다."
        tokens = tokenize_with_soynlp(text)
        self.assertIn("자연어", tokens)
        self.assertIn("처리", tokens)

if __name__ == '__main__':
    unittest.main()