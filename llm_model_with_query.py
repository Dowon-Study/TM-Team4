"""
03_query_pipeline.py   (2025-06-07 최종판)
────────────────────────────────────────────────────────────────────────────
· 약관 전문 + (조항, 라벨) 리스트 입력
  0) 약관 전문 요약
  1) 각 조항 → 도메인 분류
  2) Self-Query 2회 호출
     ─ ① label 그대로   : ‘왜 유리/불리한가’ 설명용 문서
     ─ ② label="유리"   : 같은 도메인 유리 사례(k개)   ※라벨이 불리인 경우만
  3) GPT-4o 한 번 호출
     ─ 설명 + (불리일 때) 개정 전·후 + 관련 법령 인용
· 임베딩  : intfloat/multilingual-e5-base  (무료·한국어 지원)
· 벡터 DB : rag_index_build/faiss_index/*
· LLM     : gpt-4o   (OPENAI_API_KEY 필요)
"""

import os, json, pickle, logging
from pathlib import Path
from typing import List, Dict

import torch, faiss
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from dotenv import load_dotenv      # ← 추가
load_dotenv()                      # ← .env 파일(프로젝트 루트)에 있는 키들을 환경변수로 등록

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❗ OPENAI_API_KEY 환경 변수가 비어 있습니다.")


# ────────────────────────────
# 0. 무료 E5 임베딩 래퍼
# ────────────────────────────
class E5Embeddings(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]):
        return self.model.encode(
            [f"passage: {t}" for t in texts], normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str):
        return self.model.encode(
            f"query: {text}", normalize_embeddings=True
        ).tolist()

# ────────────────────────────
# 1. 벡터 DB 로드
# ────────────────────────────
IDX_DIR = Path("rag_index_build/faiss_index")
faiss_index = faiss.read_index(str(IDX_DIR / "index.faiss"))
ids_list    = pickle.loads((IDX_DIR / "ids_list.pkl").read_bytes())
id2meta     = pickle.loads((IDX_DIR / "index_meta.pkl").read_bytes())

docs = [
    Document(
        page_content=id2meta[_id]["clause_text"],
        metadata={k: v for k, v in id2meta[_id].items()
                  if k not in ("clause_text", "std_text")},
    )
    for _id in ids_list
]
vectorstore = FAISS(
    embedding_function=E5Embeddings(),
    index=faiss_index,
    docstore=InMemoryDocstore({d.metadata["id"]: d for d in docs}),
    index_to_docstore_id={i: d.metadata["id"] for i, d in enumerate(docs)},
)

# ────────────────────────────
# 2. LLM & Self-Query 설정
# ────────────────────────────
DOMAIN_CATS = [
    "A. 금융기관","B. 전자지급·핀테크","C. 보험","D. 증권·투자",
    "E. 유통·사이버몰","F. 프랜차이즈·공급·분양·신탁","G. 부동산·임대차·리스",
    "H. 운송·물류","I. 여행·레저·게임","J. 생활서비스","K. 기타 계약·보증",
]
META_INFO = [
    AttributeInfo("advantageous","유리/불리","string",enum=["유리","불리"]),
    AttributeInfo("domain","11개 도메인","string",enum=DOMAIN_CATS),
    AttributeInfo("has_related_laws","법령 기재 여부","boolean"),
]
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=api_key)

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, "약관 조항 전문",
    metadata_field_info=META_INFO, search_kwargs={"k":4},
)

# ────────────────────────────
# 3. 헬퍼
# ────────────────────────────
def brief_summarize(text:str)->str:
    return llm.predict(
        "다음 약관 전문을 소비자 관점에서 이해하기 쉽도록 요약하세요.\n\n"+text
    ).strip()

def classify_domain(text:str)->str:
    rsp = llm.predict(
        "아래 조항의 산업·서비스 분야를 하나 선택:\n"
        + "\n".join(DOMAIN_CATS) + "\n\n조항:\n" + text
    ).strip()
    return rsp if rsp in DOMAIN_CATS else "K. 기타 계약·보증"

def self_query_docs(clause:str, label:str, domain:str, k:int=4)->List[Document]:
    """
    label, domain 조건으로 Self-Query 검색 후 k개 반환
    """
    return retriever.get_relevant_documents(
        clause,
        structured_query_filter={
            "advantageous": label,
            "domain": domain
        },
        k=k
    )

# ────────────────────────────
# 4. 조항 단위 처리
# ────────────────────────────
def process_clause(clause:str, label:str, k_adv:int=3)->Dict:
    domain = classify_domain(clause)

    # ① 설명용 문서 (라벨 그대로)
    explain_docs = self_query_docs(clause, label, domain, k=4)

    # ② 유리 사례 (불리일 때만)
    rev_docs = []
    if label == "불리":
        rev_docs = self_query_docs(clause, "유리", domain, k=k_adv)

    # ③ GPT-4o 단일 호출
    prompt = f"""
[원문 조항]
{clause}

[설명용 유사 조항 ({label})]
{chr(10).join(d.page_content for d in explain_docs)}

{('[개정 참고 – 같은 도메인에서 유사한 유리 조항]\n' + chr(10).join(d.page_content for d in rev_docs)) if rev_docs else ''}

작업:
1) 위 정보를 근거로 왜 소비자에게 {label}한지 2~3문장으로 설명하세요.
2) {"불리 조항을 소비자에게 유리하게 개정하세요. 개정 할 시에 본 약관에서의 조항의 역할(내용)이 심하게 바뀌지 않게 주의하세요."
     "형식: 개정 전: / 개정 후:" if label=="불리" else
     "유리 조항은 그대로 두고, 개선할 부분이 있으면 간단히 제안하세요."}
3) source 문서에 관련 법령이 있으면 인용·설명에 포함하세요.
"""
    llm_result = llm.predict(prompt=prompt).strip()

    # 법령 모음
    laws = {lw for d in explain_docs+rev_docs
                for lw in d.metadata.get("related_laws","").split("; ") if lw}

    return {
        "domain": domain,
        "llm_result": llm_result,
        "law_refs": sorted(laws),
        "sources_explain": explain_docs,
        "sources_revision": rev_docs,
    }

# ────────────────────────────
# 5. 전체 파이프라인
# ────────────────────────────
def run_terms_analysis(terms:str, clauses:List[str], labels:List[str])->Dict:
    if len(clauses)!=len(labels):
        raise ValueError("clauses와 labels 길이가 다릅니다.")

    terms_summary = brief_summarize(terms)
    clause_res = [process_clause(c,l) for c,l in zip(clauses,labels)]

    return {"terms_summary":terms_summary,"clause_results":clause_res}

# ────────────────────────────
# 6. 데모 실행
# ────────────────────────────
if __name__ == "__main__":
    terms_text = Path("terms.txt").read_text(encoding="utf-8")

    clauses = [
        "제22조 (환불) 회사는 해지 신청 후 7일 내 환불한다.",
        "제5조 (청약철회) 고객은 상품 수령 후 7일 이내 청약철회 가능하다."
    ]
    labels  = ["불리","유리"]

    rep = run_terms_analysis(terms_text, clauses, labels)

    print("\n≡ 약관 전문 요약 ≡\n", rep["terms_summary"])
    for i,(r,lab) in enumerate(zip(rep["clause_results"], labels),1):
        print(f"\n── 조항 {i} ({lab}) ──")
        print("· 도메인:", r["domain"])
        print("· 결과:\n", r["llm_result"])
        if r["law_refs"]:
            print("· 관련 법령:\n", "\n".join(r["law_refs"]))
