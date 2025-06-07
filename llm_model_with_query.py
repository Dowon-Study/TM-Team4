"""
03_query_pipeline.py  (Merged: summary + domain classification)
────────────────────────────────────────────────────────────────────────────
변경 사항
1) StructuredOutputParser 로 terms_summary + domains[] 한 번에 반환
2) classify_domain 함수 삭제, process_clause 가 domain 파라미터 직접 받음
3) LLM 호출 수 ↓ (약관 전문 요약+분류 1회, 조항별 설명/개정 1회씩)
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
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from dotenv import load_dotenv

# ────────────────────────────────────
# 1. 환경 변수 & LLM
# ────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❗ OPENAI_API_KEY 환경 변수가 비어 있습니다.")

llm = ChatOpenAI(model_name="gpt-4o",
                 temperature=0.2,
                 openai_api_key=api_key)

# ────────────────────────────────────
# 2. 무료 E5 임베딩 래퍼
# ────────────────────────────────────
class E5Embeddings(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return self.model.encode([f"passage: {t}" for t in texts],
                                 normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(f"query: {text}",
                                 normalize_embeddings=True).tolist()

# ────────────────────────────────────
# 3. 벡터 DB 로드
# ────────────────────────────────────
IDX_DIR = Path("rag_index_build/faiss_index")
faiss_index = faiss.read_index(str(IDX_DIR / "index.faiss"))
ids_list = pickle.loads((IDX_DIR / "ids_list.pkl").read_bytes())
id2meta  = pickle.loads((IDX_DIR / "index_meta.pkl").read_bytes())

docs = [Document(page_content=id2meta[i]["clause_text"],
                 metadata={k:v for k,v in id2meta[i].items()
                           if k not in ("clause_text","std_text")})
        for i in ids_list]

vectorstore = FAISS(
    embedding_function=E5Embeddings(),
    index=faiss_index,
    docstore=InMemoryDocstore({d.metadata["id"]: d for d in docs}),
    index_to_docstore_id={idx: d.metadata["id"]
                          for idx, d in enumerate(docs)}
)

# ────────────────────────────────────
# 4. Self-QueryRetriever (기존)
# ────────────────────────────────────
DOMAIN_CATS = [
    "A. 금융기관","B. 전자지급·핀테크","C. 보험","D. 증권·투자",
    "E. 유통·사이버몰","F. 프랜차이즈·공급·분양·신탁","G. 부동산·임대차·리스",
    "H. 운송·물류","I. 여행·레저·게임","J. 생활서비스","K. 기타 계약·보증",
]
META_INFO = [
    AttributeInfo("advantageous","유리/불리","string",enum=["유리","불리"]),
    AttributeInfo("domain","도메인","string",enum=DOMAIN_CATS),
    AttributeInfo("has_related_laws","법령 기재 여부","boolean"),
]

retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, "약관 조항 전문",
    metadata_field_info=META_INFO, search_kwargs={"k":4},
)

# ────────────────────────────────────
# 5. 약관 요약 + 도메인 분류 (1회 호출)
# ────────────────────────────────────
summary_schema = ResponseSchema(
    name="terms_summary",
    description="약관 전문 이해하기 쉽도록록 요약"
)
domain_schema = ResponseSchema(
    name="domains",
    description="조항 목록과 동일한 순서로 도메인 문자열 배열"
)
parser = StructuredOutputParser.from_response_schemas(
    [summary_schema, domain_schema])
format_instr = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    """{format_instr}

약관 전문:
{terms_text}

조항 목록:
{clauses}

작업:
1) 약관 전문을 소비자 관점에서 이해하기 쉽도록록 요약
2) 각 조항의 산업·서비스 분야(아래 11개 중 하나)를 선택하여여 순서대로 나열

카테고리:
{domain_list}
""")

summary_chain = LLMChain(llm=llm,
                         prompt=prompt,
                         output_parser=parser)

# ────────────────────────────────────
# 6. Self-Query 도움 함수
# ────────────────────────────────────
def self_query_docs(clause, label, domain, k=4):
    return retriever.get_relevant_documents(
        clause,
        structured_query_filter={"advantageous": label,
                                 "domain": domain},
        k=k
    )

# ────────────────────────────────────
# 7. 조항 처리
# ────────────────────────────────────
def process_clause(clause, label, domain, k_adv=3):
    explain_docs = self_query_docs(clause, label, domain)
    rev_docs = self_query_docs(
        clause, "유리", domain, k_adv) if label == "불리" else []

    prompt = f"""
[원문 조항]
{clause}

[설명용 유사 조항 ({label})]
{chr(10).join(d.page_content for d in explain_docs)}

{('[개정 참고 – 같은 도메인 유리 조항]\n' + chr(10).join(d.page_content for d in rev_docs))
 if rev_docs else ''}

작업:
1) 위 정보를 근거로 왜 소비자에게 {label}한지 2~3문장 설명하세요.
2) {"불리 조항을 소비자에게 유리하게 개정하세요. "
     "형식: 개정 전: / 개정 후:" if label == "불리"
     else "유리 조항은 그대로 두고, 개선할 부분이 있으면 간단히 제안하세요."}
3) source 문서에 관련 법령이 있으면 인용·설명에 포함하세요.
"""
    llm_result = llm.predict(prompt=prompt).strip()

    laws = {lw for d in explain_docs+rev_docs
                for lw in d.metadata.get("related_laws","").split("; ") if lw}

    return {"domain": domain,
            "llm_result": llm_result,
            "law_refs": sorted(laws)}

# ────────────────────────────────────
# 8. 메인 파이프라인
# ────────────────────────────────────
def run_terms_analysis(terms_text, clauses, labels):
    if len(clauses) != len(labels):
        raise ValueError("clauses와 labels 길이가 다릅니다.")

    # ➊ 요약 + 도메인 분류 한 번에
    parsed = summary_chain.predict_and_parse(
        terms_text=terms_text,
        clauses="\n".join(f"{i+1}) {c}"
                          for i, c in enumerate(clauses)),
        format_instr=format_instr,
        domain_list="\n".join(DOMAIN_CATS)
    )

    terms_summary = parsed["terms_summary"]
    domains = parsed["domains"]

    # fallback 길이 체크
    if len(domains) != len(clauses):
        domains = ["K. 기타 계약·보증"] * len(clauses)

    clause_results = [process_clause(c, l, d)
                      for c, l, d in zip(clauses, labels, domains)]

    return {"terms_summary": terms_summary,
            "clause_results": clause_results}

# ────────────────────────────────────
# 9. 데모
# ────────────────────────────────────
if __name__ == "__main__":
    terms = Path("terms.txt").read_text(encoding="utf-8")
    clauses_demo = [
        "제22조 (환불) 회사는 해지 신청 후 7일 내 환불한다.",
        "제5조 (청약철회) 고객은 상품 수령 후 7일 이내 청약철회 가능하다."
    ]
    labels_demo = ["불리", "유리"]

    report = run_terms_analysis(terms, clauses_demo, labels_demo)

    print("\n◆ 약관 요약 ◆\n", report["terms_summary"])
    for i, (r, lab) in enumerate(zip(report["clause_results"], labels_demo), 1):
        print(f"\n◇ 조항 {i} ({lab})")
        print("도메인:", r["domain"])
        print(r["llm_result"])
        if r["law_refs"]:
            print("관련 법령:", ", ".join(r["law_refs"]))
