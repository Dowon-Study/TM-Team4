
import os
import pickle
import faiss
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai

# ───────────────────────────── 0. 로깅 설정 ────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"rag_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)
logger.info("로그 파일 생성: %s", log_file)

# ─────────────────────── 1. 경로·모델·키 설정 ─────────────────────────────
BASE_DIR   = Path("rag_index_build")
INDEX_PATH = BASE_DIR / "faiss_index/index.faiss"
IDS_PKL    = BASE_DIR / "faiss_index/ids_list.pkl"
META_PKL   = BASE_DIR / "faiss_index/index_meta.pkl"

EMB_MODEL  = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL  = "gpt-4o"

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.critical("OPENAI_API_KEY 환경 변수가 비어 있습니다.")
    raise EnvironmentError("OPENAI_API_KEY not set")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("DEVICE=%s | EMB_MODEL=%s | LLM_MODEL=%s", DEVICE, EMB_MODEL, LLM_MODEL)

# ─────────────────────── 2. FAISS 인덱스 & 문서 로드 ───────────────────────
logger.info("FAISS 인덱스 로드: %s", INDEX_PATH)
faiss_index = faiss.read_index(str(INDEX_PATH))

ids_list: List[str] = pickle.loads(Path(IDS_PKL).read_bytes())
id2meta: Dict[str, dict] = pickle.loads(Path(META_PKL).read_bytes())
logger.debug("문서 개수=%d", len(ids_list))

docs = [
    Document(
        page_content=id2meta[doc_id]["orig_text"],
        metadata={
            "id":           doc_id,
            "std_text":     id2meta[doc_id]["std_text"],
            "advantageous": "유리" if id2meta[doc_id]["dvAntageous"] == "1" else "불리",
            "source_file":  id2meta[doc_id]["source_file"],
        },
    )
    for doc_id in ids_list
]
docstore = InMemoryDocstore({d.metadata["id"]: d for d in docs})

embedding = SentenceTransformerEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={"device": DEVICE},
)
vectorstore = FAISS(
    embedding_function=embedding,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id={i: doc.metadata["id"] for i, doc in enumerate(docs)},
)
logger.info("VectorStore 생성 완료 (문서 %d 개)", len(docs))

# ─────────────────────── 3. Self-Query Retriever & QA 체인 ─────────────────
metadata_field_info = [
    AttributeInfo(
        name="advantageous",
        description="조항이 소비자에게 유리한지 불리한지 여부(유리/불리)",
        type="string",
        enum=["유리", "불리"],
    )
]
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.2)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="조항 전문",
    metadata_field_info=metadata_field_info,
    search_kwargs={"k": 3},
    verbose=True,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
logger.info("Self-Query Retriever & QA 체인 초기화 완료")

# ───────────────────────── 4. 기능 함수 정의 ──────────────────────────────
def summarize_terms(full_text: str) -> str:
    logger.debug("약관 요약 시작 (chars=%d)", len(full_text))
    prompt = (
        "다음 약관 전문을 소비자 관점에서 이해하기 쉽도록 요약해 주세요.\n\n"
        f"{full_text}"
    )
    summary = llm.predict(prompt=prompt).strip()
    logger.debug("약관 요약 완료 (chars=%d)", len(summary))
    return summary


def make_explanation(orig: str, std: str) -> str:
    prompt = (
        "아래 두 조항을 비교해 소비자 관점에서 왜 현재 약관이 불리한지 2~3문장으로 설명하십시오.\n\n"
        f"(현재 약관)\n{orig}\n\n(표준 예시)\n{std}"
    )
    rsp = llm.predict(prompt=prompt).strip()
    logger.debug("불리 설명 생성 (chars=%d)", len(rsp))
    return rsp


def make_revision(orig: str, std: str) -> str:
    prompt = (
        "다음 내용을 '개정 전 vs 개정 후' 형식으로 작성하세요.\n\n"
        f"(개정 전)\n{orig}\n\n(개정 후)\n{std}"
    )
    rsp = llm.predict(prompt=prompt).strip()
    logger.debug("개정 예시 생성 (chars=%d)", len(rsp))
    return rsp


def process_clauses(clause_items: List[dict], top_k: int = 3):
    """
    clause_items format:
        [
            {"text": "<조항 본문>", "label": "불리"},
            {"text": "<조항 본문>", "label": "유리"},
            ...
        ]
    label must be '유리' or '불리'
    """
    results = []
    for idx, item in enumerate(clause_items, 1):
        clause_text  = item["text"]
        clause_label = item["label"].strip()

        logger.info("▶ 조항 %d/%d 처리 | 라벨=%s", idx, len(clause_items), clause_label)
        query = f"[{clause_label} 조항] {clause_text}"
        qa_res = qa_chain({"query": query})
        llm_summary = qa_res["result"]
        src_docs = qa_res["source_documents"]
        logger.debug("   Self-Query 검색 문서 %d 개", len(src_docs))

        explanations, revisions = [], []
        for rank, d in enumerate(src_docs, 1):
            std = d.metadata["std_text"]
            explanations.append(make_explanation(clause_text, std))
            revisions.append(make_revision(clause_text, std))
            logger.debug("   후보 %d 완료", rank)

        results.append(
            {
                "orig_clause": clause_text,
                "orig_label":  clause_label,
                "llm_summary": llm_summary,
                "retrieved_docs": src_docs,
                "explanations": explanations,
                "revisions": revisions,
            }
        )
    return results

# ───────────────────────── 5. 메인 실행 예시 ──────────────────────────────
if __name__ == "__main__":
    # 약관 전문 요약
    terms_path = Path("terms.txt")
    if not terms_path.exists():
        logger.error("terms.txt 파일을 찾을 수 없습니다.")
        exit(1)

    terms_text = terms_path.read_text(encoding="utf-8")
    print("\n========= [약관 요약] =========")
    print(summarize_terms(terms_text))

    # 조항 + 라벨 입력 예시
    clause_items = [
        {
            "text": "제10조(위약금) ① 고객이 계약을 중도 해지할 경우 총 결제 금액의 50%를 위약금으로 부과합니다.",
            "label": "불리"
        },
        {
            "text": "제5조(청약철회) ① 고객은 상품 수령 후 7일 이내 자유롭게 청약철회할 수 있습니다.",
            "label": "유리"
        },
    ]

    outputs = process_clauses(clause_items, top_k=3)

    # 콘솔 출력
    for i, item in enumerate(outputs, 1):
        print(f"\n========== 조항 {i} ({item['orig_label']}) ==========")
        print(item["orig_clause"])
        print("\n[LLM 요약·논평]\n", item["llm_summary"])
        for rank, doc in enumerate(item["retrieved_docs"], 1):
            meta = doc.metadata
            print(f"\n--- 표준 후보 {rank} (ID={meta['id']}, 라벨={meta['advantageous']}) ---")
            print("표준 예시:\n", meta["std_text"])
            print("\n왜 불리한지:\n", item["explanations"][rank-1])
            print("\n개정 전 vs 개정 후:\n", item["revisions"][rank-1])

    logger.info("스크립트 실행 완료")