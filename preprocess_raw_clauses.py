import os
import re
import xml.etree.ElementTree as ET

def extract_clauses_from_xml(xml_path):
    """개별 XML 파일에서 <cn> 태그의 약관 본문을 조항 단위로 분리"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        cn_text = root.find('.//cn').text
        if not cn_text:
            return []

        # '제X조' 또는 '제 X 조' 기준으로 분할
        split_text = re.split(r'(제\s?\d+\s?조)', cn_text)

        clauses = []
        for i in range(1, len(split_text), 2):
            title = split_text[i].strip()
            body = split_text[i+1].strip() if i+1 < len(split_text) else ""
            clauses.append(f"{title} {body}")

        return clauses

    except Exception as e:
        print(f"[오류] {xml_path} 처리 실패: {e}")
        return []

def load_clauses_from_paths(folder_paths):
    """여러 개 경로에서 약관 조항 추출"""
    all_clauses = []
    for folder in folder_paths:
        print(f"📂 처리 중: {folder}")
        for fname in os.listdir(folder):
            if fname.endswith(".xml"):
                full_path = os.path.join(folder, fname)
                clauses = extract_clauses_from_xml(full_path)
                all_clauses.extend(clauses)
    return all_clauses

def save_clauses_to_txt(clauses, output_path="processed_clauses.txt"):
    """조항 리스트를 텍스트 파일로 저장 (조항당 한 줄)"""
    with open(output_path, "w", encoding="utf-8") as f:
        for clause in clauses:
            f.write(clause.strip() + "\n")
    print(f"✅ 총 {len(clauses)}개 조항이 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # ✅ 경로 설정 (TS + VS 하위 폴더)
    folder_paths = [
        "C:/TS_2.약관/01.유리",
        "C:/TS_2.약관/02.불리",
        "C:/VS_2.약관/01.유리",
        "C:/VS_2.약관/02.불리",
    ]

    # 1. 조항 추출
    clauses = load_clauses_from_paths(folder_paths)

    # 2. 저장
    save_clauses_to_txt(clauses, "processed_clauses.txt")