import re
import os
import glob
import pymupdf
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import Engine
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy.sql import text

# --- 1. 설정 및 로깅 ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 로드
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
LAW_DIRECTORY = os.getenv("LAW_DIRECTORY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

try:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))
except (TypeError, ValueError):
    logging.error("EMBEDDING_DIMENSION is not set or not an integer in .env file.")
    exit(1)

# 필수 환경 변수 검증
required_vars = [DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, LAW_DIRECTORY, EMBEDDING_MODEL_NAME]
if not all(required_vars):
    logging.error("Not all required environment variables are set. Please check your .env file.")
    exit(1)
    
DB_CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# --- 2. 법률 파싱 함수 ---

PAT_PYEON = re.compile(r'^(제\d+편.*)')
PAT_JANG = re.compile(r'^(제\d+장.*)')
PAT_JEOL = re.compile(r'^(제\d+절.*)')
PAT_BUCHIK = re.compile(r'^\s*부\s*칙')
PAT_JO = re.compile(r'^((제\d+조(?:의\d+)?\s*\([^)]+\))).*') 

def extract_full_text(pdf_path: str) -> str:
    """PDF 파일에서 정렬된 텍스트를 추출합니다."""
    try:
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text(sort=True) + "\n"
        doc.close()
        return full_text
    except Exception as e:
        logging.error(f"Error processing file {pdf_path}: {e}")
        return ""

def clean_and_prepare_text(full_text: str) -> str:
    """머리글, 바닥글, 괄호 주석을 제거합니다."""
    header_footer_patterns = [
        r"^\s*법제처\s*.*?국가법령정보센터\s*$", 
        r"^\s*민법\s*$",              
        r"^\s*\d+\s*$",                
    ]
    cleaned_text = full_text
    for pattern in header_footer_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\[[^\]]+\]', '', cleaned_text)
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text, flags=re.MULTILINE)
    return cleaned_text.strip()

def save_previous_chunk(chunks, lines, source, pyeon, jang, jeol, jo_title):
    """이전 '조' 청크를 저장하는 헬퍼 함수"""
    if lines and jo_title:
        chunks.append({
            "page_content": "\n".join(lines),
            "metadata": {
                "source": source,
                "pyeon": pyeon,
                "jang": jang,
                "jeol": jeol,
                "jo_title": jo_title
            }
        })

def generate_law_chunks(cleaned_text: str, source_name: str) -> list[dict]:
    """텍스트를 '조' 단위로 청킹하고 계층적 메타데이터를 추가합니다."""
    chunks = []
    current_pyeon = "N/A"
    current_jang = "N/A"
    current_jeol = "N/A"
    current_jo_title = None
    current_chunk_lines = []
    started = False

    lines = cleaned_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue

        pyeon_match = PAT_PYEON.match(line)
        if pyeon_match:
            save_previous_chunk(chunks, current_chunk_lines, source_name, current_pyeon, current_jang, current_jeol, current_jo_title)
            started = True
            current_pyeon = pyeon_match.group(1).strip()
            current_jang = "N/A"; current_jeol = "N/A"; current_jo_title = None
            current_chunk_lines = []
            continue

        jang_match = PAT_JANG.match(line)
        if jang_match:
            save_previous_chunk(chunks, current_chunk_lines, source_name, current_pyeon, current_jang, current_jeol, current_jo_title)
            started = True
            current_jang = jang_match.group(1).strip()
            current_jeol = "N/A"; current_jo_title = None
            current_chunk_lines = []
            continue

        jeol_match = PAT_JEOL.match(line)
        if jeol_match:
            save_previous_chunk(chunks, current_chunk_lines, source_name, current_pyeon, current_jang, current_jeol, current_jo_title)
            started = True
            current_jeol = jeol_match.group(1).strip()
            current_jo_title = None
            current_chunk_lines = []
            continue

        buchik_match = PAT_BUCHIK.match(line)
        if buchik_match:
            save_previous_chunk(chunks, current_chunk_lines, source_name, current_pyeon, current_jang, current_jeol, current_jo_title)
            started = True
            current_pyeon = "부칙"; current_jang = "N/A"; current_jeol = "N/A"
            current_jo_title = None
            current_chunk_lines = []
            continue

        jo_match = PAT_JO.match(line)
        if jo_match:
            save_previous_chunk(chunks, current_chunk_lines, source_name, current_pyeon, current_jang, current_jeol, current_jo_title)
            started = True
            current_jo_title = jo_match.group(1).strip()
            current_chunk_lines = [line]
            continue
            
        if not started: continue

        if current_jo_title:
            current_chunk_lines.append(line)

    save_previous_chunk(chunks, current_chunk_lines, source_name, current_pyeon, current_jang, current_jeol, current_jo_title)
    return chunks

# --- 3. PGvector 데이터베이스 설정 ---

Base = declarative_base()

class LawDocument(Base):
    """법률 문서 청크를 저장하기 위한 SQLAlchemy 모델"""
    __tablename__ = 'law_documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(JSONB)
    embedding = Column(Vector(EMBEDDING_DIMENSION))

def setup_database() -> Engine:
    """데이터베이스 연결 및 테이블 생성"""
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        
        with engine.connect() as connection:
            connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            connection.commit()

        Base.metadata.create_all(engine)
        logging.info(f"Table '{LawDocument.__tablename__}' ready and PGvector extension enabled.")
        return engine
        
    except Exception as e: # 요청대로 단일 Exception으로 처리
        logging.error(f"Database connection failed: {e}")
        logging.error("Check if PGvector Docker container is running and .env settings are correct.")
        exit(1)

# --- 4. 메인 실행 스크립트 ---

def main():
    """'법률' 디렉토리의 모든 PDF를 PGvector에 적재합니다."""
    
    engine = setup_database()
    Session = sessionmaker(bind=engine)
    session = Session()

    logging.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        session.close()
        return

    pdf_files = glob.glob(os.path.join(LAW_DIRECTORY, "*.pdf"))
    if not pdf_files:
        logging.warning(f"Warning: No PDF files found in directory: '{LAW_DIRECTORY}'")
        session.close()
        return
        
    logging.info(f"Found {len(pdf_files)} PDF files to process.")

    total_chunks = 0
    try:
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            logging.info(f"\n--- Processing file: '{filename}' ---")

            full_text = extract_full_text(pdf_path)
            if not full_text:
                logging.warning(f"Could not extract text from '{filename}'. Skipping.")
                continue
            
            cleaned_text = clean_and_prepare_text(full_text)
            
            chunks = generate_law_chunks(cleaned_text, source_name=filename)
            if not chunks:
                logging.warning(f"No valid law chunks found in '{filename}'. Skipping.")
                continue

            logging.info(f"Generated {len(chunks)} chunks from '{filename}'.")

            contents_to_embed = [chunk['page_content'] for chunk in chunks]
            logging.info(f"Embedding {len(contents_to_embed)} chunks...")
            embeddings = model.encode(contents_to_embed, show_progress_bar=True)
            logging.info("Embedding complete.")

            documents_to_add = []
            for i, chunk in enumerate(chunks):
                doc = LawDocument(
                    content=chunk['page_content'],
                    doc_metadata=chunk['metadata'],
                    embedding=embeddings[i]
                )
                documents_to_add.append(doc)
            
            session.add_all(documents_to_add)
            session.commit()
            
            logging.info(f"Successfully ingested {len(documents_to_add)} chunks from '{filename}'.")
            total_chunks += len(documents_to_add)

    except Exception as e:
        logging.error(f"Error during ingestion process: {e}")
        session.rollback()
    finally:
        session.close()
        logging.info(f"\n--- Ingestion complete ---")
        logging.info(f"Total {total_chunks} law chunks ingested into PGvector.")

if __name__ == "__main__":
    main()