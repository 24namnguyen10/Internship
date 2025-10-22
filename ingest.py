import os
# KHÔNG cần dotenv nữa
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- Giữ nguyên bản sửa lỗi này
from langchain_huggingface import HuggingFaceEmbeddings # <-- THAY ĐỔI
from langchain_chroma import Chroma # <-- Giữ nguyên bản sửa lỗi này

# --- Hằng số ---
PDF_PATH = "ChiTietTTHC_2.002229.pdf"
DB_DIR = "db_chroma"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # <-- Model local

def main():
    # 1. Kiểm tra file PDF
    if not os.path.exists(PDF_PATH):
        print(f"Lỗi: Không tìm thấy file '{PDF_PATH}'.")
        return
    print(f"Đang tải file {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Đã tải {len(documents)} trang từ PDF.")

    # 2. Chia nhỏ văn bản
    print("Đang chia nhỏ văn bản thành các đoạn (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia văn bản thành {len(chunks)} đoạn.")

    # 3. Tải (Load) Embedding Model (Local) <-- THAY ĐỔI
    print(f"Đang tải embedding model: {EMBED_MODEL}...")
    print("Việc này có thể mất vài phút nếu là lần đầu tiên chạy...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'} # Sử dụng CPU
    )
    print("Đã tải xong embedding model.")

    # 4. Tạo và Lưu trữ (Ingest) vào ChromaDB
    print(f"Đang tạo và lưu trữ vector vào thư mục: {DB_DIR}...")
    
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=DB_DIR
    )
    
    print("\n--- HOÀN TẤT! ---")
    print(f"Đã tạo vector store thành công với Model Local.")
    print(f"Dữ liệu đã được lưu tại: {os.path.abspath(DB_DIR)}")

if __name__ == "__main__":
    main()