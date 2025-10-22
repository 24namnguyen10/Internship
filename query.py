import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # <-- THAY ĐỔI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Hằng số ---
DB_DIR = "db_chroma"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # <-- Model local

# --- Prompt Template (Giữ nguyên) ---
RAG_PROMPT_TEMPLATE = """
Dựa CHÍNH XÁC vào nội dung được cung cấp dưới đây để trả lời câu hỏi.
KHÔNG sử dụng bất kỳ kiến thức nào khác bên ngoài.
Nếu nội dung không chứa câu trả lời, hãy nói: "Tôi không tìm thấy thông tin trả lời trong tài liệu."

Nội dung:
{context}

Câu hỏi:
{question}

Câu trả lời (chỉ dựa vào nội dung trên):
"""

def main():
    # 1. Tải API Key (Vẫn cần cho LLM chat)
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Lỗi: Không tìm thấy GOOGLE_API_KEY.")
        return

    # 2. Kiểm tra thư mục DB
    if not os.path.exists(DB_DIR):
        print(f"Lỗi: Không tìm thấy thư mục vector store '{DB_DIR}'.")
        print("Vui lòng chạy file 'ingest.py' trước để tạo database.")
        return

    # 3. Tải (Load) Embedding Model (Local) <-- THAY ĐỔI
    print(f"Đang tải embedding model: {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # 4. Tải (Load) ChromaDB đã tồn tại
    print(f"Đang tải vector store từ thư mục: {DB_DIR}...")
    db = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=embeddings # <-- Sử dụng embedding local
    )

    # 5. Tạo Retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 6. Tải (Load) LLM (Mô hình Gemini)
    print("Đang khởi tạo mô hình ngôn ngữ (LLM)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Giữ nguyên model chat
        google_api_key=api_key
    )

    # 7. Tạo RAG Chain (Giữ nguyên)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {"context": lambda x: format_docs(retriever.invoke(x["question"])), "question": lambda x: x["question"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 8. Bắt đầu vòng lặp hỏi-đáp
    print("\n--- 🤖 Bot đã sẵn sàng! ---")
    print("Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát).")

    while True:
        question_text = input("\nBạn hỏi: ")
        if question_text.lower() == 'exit':
            break
        
        print("Bot đang tìm câu trả lời...")
        
        try:
            chain_input = {"question": question_text}
            answer = rag_chain.invoke(chain_input)
            print(f"\nBot trả lời:\n{answer}")
        except Exception as e:
            # Lỗi 429 vẫn có thể xảy ra ở đây (cho LLM chat), nhưng ít hơn
            print(f"Đã xảy ra lỗi khi xử lý câu hỏi của bạn: {e}")

    print("Tạm biệt!")

if __name__ == "__main__":
    main()