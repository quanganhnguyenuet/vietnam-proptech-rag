# Vietnam Legal & PropTech RAG (`vn-legal-rag`)

Hệ thống RAG trợ lý tư vấn Pháp luật & Bất động sản Việt Nam, hỗ trợ tra cứu Luật Đất đai, Luật Nhà ở và các Nghị định hướng dẫn.

## 🛠️ Tech Stack
- **Framework:** FastAPI / LangChain (hoặc LlamaIndex)
- **Vector DB:** ChromaDB / Qdrant (Local)
- **Embedding & Reranker:** PhoBERT / BGE-Reranker / OpenAI
- **Search:** Hybrid Search (BM25 + Vector Search)

## 🚀 Hướng dẫn cài đặt & Chạy Local
1. Clone repo và tạo virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate