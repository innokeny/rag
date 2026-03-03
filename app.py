from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse 
from pydantic import BaseModel
import os
from src.config import config
from src.data_loader import load_all_documents
from src.chunking import split_into_chunks
from src.embeddings import EmbeddingManager
from src.retriever import Retriever
from src.generator import Generator
from src.graph import RAGGraph
from src.utils import logger
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    global emb_manager, retriever, generator, rag_graph
    logger.info("Инициализация RAG-системы...")
    index_path = os.path.join(config.index_dir, "faiss.index")
    meta_path = os.path.join(config.index_dir, "chunks_metadata.pkl")

    emb_manager = EmbeddingManager()

    if os.path.exists(index_path) and os.path.exists(meta_path):
        logger.info("Загрузка существующего индекса...")
        try:
            emb_manager.load(index_path, meta_path)
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}. Создание нового.")
            pages = load_all_documents()
            chunks = split_into_chunks(pages)
            emb_manager.build_index(chunks)
            emb_manager.save(index_path, meta_path)
    else:
        logger.info("Индекс не найден. Создание нового...")
        pages = load_all_documents()
        chunks = split_into_chunks(pages)
        emb_manager.build_index(chunks)
        emb_manager.save(index_path, meta_path)

    retriever = Retriever(emb_manager)
    generator = Generator()
    rag_graph = RAGGraph(retriever, generator)
    logger.info("Система готова к работе.")

    yield 

    logger.info("Остановка RAG-системы...")


app = FastAPI(
    title="RAG",
    lifespan=lifespan 
)

emb_manager = None
retriever = None
generator = None
rag_graph = None

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list

@app.post("/ask", response_model=AnswerResponse) 
async def ask_question(request: QuestionRequest):
    if not rag_graph:
        raise HTTPException(status_code=503, detail="Система ещё не инициализирована")
    try:
        result = rag_graph.run(request.question)
        sources = [
            {
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"].get("page"), 
                "score": chunk["score"]
            }
            for chunk in result.get("context_chunks", []) 
        ]
        return AnswerResponse(answer=result.get("answer", ""), sources=sources)
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.get("/health")
async def health():
    if rag_graph:
        return {"status": "ok", "rag_initialized": True}
    else:
        return JSONResponse(content={"status": "degraded", "rag_initialized": False}, status_code=503)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
