import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_all_documents
from src.chunking import split_into_chunks
from src.embeddings import EmbeddingManager
from src.config import config
from src.utils import logger

def build_index_for_model(model_name_or_path: str, local: bool = False, force: bool = False):
    """
    Строит индекс для заданной модели эмбеддингов.
    
    Args:
        model_name_or_path: имя модели на HF Hub или локальный путь
        local: если True, загружать модель из локального пути
        force: если True, перестроить индекс даже если он уже существует
    """
    safe_name = model_name_or_path.replace('/', '_').replace('\\', '_')
    if local and os.path.exists(model_name_or_path):
        safe_name = Path(model_name_or_path).stem
    
    index_dir = Path(config.index_dir) / safe_name
    index_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = index_dir / 'faiss.index'
    meta_path = index_dir / 'chunks_metadata.pkl'
    
    if not force and index_path.exists() and meta_path.exists():
        logger.info(f"Индекс для {model_name_or_path} уже существует в {index_dir}. Используйте --force для перестроения.")
        return
    
    logger.info(f"Загрузка документов и создание чанков...")
    pages = load_all_documents()
    chunks = split_into_chunks(pages)
    logger.info(f"Создано {len(chunks)} чанков.")
    
    logger.info(f"Построение индекса с моделью {model_name_or_path}...")
    emb_manager = EmbeddingManager(model_name_or_path=model_name_or_path, local=local)
    emb_manager.build_index(chunks)
    emb_manager.save(str(index_path), str(meta_path))
    logger.info(f"Индекс сохранён в {index_dir}")

def main():
    parser = argparse.ArgumentParser(description="Построение FAISS индекса для модели эмбеддингов.")
    parser.add_argument('--model', type=str, default=config.embedding_model,
                        help=f"Имя модели на HF Hub или локальный путь (по умолчанию: {config.embedding_model})")
    parser.add_argument('--local', action='store_true',
                        help="Указывает, что model — локальный путь")
    parser.add_argument('--force', action='store_true',
                        help="Перестроить индекс, даже если он уже существует")
    
    args = parser.parse_args()
    build_index_for_model(args.model, local=args.local, force=args.force)

if __name__ == "__main__":
    main()