import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import pickle
from .config import config
from .utils import logger
import torch


class EmbeddingManager:
    def __init__(
        self,
        model_name_or_path: str = config.embedding_model,
        local: bool = False,
        device: str | None = None
    ):
        """Объект для работы с моделями эмбеддингов

        :param str model_name_or_path: путь к модели или ее название, defaults to settings.embedding_model
        :param bool local: если True, загружать модель из локального пути, defaults to False
        :param str | None device: устройство для вычислений, defaults to None
        """

        logger.info(f"Загрузка модели эмбеддингов {model_name_or_path}...")

        if local or os.path.exists(model_name_or_path):
            self.model = SentenceTransformer(model_name_or_path)
        else:
            self.model = SentenceTransformer(model_name_or_path)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model.to(device)
        logger.info(f"Модель эмбеддингов перемещена на {device}")

        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks_metadata = []
        self.model_name = model_name_or_path

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Вычислить эмбеддинги

        :param List[str] texts: список текстов
        :param bool is_query: является ли текст запросом, defaults to False
        :return np.ndarray: массив эмбеддингов
        """
        if "e5" in self.model_name.lower():
            if is_query:
                texts = [f"query: {t}" for t in texts]
            else:
                texts = [f"passage: {t}" for t in texts]

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def build_index(self, chunks: List[Dict]):
        """Построить индекс

        :param List[Dict] chunks: список чанков
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.encode(texts, is_query=False)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks_metadata = chunks
        logger.info(f"Индекс построен, добавлено {len(chunks)} чанков.")

    def save(
        self,
        index_path: str = os.path.join(config.index_dir, "faiss.index"),
        meta_path: str = os.path.join(
            config.index_dir, "chunks_metadata.pkl")
    ):
        """Сохранить индекс

        :param str index_path: путь к индексу, defaults to os.path.join(config.index_dir, "faiss.index")
        :param str meta_path: путь к файлу метаданных, defaults to os.path.join( config.index_dir, "chunks_metadata.pkl")
        """
        os.makedirs(config.index_dir, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)
        logger.info(f"Индекс сохранён в {index_path}")

    def load(
        self,
        index_path: str = os.path.join(config.index_dir, "faiss.index"),
        meta_path: str = os.path.join(config.index_dir, "chunks_metadata.pkl")
    ):
        """Загрузить индекс

        :param str index_path: путь к индексу, defaults to os.path.join(config.index_dir, "faiss.index")
        :param str meta_path: путь к файлу метаданных, defaults to os.path.join( config.index_dir, "chunks_metadata.pkl")
        """
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.chunks_metadata = pickle.load(f)
        logger.info(
            f"Индекс загружен из {index_path}, чанков: {len(self.chunks_metadata)}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск в индексе

        :param str query: текстовый запрос
        :param int top_k: _, defaults to 5
        :raises ValueError: индекс не загружен
        :return List[Dict]: список чанков
        """
        if self.index is None:
            raise ValueError("Индекс не загружен или не построен.")
        query_emb = self.encode([query], is_query=True)
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                chunk = self.chunks_metadata[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)
        return results
