from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import config


def split_into_chunks(pages: List[Dict]) -> List[Dict]:
    """Разбивает текст на чанки.

    :param List[Dict] pages: Страницы документа
    :return List[Dict]: Список чанков
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for page in pages:
        page_text = page['text']
        metadata = {
            "source": page['source'],
            "page": page['page']
        }
        page_chunks = text_splitter.split_text(page_text)
        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": metadata,
                "chunk_id": f"{metadata['source']}_p{metadata['page']}_c{i}"
            })

    return chunks
