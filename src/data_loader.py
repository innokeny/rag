import os
from pypdf import PdfReader
from typing import List, Dict
from .config import config
from .utils import logger


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Разбивает pdf-файл на страницы и извлекает текст

    :param str pdf_path: Путь к pdf-файлу
    :return List[Dict]: Список страниц
    """

    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            pages.append({
                "page": page_num,
                "text": text,
                "source": os.path.basename(pdf_path)
            })
    return pages


def load_all_documents() -> List[Dict]:
    """Загружает все pdf-файлы из директории и извлекает текст

    :return List[Dict]: Список страниц
    """
    
    all_pages = []
    for filename in os.listdir(config.docs_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(config.docs_dir, filename)
            logger.info(f"Обработка {pdf_path}...")
            pages = extract_text_from_pdf(pdf_path)
            all_pages.extend(pages)
    logger.info(f"Всего загружено {len(all_pages)} страниц.")
    return all_pages
