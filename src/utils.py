import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deduplicate_chunks(chunks: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
    if len(chunks) <= 1:
        return chunks
    
    texts = [chunk['text'] for chunk in chunks]
    
    unique_chunks = []
    seen_texts = set()
    
    for chunk in chunks:
        signature = chunk['text'][:100]
        if signature not in seen_texts:
            seen_texts.add(signature)
            unique_chunks.append(chunk)
    
    return unique_chunks

def format_context(chunks: List[Dict], deduplicate: bool = True) -> str:
    if deduplicate:
        chunks = deduplicate_chunks(chunks)
    
    context_parts = []
    for i, chunk in enumerate(chunks):
        text = chunk['text'].strip()
        text = ' '.join(text.split())
        context_parts.append(f"[{i+1}] {text}")
    
    return "\n\n".join(context_parts)