from .embeddings import EmbeddingManager
from .config import config

class Retriever:
    def __init__(self, emb_manager: EmbeddingManager):
        self.emb_manager = emb_manager

    def retrieve(self, query: str, top_k: int = config.top_k, diversity_penalty: float = 0.5) -> list:
        """Кандидатогенерация с учетом диверсификации

        :param str query: Запрос
        :param int top_k: Размер выборки, defaults to config.top_k
        :param float diversity_penalty: Параметр диверсификации, defaults to 0.5
        :return list: Список кандидатов
        """
        results = self.emb_manager.search(query, top_k=top_k * 2)
        
        if len(results) <= top_k:
            return results
        
        selected = [results[0]]
        candidates = results[1:]
        
        while len(selected) < top_k and candidates:
            best_candidate = None
            best_score = -float('inf')
            
            for i, cand in enumerate(candidates):
                relevance = cand['score']
                
                max_similarity = 0
                for sel in selected:
                    if cand['metadata']['source'] == sel['metadata']['source'] and \
                    abs(cand['metadata']['page'] - sel['metadata']['page']) < 5:
                        max_similarity = 0.9
                
                mmr_score = relevance - diversity_penalty * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = i
            
            if best_candidate is not None:
                selected.append(candidates.pop(best_candidate))
            else:
                break
        
        filtered = [r for r in selected if r['score'] >= config.similarity_threshold]
        return filtered if filtered else selected