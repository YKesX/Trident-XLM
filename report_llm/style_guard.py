# Optional style/restriction guard using embedding-based similarity.
# Not required for core generation; uses only the candidate text (no wrappers).
from typing import List
from sentence_transformers import SentenceTransformer, util

class StyleGuard:
    def __init__(self, model_name: str = "google/embeddinggemma-300m"):
        self.model = SentenceTransformer(model_name)
    def score(self, candidate: str, anchors: List[str], banned: List[str]) -> tuple[float,float]:
        v_c = self.model.encode(candidate, normalize_embeddings=True)
        v_a = self.model.encode(anchors,  normalize_embeddings=True)
        v_b = self.model.encode(" ".join(banned), normalize_embeddings=True)
        sim_anchor = float(max(util.cos_sim(v_c, v_a)[0]))
        sim_banned = float(util.cos_sim(v_c, v_b))
        return sim_anchor, sim_banned
    def passes(self, candidate: str, anchors: List[str], banned: List[str],
               min_anchor: float = 0.75, max_banned: float = 0.20) -> bool:
        sa, sb = self.score(candidate, anchors, banned)
        return sa >= min_anchor and sb <= max_banned
