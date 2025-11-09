from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)

USER_REQUEST_PROMPT = (
    "Interpret the following text as a description of the kind of song or musical mood the person wants.\n"
    "Focus on emotional tone, atmosphere, and energy level rather than literal meaning.\n\n"
    'Request: "{user_text}"'
)

LYRICS_PROMPT = (
    "Interpret the following lyrics as expressing a musical mood or emotional atmosphere.\n"
    "Focus on feelings, tone, and intensity rather than the literal story or characters.\n\n"
    'Lyrics: "{lyrics_text}"'
)


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        try:
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_song(self, song_name: str, artist: str, lyrics: Optional[str] = None) -> np.ndarray:
        text_parts = [
            USER_REQUEST_PROMPT.format(user_text=f"{song_name} by {artist}")
        ]
        if lyrics:
            text_parts.append(LYRICS_PROMPT.format(lyrics_text=lyrics))
        
        combined_text = "\n\n".join(text_parts)
        return self.encode_text(combined_text)
    
    def encode_emotion(self, emotion: str) -> np.ndarray:
        emotion_text = USER_REQUEST_PROMPT.format(
            user_text=f"This music feels {emotion}. The mood is {emotion}."
        )
        return self.encode_text(emotion_text)
    
    def combine_embeddings(
        self,
        embeddings: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        embeddings_array = np.array(embeddings)
        
        if weights is None:
            combined = np.mean(embeddings_array, axis=0)
        else:
            if len(weights) != len(embeddings):
                raise ValueError("Number of weights must match number of embeddings")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            
            weights_array = np.array(weights).reshape(-1, 1)
            combined = np.sum(embeddings_array * weights_array, axis=0)
        
        combined = combined / np.linalg.norm(combined)
        return combined
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        normalized_similarity = (similarity + 1) / 2
        
        return float(normalized_similarity)
    
    def batch_similarity(self, query_emb: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        
        query_emb = query_emb.flatten()

        query_norm = query_emb / np.linalg.norm(query_emb)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = np.dot(embeddings_norm, query_norm)
        normalized_similarities = (similarities + 1) / 2
        
        return normalized_similarities
