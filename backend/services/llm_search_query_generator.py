import logging
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class LLMSearchQueryGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", spotify_service=None):
        logger.info(f"Initializing LLM Search Query Generator with {model_name}")
        self.model = SentenceTransformer(model_name)
        # Note: spotify_service kept for compatibility but genre APIs are deprecated
        self.spotify_service = spotify_service
        
        self._build_search_vocabulary()
        
        logger.info("LLM Search Query Generator initialized with learned vocabulary")
    
    def _build_search_vocabulary(self):
        self.genre_descriptions = {
            "indie": "independent alternative music with artistic creativity and emotional depth",
            "pop": "popular mainstream catchy accessible music with broad appeal",
            "rock": "guitar-driven energetic powerful music with strong rhythms",
            "electronic": "synthesizer electronic beats digital production modern sounds",
            "folk": "acoustic traditional storytelling organic natural instruments",
            "r&b": "rhythm and blues soul smooth vocals emotional expression",
            "hip-hop": "rap beats urban poetry rhythmic spoken word",
            "jazz": "improvisation sophisticated complex harmonies instrumental",
            "classical": "orchestral traditional composed instrumental sophisticated",
            "metal": "heavy distorted aggressive intense powerful dark",
            "punk": "fast raw rebellious energetic simple direct",
            "ambient": "atmospheric soundscape minimal relaxing textural",
            "soul": "emotional expressive vocals heartfelt passion",
            "country": "storytelling acoustic traditional americana roots",
            "reggae": "offbeat rhythm relaxed caribbean groove positive",
            "blues": "emotional guitar melancholic storytelling raw feelings",
        }
        
        self.mood_descriptors = {
            "uplifting": "inspiring hopeful positive bright encouraging energizing",
            "melancholic": "sad reflective nostalgic bittersweet wistful longing",
            "energetic": "active dynamic powerful intense driving exciting",
            "calm": "peaceful relaxing soothing gentle quiet tranquil",
            "dark": "moody atmospheric somber brooding intense heavy",
            "romantic": "loving tender intimate passionate affectionate emotional",
            "aggressive": "intense powerful forceful angry rebellious raw",
            "dreamy": "ethereal floating soft atmospheric hazy ambient",
            "groovy": "funky rhythmic danceable smooth moving infectious",
            "raw": "authentic unpolished emotional honest direct stripped",
        }
        
        logger.info("Encoding genre and mood vocabulary...")
        self.genre_embeddings = {
            genre: self.model.encode(desc, convert_to_numpy=True)
            for genre, desc in self.genre_descriptions.items()
        }
        self.mood_embeddings = {
            mood: self.model.encode(desc, convert_to_numpy=True)
            for mood, desc in self.mood_descriptors.items()
        }
        
        logger.info(f"Encoded {len(self.genre_embeddings)} genres and {len(self.mood_embeddings)} moods")
    
    def filter_relevant_genres(
        self,
        emotion: str,
        min_similarity: float = 0.15,
        max_genres: int = 10
    ) -> List[tuple]:
        """
        Filter predefined genres by semantic similarity to emotion.
        Note: Spotify genre seeds API is deprecated, so we use predefined genres only.
        """
        logger.info(f"Filtering {len(self.genre_embeddings)} predefined genres for emotion '{emotion}'")
        
        emotion_embedding = self.model.encode(
            f"music that feels {emotion}, songs with {emotion} mood and emotional vibe",
            convert_to_numpy=True
        )
        
        genre_scores = []
        for genre, genre_emb in self.genre_embeddings.items():
            similarity = self._cosine_similarity(emotion_embedding, genre_emb)
            if similarity >= min_similarity:  # Filter out irrelevant genres
                genre_scores.append((genre, float(similarity)))
        
        genre_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_genres = genre_scores[:max_genres]
        
        logger.info(
            f"Filtered to {len(relevant_genres)} relevant genres for '{emotion}': "
            f"{[f'{g}({s:.2f})' for g, s in relevant_genres[:5]]}"
        )
        
        if len(relevant_genres) < max_genres * 0.3: 
            logger.warning(
                f"Only found {len(relevant_genres)} relevant genres (threshold={min_similarity}), "
                f"might want to lower threshold"
            )
        
        return relevant_genres
    
    def generate_queries_for_emotion(
        self,
        emotion: str,
        num_queries: int = 6,
        include_year: bool = True,
        use_runtime_filtering: bool = True
    ) -> List[str]:
        """
        Generate search queries for an emotion using predefined genre embeddings.
        Note: Spotify genre seeds API is deprecated, so runtime corpus loading is disabled.
        """
        logger.info(f"Generating search queries for emotion: '{emotion}' using predefined genres")
        
        emotion_embedding = self.model.encode(
            f"music that feels {emotion}, songs with {emotion} mood and emotional vibe",
            convert_to_numpy=True
        )
        
        genre_similarities = {}
        for genre, genre_emb in self.genre_embeddings.items():
            similarity = self._cosine_similarity(emotion_embedding, genre_emb)
            genre_similarities[genre] = similarity
        
        relevant_genres = {g: s for g, s in genre_similarities.items() if s >= 0.15}
        
        top_genres = sorted(relevant_genres.items(), key=lambda x: x[1], reverse=True)[:6]
        
        logger.info(
            f"Filtered predefined genres for '{emotion}': {[f'{g}({s:.2f})' for g, s in top_genres]}"
        )
        
        queries = []
        
        for genre, score in top_genres[:4]:
            if include_year and score > 0.25:
                queries.append(f"genre:{genre} year:2015-2024")
            else:
                queries.append(f"genre:{genre}")
        
        if top_genres and include_year:
            for genre, _ in top_genres[:2]:
                queries.append(f"genre:{genre} year:2010-2024")
        
        logger.info(f"Generated {len(queries)} queries from predefined genres")
        return queries[:num_queries]
    
    def generate_queries_for_seed_songs(
        self,
        seed_songs: List[tuple],
        num_queries: int = 7
    ) -> List[str]:
        logger.info(f"Generating search queries from {len(seed_songs)} seed songs")
        
        seed_texts = [f"{song} by {artist}" for song, artist in seed_songs]
        seed_embeddings = self.model.encode(seed_texts, convert_to_numpy=True)
        
        if len(seed_embeddings.shape) == 1:
            avg_embedding = seed_embeddings
        else:
            avg_embedding = np.mean(seed_embeddings, axis=0)
        
        genre_similarities = {}
        for genre, genre_emb in self.genre_embeddings.items():
            similarity = self._cosine_similarity(avg_embedding, genre_emb)
            genre_similarities[genre] = similarity
        
        mood_similarities = {}
        for mood, mood_emb in self.mood_embeddings.items():
            similarity = self._cosine_similarity(avg_embedding, mood_emb)
            mood_similarities[mood] = similarity
        
        top_genres = sorted(genre_similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        top_moods = sorted(mood_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        logger.info(
            f"Inferred from seeds - genres: {[g[0] for g in top_genres[:3]]}, "
            f"moods: {[m[0] for m in top_moods[:2]]}"
        )
        
        queries = []
        
        for genre, score in top_genres[:3]:
            queries.append(f"genre:{genre}")
        
        if top_moods and top_genres:
            mood = top_moods[0][0]
            for genre, _ in top_genres[:2]:
                queries.append(f"genre:{genre} {mood}")
                if len(queries) >= num_queries:
                    break
        
        if top_genres:
            queries.append(f"genre:{top_genres[0][0]} year:2010-2024")
        
        logger.info(f"Generated {len(queries)} seed-based queries")
        return queries[:num_queries]
    
    def infer_emotion_from_seeds(
        self,
        seed_songs: List[tuple],
        top_k: int = 3
    ) -> List[tuple]:
        logger.info(f"Inferring emotions from {len(seed_songs)} seed songs")
        
        seed_texts = [f"{song} by {artist}" for song, artist in seed_songs]
        seed_embeddings = self.model.encode(seed_texts, convert_to_numpy=True)
        
        if len(seed_embeddings.shape) == 1:
            avg_embedding = seed_embeddings
        else:
            avg_embedding = np.mean(seed_embeddings, axis=0)
        
        mood_scores = []
        for mood, mood_emb in self.mood_embeddings.items():
            similarity = self._cosine_similarity(avg_embedding, mood_emb)
            mood_scores.append((mood, float(similarity)))
        
        mood_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Inferred emotions: {mood_scores[:top_k]}")
        return mood_scores[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot_product = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
