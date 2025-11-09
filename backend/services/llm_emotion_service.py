import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class LLMEmotionService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing LLM Emotion Service with {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.emotion_embeddings = self._build_emotion_embeddings()
        
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized with {len(self.emotion_embeddings)} emotion profiles")
    
    def _build_emotion_embeddings(self) -> Dict[str, np.ndarray]:
        emotion_contexts = {
            "happy": [
                "upbeat energetic joyful cheerful music that makes you want to dance and smile",
                "bright positive optimistic songs with uplifting melodies and happy lyrics",
                "feel-good party music with infectious energy and celebration vibes",
                "music that radiates joy, sunshine, and good vibes all around"
            ],
            "sad": [
                "melancholic emotional music about heartbreak and loss that brings tears",
                "slow somber songs with sorrowful melodies expressing pain and loneliness",
                "music about missing someone, feeling empty and broken inside",
                "tearjerker ballads with emotional depth and vulnerability"
            ],
            "energetic": [
                "high-energy intense powerful music that pumps you up for action",
                "fast-paced adrenaline-fueled songs perfect for workouts and motivation",
                "explosive dynamic tracks with driving beats and aggressive intensity",
                "music that makes you want to move, run, jump and unleash energy"
            ],
            "calm": [
                "peaceful relaxing soothing music for meditation and tranquility",
                "gentle quiet ambient sounds that help you breathe and unwind",
                "serene calming melodies for rest, sleep and stress relief",
                "soft mellow music that brings inner peace and stillness"
            ],
            "angry": [
                "aggressive intense furious music expressing rage and rebellion",
                "hard-hitting heavy songs with violent angry energy and distortion",
                "music channeling frustration, hatred and explosive emotions",
                "raw powerful tracks about fighting back and destructive fury"
            ],
            "melancholic": [
                "bittersweet nostalgic music tinged with sadness and longing",
                "wistful reflective songs about memories and things that fade away",
                "music with emotional depth expressing regret and yearning",
                "moody atmospheric tracks capturing autumn rain and dusk feelings"
            ],
            "hopeful": [
                "inspiring uplifting music about believing in better tomorrow",
                "optimistic encouraging songs about dreams, faith and rising above",
                "music that gives hope, motivation and belief in possibilities",
                "tracks about new beginnings, light after darkness and positive change"
            ],
            "romantic": [
                "intimate tender love songs about deep connection and devotion",
                "passionate romantic music expressing desire and affection",
                "soft sensual ballads about lovers, hearts and beautiful moments",
                "music capturing warmth, closeness and the magic of being in love"
            ],
            "anxious": [
                "tense nervous unsettling music expressing worry and fear",
                "restless uncertain tracks with building pressure and unease",
                "music capturing stress, panic and anxious overwhelming feelings",
                "dark suspenseful songs about doubt and nervous anticipation"
            ],
            "peaceful": [
                "tranquil harmonious serene music bringing balance and zen",
                "nature-inspired calming sounds of ocean breeze and gentle streams",
                "meditative peaceful tracks for mindfulness and inner stillness",
                "soothing ambient music creating atmosphere of complete peace"
            ]
        }
        
        emotion_embeddings = {}
        for emotion, contexts in emotion_contexts.items():
            embeddings = self.model.encode(contexts, convert_to_numpy=True)
            emotion_embeddings[emotion] = np.mean(embeddings, axis=0)
        
        return emotion_embeddings
    
    def get_emotion_embedding(self, emotion: str) -> np.ndarray:
        emotion_lower = emotion.lower().strip()
        
        if emotion_lower in self.emotion_embeddings:
            return self.emotion_embeddings[emotion_lower]
        
        if emotion_lower in self._embedding_cache:
            logger.debug(f"Using cached embedding for learned emotion '{emotion}'")
            return self._embedding_cache[emotion_lower]
        
        logger.info(f"Learning new emotion '{emotion}' through contextual understanding")
        
        contexts = [
            f"music that feels {emotion_lower}",
            f"songs with {emotion_lower} mood and atmosphere",
            f"{emotion_lower} emotional vibes and energy",
            f"the feeling of being {emotion_lower}",
            f"music that captures {emotion_lower} emotions"
        ]
        
        embeddings = self.model.encode(contexts, convert_to_numpy=True)
        emotion_emb = np.mean(embeddings, axis=0)
        
        self._embedding_cache[emotion_lower] = emotion_emb
        
        related = self.find_related_emotions(emotion_lower, top_k=3)
        if related:
            related_names = [name for name, _ in related]
            logger.info(
                f"Learned emotion '{emotion}' - most similar to: {', '.join(related_names)}"
            )
        
        return emotion_emb
    
    def compute_emotion_similarity(
        self, 
        text: str, 
        target_emotion: str,
        context: str = "song"
    ) -> float:
        if context == "song":
            contextualized = f"This song is: {text}"
        elif context == "lyrics":
            contextualized = f"Lyrics expressing emotion: {text}"
        else:
            contextualized = text
        
        text_emb = self.model.encode([contextualized], convert_to_numpy=True)[0]
        emotion_emb = self.get_emotion_embedding(target_emotion)
        
        similarity = cosine_similarity(text_emb, emotion_emb)
        
        normalized = (similarity + 1) / 2
        
        return float(normalized)
    
    def find_related_emotions(
        self,
        emotion: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        query_emb = self.get_emotion_embedding(emotion)
        
        similarities = []
        for emo_name, emo_emb in self.emotion_embeddings.items():
            if emo_name.lower() == emotion.lower():
                continue
            
            sim = cosine_similarity(query_emb, emo_emb)
            similarities.append((emo_name, float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
   
    def analyze_multi_emotion_query(
        self,
        emotions: List[str]
    ) -> Dict[str, Any]:
        if not emotions:
            return {"error": "No emotions provided"}
        
        logger.info(f"Analyzing multi-emotion query: {emotions}")
        
        emotion_embs = []
        learned_emotions = []
        for e in emotions:
            emb = self.get_emotion_embedding(e)
            emotion_embs.append(emb)
            # Track which emotions were newly learned
            if e.lower() not in self.emotion_embeddings:
                learned_emotions.append(e)
        
        if learned_emotions:
            logger.info(f"Learned {len(learned_emotions)} new emotions: {learned_emotions}")
        
        similarities = []
        for i, e1 in enumerate(emotions):
            for j, e2 in enumerate(emotions):
                if i < j:
                    sim = cosine_similarity(emotion_embs[i], emotion_embs[j])
                    similarities.append((e1, e2, float(sim)))
        
        conflicts = [(e1, e2, sim) for e1, e2, sim in similarities if sim < 0.3]
        
        harmonies = [(e1, e2, sim) for e1, e2, sim in similarities if sim > 0.7]
        
        blended_emb = np.mean(emotion_embs, axis=0)
        
        best_match = None
        best_score = -1
        for emo_name, emo_emb in self.emotion_embeddings.items():
            sim = cosine_similarity(blended_emb, emo_emb)
            if sim > best_score:
                best_score = sim
                best_match = emo_name
        
        analysis = {
            "emotions": emotions,
            "blended_emotion": best_match,
            "blend_confidence": float(best_score),
            "conflicts": conflicts,
            "harmonies": harmonies,
            "is_coherent": len(conflicts) == 0,
            "learned_emotions": learned_emotions
        }
        
        if conflicts:
            logger.warning(f"Emotion conflicts detected: {[(e1, e2) for e1, e2, _ in conflicts]}")
        
        if harmonies:
            logger.info(f"Harmonious emotion pairs: {[(e1, e2) for e1, e2, _ in harmonies]}")
        
        return analysis
    
    def get_learned_emotions(self) -> List[str]:
        return list(self._embedding_cache.keys())
