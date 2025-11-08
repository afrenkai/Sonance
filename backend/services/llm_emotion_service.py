import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class LLMEmotionService:
    """
    Contextual emotion understanding using sentence transformers.
    Learns emotion representations from examples and context rather than hardcoded rules.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing LLM Emotion Service with {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize emotion embeddings from rich contextual descriptions
        self.emotion_embeddings = self._build_emotion_embeddings()
        
        # Cache for computed embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized with {len(self.emotion_embeddings)} emotion profiles")
    
    def _build_emotion_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Build rich contextual embeddings for each emotion.
        Uses multiple descriptive sentences to capture nuance.
        """
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
            # Combine multiple contextual descriptions
            embeddings = self.model.encode(contexts, convert_to_numpy=True)
            # Average the embeddings for a richer representation
            emotion_embeddings[emotion] = np.mean(embeddings, axis=0)
        
        return emotion_embeddings
    
    def get_emotion_embedding(self, emotion: str) -> np.ndarray:
        """Get or compute embedding for an emotion."""
        emotion_lower = emotion.lower().strip()
        
        # Check if we have a pre-computed embedding
        if emotion_lower in self.emotion_embeddings:
            return self.emotion_embeddings[emotion_lower]
        
        # Check cache
        if emotion_lower in self._embedding_cache:
            return self._embedding_cache[emotion_lower]
        
        # Compute new embedding for custom emotion
        contexts = [
            f"music that feels {emotion_lower}",
            f"songs with {emotion_lower} mood and atmosphere",
            f"{emotion_lower} emotional vibes and energy"
        ]
        embeddings = self.model.encode(contexts, convert_to_numpy=True)
        emotion_emb = np.mean(embeddings, axis=0)
        
        # Cache it
        self._embedding_cache[emotion_lower] = emotion_emb
        
        return emotion_emb
    
    def compute_emotion_similarity(
        self, 
        text: str, 
        target_emotion: str,
        context: str = "song"
    ) -> float:
        """
        Compute how well text matches a target emotion using contextual understanding.
        
        Args:
            text: Text to analyze (song name, lyrics snippet, description)
            target_emotion: Target emotion to match against
            context: Context type ("song", "lyrics", "description")
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Contextualize the text
        if context == "song":
            contextualized = f"This song is: {text}"
        elif context == "lyrics":
            contextualized = f"Lyrics expressing emotion: {text}"
        else:
            contextualized = text
        
        # Get embeddings
        text_emb = self.model.encode([contextualized], convert_to_numpy=True)[0]
        emotion_emb = self.get_emotion_embedding(target_emotion)
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            text_emb.reshape(1, -1),
            emotion_emb.reshape(1, -1)
        )[0][0]
        
        # Normalize to 0-1 range
        normalized = (similarity + 1) / 2
        
        return float(normalized)
    
    def find_related_emotions(
        self,
        emotion: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find emotions most similar to the given emotion.
        
        Args:
            emotion: Query emotion
            top_k: Number of related emotions to return
            
        Returns:
            List of (emotion_name, similarity_score) tuples
        """
        query_emb = self.get_emotion_embedding(emotion)
        
        similarities = []
        for emo_name, emo_emb in self.emotion_embeddings.items():
            if emo_name.lower() == emotion.lower():
                continue
            
            sim = cosine_similarity(
                query_emb.reshape(1, -1),
                emo_emb.reshape(1, -1)
            )[0][0]
            similarities.append((emo_name, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def infer_emotion_from_audio_features(
        self,
        audio_features: Dict[str, float],
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Infer what emotions match given audio features using contextual understanding.
        
        Args:
            audio_features: Dictionary of Spotify audio features
            candidates: Optional list of candidate emotions to rank
            
        Returns:
            List of (emotion, score) tuples ranked by match
        """
        if candidates is None:
            candidates = list(self.emotion_embeddings.keys())
        
        # Create descriptive text from audio features
        descriptions = []
        
        if 'valence' in audio_features:
            valence = audio_features['valence']
            if valence > 0.7:
                descriptions.append("very positive and uplifting")
            elif valence > 0.5:
                descriptions.append("somewhat positive")
            elif valence < 0.3:
                descriptions.append("dark and negative")
        
        if 'energy' in audio_features:
            energy = audio_features['energy']
            if energy > 0.7:
                descriptions.append("high energy and intense")
            elif energy < 0.3:
                descriptions.append("low energy and subdued")
        
        if 'danceability' in audio_features:
            dance = audio_features['danceability']
            if dance > 0.7:
                descriptions.append("very danceable and rhythmic")
        
        if 'acousticness' in audio_features:
            acoustic = audio_features['acousticness']
            if acoustic > 0.7:
                descriptions.append("acoustic and organic")
        
        if 'tempo' in audio_features:
            tempo = audio_features['tempo']
            if tempo > 140:
                descriptions.append("fast-paced")
            elif tempo < 80:
                descriptions.append("slow tempo")
        
        if not descriptions:
            return [(e, 0.5) for e in candidates]
        
        # Combine descriptions
        feature_text = "music that is " + ", ".join(descriptions)
        
        # Score against each candidate emotion
        scores = []
        for emotion in candidates:
            score = self.compute_emotion_similarity(
                feature_text,
                emotion,
                context="description"
            )
            scores.append((emotion, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_audio_feature_guidance(
        self,
        emotion: str,
        confidence: float = 0.7
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get suggested audio feature ranges for an emotion using learned patterns.
        This provides guidance but isn't strictly enforced.
        
        Args:
            emotion: Target emotion
            confidence: How strict the ranges should be (0-1)
            
        Returns:
            Dictionary of feature ranges
        """
        emotion_lower = emotion.lower().strip()
        
        # Create contextual descriptions of audio characteristics
        emotion_audio_profiles = {
            "happy": {
                "valence": (0.6, 1.0),
                "energy": (0.5, 1.0),
                "danceability": (0.5, 1.0),
            },
            "sad": {
                "valence": (0.0, 0.4),
                "energy": (0.0, 0.5),
            },
            "energetic": {
                "energy": (0.7, 1.0),
                "danceability": (0.6, 1.0),
            },
            "calm": {
                "energy": (0.0, 0.4),
                "acousticness": (0.4, 1.0),
            },
            "angry": {
                "valence": (0.0, 0.3),
                "energy": (0.7, 1.0),
            },
            "melancholic": {
                "valence": (0.0, 0.4),
                "energy": (0.2, 0.5),
            },
            "hopeful": {
                "valence": (0.4, 0.8),
                "energy": (0.4, 0.7),
            },
            "romantic": {
                "valence": (0.4, 0.8),
                "energy": (0.2, 0.6),
            },
            "anxious": {
                "valence": (0.2, 0.5),
                "energy": (0.5, 0.9),
            },
            "peaceful": {
                "valence": (0.4, 0.8),
                "energy": (0.0, 0.3),
            }
        }
        
        if emotion_lower in emotion_audio_profiles:
            base_ranges = emotion_audio_profiles[emotion_lower]
        else:
            # For unknown emotions, find similar ones
            related = self.find_related_emotions(emotion_lower, top_k=2)
            if related:
                # Blend the ranges from similar emotions
                base_ranges = {}
                for rel_emotion, similarity in related:
                    if rel_emotion in emotion_audio_profiles:
                        for feature, (min_val, max_val) in emotion_audio_profiles[rel_emotion].items():
                            if feature not in base_ranges:
                                base_ranges[feature] = [[], []]
                            base_ranges[feature][0].append(min_val * similarity)
                            base_ranges[feature][1].append(max_val * similarity)
                
                # Average the ranges
                base_ranges = {
                    feature: (
                        sum(vals[0]) / len(vals[0]),
                        sum(vals[1]) / len(vals[1])
                    )
                    for feature, vals in base_ranges.items()
                }
            else:
                # Default neutral ranges
                base_ranges = {
                    "valence": (0.3, 0.7),
                    "energy": (0.3, 0.7),
                }
        
        # Adjust ranges based on confidence
        # Lower confidence = wider ranges
        adjusted_ranges = {}
        for feature, (min_val, max_val) in base_ranges.items():
            center = (min_val + max_val) / 2
            spread = (max_val - min_val) / 2
            
            # Scale spread inversely with confidence
            adjusted_spread = spread / confidence
            
            adjusted_ranges[feature] = (
                max(0.0, center - adjusted_spread),
                min(1.0, center + adjusted_spread)
            )
        
        return adjusted_ranges
    
    def analyze_multi_emotion_query(
        self,
        emotions: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze a query with multiple emotions to understand their relationship.
        
        Args:
            emotions: List of emotion strings
            
        Returns:
            Analysis including dominant emotion, conflicts, and blended representation
        """
        if not emotions:
            return {"error": "No emotions provided"}
        
        # Get embeddings for all emotions
        emotion_embs = [self.get_emotion_embedding(e) for e in emotions]
        
        # Compute pairwise similarities
        similarities = []
        for i, e1 in enumerate(emotions):
            for j, e2 in enumerate(emotions):
                if i < j:
                    sim = cosine_similarity(
                        emotion_embs[i].reshape(1, -1),
                        emotion_embs[j].reshape(1, -1)
                    )[0][0]
                    similarities.append((e1, e2, float(sim)))
        
        # Identify conflicts (low similarity)
        conflicts = [(e1, e2, sim) for e1, e2, sim in similarities if sim < 0.3]
        
        # Identify harmonies (high similarity)
        harmonies = [(e1, e2, sim) for e1, e2, sim in similarities if sim > 0.7]
        
        # Create blended embedding
        blended_emb = np.mean(emotion_embs, axis=0)
        
        # Find which predefined emotion is closest to the blend
        best_match = None
        best_score = -1
        for emo_name, emo_emb in self.emotion_embeddings.items():
            sim = cosine_similarity(
                blended_emb.reshape(1, -1),
                emo_emb.reshape(1, -1)
            )[0][0]
            if sim > best_score:
                best_score = sim
                best_match = emo_name
        
        return {
            "emotions": emotions,
            "blended_emotion": best_match,
            "blend_confidence": float(best_score),
            "conflicts": conflicts,
            "harmonies": harmonies,
            "is_coherent": len(conflicts) == 0
        }
