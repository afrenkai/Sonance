from typing import Dict, Tuple, Optional, List, Union
import logging
from backend.models.schemas import EmotionType
from backend.services.llm_emotion_service import LLMEmotionService

logger = logging.getLogger(__name__)


class EmotionMapper:
    def __init__(self, use_llm: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        self.use_llm = use_llm
        
        if use_llm:
            logger.info("Initializing emotion mapper with LLM-based contextual learning")
            self.llm_emotion_service = LLMEmotionService(model_name=model_name)
        else:
            logger.info("Initializing emotion mapper with static rules")
            self.llm_emotion_service = None
        
        self.emotion_mappings: Dict[str, Dict[str, Tuple[float, float]]] = {
            EmotionType.HAPPY: {
                "valence": (0.6, 1.0),
                "energy": (0.5, 1.0),
                "danceability": (0.5, 1.0),
                "tempo": (100, 180),
            },
            EmotionType.SAD: {
                "valence": (0.0, 0.4),
                "energy": (0.0, 0.5),
                "acousticness": (0.3, 1.0),
                "tempo": (60, 100),
            },
            EmotionType.ENERGETIC: {
                "energy": (0.7, 1.0),
                "danceability": (0.6, 1.0),
                "tempo": (120, 200),
            },
            EmotionType.CALM: {
                "valence": (0.3, 0.7),
                "energy": (0.0, 0.4),
                "acousticness": (0.4, 1.0),
                "tempo": (60, 100),
            },
            EmotionType.ANGRY: {
                "valence": (0.0, 0.3),
                "energy": (0.7, 1.0),
                "loudness": (-10, 0),
                "tempo": (120, 180),
            },
            EmotionType.MELANCHOLIC: {
                "valence": (0.0, 0.4),
                "energy": (0.2, 0.5),
                "acousticness": (0.4, 1.0),
                "instrumentalness": (0.0, 0.7),
            },
            EmotionType.HOPEFUL: {
                "valence": (0.4, 0.8),
                "energy": (0.4, 0.7),
                "acousticness": (0.2, 0.8),
            },
            EmotionType.ROMANTIC: {
                "valence": (0.4, 0.8),
                "energy": (0.2, 0.6),
                "acousticness": (0.3, 0.9),
                "danceability": (0.3, 0.7),
            },
            EmotionType.ANXIOUS: {
                "valence": (0.2, 0.5),
                "energy": (0.5, 0.9),
                "tempo": (100, 160),
            },
            EmotionType.PEACEFUL: {
                "valence": (0.4, 0.8),
                "energy": (0.0, 0.3),
                "acousticness": (0.5, 1.0),
                "instrumentalness": (0.2, 1.0),
            },
            # Extended emotions
            EmotionType.TIRED: {
                "valence": (0.2, 0.5),
                "energy": (0.0, 0.3),
                "tempo": (60, 90),
                "acousticness": (0.3, 0.8),
            },
            EmotionType.CONFIDENT: {
                "valence": (0.5, 0.9),
                "energy": (0.6, 1.0),
                "loudness": (-8, 0),
                "tempo": (100, 140),
            },
            EmotionType.REBELLIOUS: {
                "valence": (0.3, 0.7),
                "energy": (0.7, 1.0),
                "loudness": (-8, 0),
                "tempo": (120, 180),
            },
            EmotionType.PLAYFUL: {
                "valence": (0.6, 1.0),
                "energy": (0.5, 0.9),
                "danceability": (0.5, 1.0),
                "tempo": (100, 150),
            },
            EmotionType.SENSUAL: {
                "valence": (0.4, 0.7),
                "energy": (0.3, 0.6),
                "danceability": (0.4, 0.8),
                "tempo": (80, 120),
            },
            EmotionType.EMPOWERED: {
                "valence": (0.5, 0.9),
                "energy": (0.6, 1.0),
                "loudness": (-10, 0),
                "tempo": (100, 150),
            },
            EmotionType.VULNERABLE: {
                "valence": (0.2, 0.6),
                "energy": (0.2, 0.5),
                "acousticness": (0.4, 1.0),
                "tempo": (70, 110),
            },
            EmotionType.MYSTERIOUS: {
                "valence": (0.2, 0.5),
                "energy": (0.3, 0.6),
                "acousticness": (0.2, 0.7),
                "instrumentalness": (0.3, 0.8),
            },
            EmotionType.DREAMY: {
                "valence": (0.3, 0.7),
                "energy": (0.2, 0.5),
                "acousticness": (0.3, 0.8),
                "instrumentalness": (0.1, 0.6),
            },
            EmotionType.GRATEFUL: {
                "valence": (0.6, 0.9),
                "energy": (0.3, 0.7),
                "acousticness": (0.3, 0.8),
            },
            EmotionType.LONELY: {
                "valence": (0.0, 0.3),
                "energy": (0.1, 0.4),
                "acousticness": (0.4, 1.0),
                "tempo": (60, 100),
            },
            EmotionType.INSPIRED: {
                "valence": (0.5, 0.9),
                "energy": (0.5, 0.9),
                "tempo": (100, 140),
            },
            EmotionType.CONFLICTED: {
                "valence": (0.2, 0.5),
                "energy": (0.4, 0.7),
                "tempo": (90, 130),
            },
            EmotionType.CAREFREE: {
                "valence": (0.6, 1.0),
                "energy": (0.4, 0.8),
                "danceability": (0.4, 0.8),
                "tempo": (90, 130),
            },
            EmotionType.LOVE: {
                "valence": (0.5, 0.9),
                "energy": (0.3, 0.7),
                "acousticness": (0.2, 0.8),
                "danceability": (0.3, 0.7),
            },
            EmotionType.NOSTALGIC: {
                "valence": (0.3, 0.6),
                "energy": (0.2, 0.5),
                "acousticness": (0.4, 0.9),
                "tempo": (70, 110),
            },
        }
        
        logger.info(f"Emotion mapper initialized with {len(self.emotion_mappings)} predefined emotions (LLM: {use_llm})")
    
   
    def _parse_custom_emotion(self, emotion: str) -> Dict[str, Tuple[float, float]]:
        feature_ranges = {}
        
        keywords = {
            "happy": EmotionType.HAPPY,
            "joy": EmotionType.HAPPY,
            "joyful": EmotionType.HAPPY,
            "cheerful": EmotionType.HAPPY,
            "glad": EmotionType.HAPPY,
            "excited": EmotionType.HAPPY,
            
            "sad": EmotionType.SAD,
            "depressed": EmotionType.SAD,
            "down": EmotionType.SAD,
            "unhappy": EmotionType.SAD,
            "sorrowful": EmotionType.SAD,
            
            "melancholy": EmotionType.MELANCHOLIC,
            "melancholic": EmotionType.MELANCHOLIC,
            "wistful": EmotionType.MELANCHOLIC,
            "bittersweet": EmotionType.MELANCHOLIC,
            
            "energetic": EmotionType.ENERGETIC,
            "hyper": EmotionType.ENERGETIC,
            "upbeat": EmotionType.ENERGETIC,
            "pumped": EmotionType.ENERGETIC,
            "active": EmotionType.ENERGETIC,
            
            "calm": EmotionType.CALM,
            "relaxed": EmotionType.CALM,
            "chill": EmotionType.CALM,
            "mellow": EmotionType.CALM,
            "tranquil": EmotionType.CALM,
            
            "angry": EmotionType.ANGRY,
            "rage": EmotionType.ANGRY,
            "aggressive": EmotionType.ANGRY,
            "furious": EmotionType.ANGRY,
            "mad": EmotionType.ANGRY,
            
            "hopeful": EmotionType.HOPEFUL,
            "optimistic": EmotionType.HOPEFUL,
            
            "romantic": EmotionType.ROMANTIC,
            "love": EmotionType.LOVE,
            "loving": EmotionType.LOVE,
            
            "anxious": EmotionType.ANXIOUS,
            "nervous": EmotionType.ANXIOUS,
            "worried": EmotionType.ANXIOUS,
            "stressed": EmotionType.ANXIOUS,
            
            "peaceful": EmotionType.PEACEFUL,
            "serene": EmotionType.PEACEFUL,
            
            # Extended emotions
            "tired": EmotionType.TIRED,
            "exhausted": EmotionType.TIRED,
            "weary": EmotionType.TIRED,
            "sleepy": EmotionType.TIRED,
            
            "confident": EmotionType.CONFIDENT,
            "bold": EmotionType.CONFIDENT,
            "brave": EmotionType.CONFIDENT,
            "proud": EmotionType.CONFIDENT,
            
            "rebellious": EmotionType.REBELLIOUS,
            "rebel": EmotionType.REBELLIOUS,
            "defiant": EmotionType.REBELLIOUS,
            
            "playful": EmotionType.PLAYFUL,
            "fun": EmotionType.PLAYFUL,
            "silly": EmotionType.PLAYFUL,
            "lighthearted": EmotionType.PLAYFUL,
            
            "sensual": EmotionType.SENSUAL,
            "sexy": EmotionType.SENSUAL,
            "seductive": EmotionType.SENSUAL,
            
            "empowered": EmotionType.EMPOWERED,
            "powerful": EmotionType.EMPOWERED,
            "strong": EmotionType.EMPOWERED,
            
            "vulnerable": EmotionType.VULNERABLE,
            "fragile": EmotionType.VULNERABLE,
            "exposed": EmotionType.VULNERABLE,
            
            "mysterious": EmotionType.MYSTERIOUS,
            "enigmatic": EmotionType.MYSTERIOUS,
            "dark": EmotionType.MYSTERIOUS,
            
            "dreamy": EmotionType.DREAMY,
            "ethereal": EmotionType.DREAMY,
            "floating": EmotionType.DREAMY,
            
            "grateful": EmotionType.GRATEFUL,
            "thankful": EmotionType.GRATEFUL,
            "blessed": EmotionType.GRATEFUL,
            
            "lonely": EmotionType.LONELY,
            "alone": EmotionType.LONELY,
            "isolated": EmotionType.LONELY,
            
            "inspired": EmotionType.INSPIRED,
            "motivated": EmotionType.INSPIRED,
            "driven": EmotionType.INSPIRED,
            
            "conflicted": EmotionType.CONFLICTED,
            "torn": EmotionType.CONFLICTED,
            "confused": EmotionType.CONFLICTED,
            
            "carefree": EmotionType.CAREFREE,
            "free": EmotionType.CAREFREE,
            "easy": EmotionType.CAREFREE,
            
            "nostalgic": EmotionType.NOSTALGIC,
            "reminiscent": EmotionType.NOSTALGIC,
        }
        
        matched_emotions = []
        for keyword, emotion_type in keywords.items():
            if keyword in emotion:
                matched_emotions.append(emotion_type)

        if matched_emotions:
            feature_ranges = self._combine_emotion_ranges(matched_emotions)
        
        return feature_ranges
    
    def _combine_emotion_ranges(
        self,
        emotions: list
    ) -> Dict[str, Tuple[float, float]]:
        
        if not emotions:
            return {}
        
        combined = {}
        
       
        all_features = set()
        for emotion in emotions:
            all_features.update(self.emotion_mappings[emotion].keys())
        
        for feature in all_features:
            mins, maxs = [], []
            for emotion in emotions:
                if feature in self.emotion_mappings[emotion]:
                    min_val, max_val = self.emotion_mappings[emotion][feature]
                    mins.append(min_val)
                    maxs.append(max_val)
            
            if mins and maxs:
                combined[feature] = (
                    sum(mins) / len(mins),
                    sum(maxs) / len(maxs)
                )
        
        return combined
    
   
    def analyze_emotions(self, emotions: list) -> dict:
        if not self.use_llm or not self.llm_emotion_service:
            logger.warning("Emotion analysis requires LLM to be enabled")
            return {"error": "LLM emotion service not available"}
        
        return self.llm_emotion_service.analyze_multi_emotion_query(emotions)
    
    def find_similar_emotions(self, emotion: str, top_k: int = 3) -> list:
        if not self.use_llm or not self.llm_emotion_service:
            logger.warning("Finding similar emotions requires LLM to be enabled")
            return []
        
        return self.llm_emotion_service.find_related_emotions(emotion, top_k=top_k)
