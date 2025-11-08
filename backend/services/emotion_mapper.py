from typing import Dict, Tuple, Optional
import logging
from backend.models.schemas import EmotionType
from backend.services.llm_emotion_service import LLMEmotionService

logger = logging.getLogger(__name__)


class EmotionMapper:
    """
    Emotion mapper with contextual learning via sentence transformers.
    Now uses LLMEmotionService for intelligent, learnable emotion understanding.
    """
    
    def __init__(self, use_llm: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize emotion mapper.
        
        Args:
            use_llm: Whether to use LLM-based contextual emotion understanding
            model_name: Sentence transformer model to use
        """
        self.use_llm = use_llm
        
        if use_llm:
            logger.info("Initializing emotion mapper with LLM-based contextual learning")
            self.llm_emotion_service = LLMEmotionService(model_name=model_name)
        else:
            logger.info("Initializing emotion mapper with static rules")
            self.llm_emotion_service = None
        
        # Fallback static mappings (used when LLM is disabled or as guidance)
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
        }
        
        logger.info(f"Emotion mapper initialized with {len(self.emotion_mappings)} predefined emotions (LLM: {use_llm})")
    
    def get_feature_ranges(self, emotion: str) -> Dict[str, Tuple[float, float]]:
        """
        Get audio feature ranges for an emotion.
        If LLM is enabled, uses contextual learning; otherwise uses static mappings.
        """
        emotion_lower = emotion.lower().strip()
        
        # Use LLM-based guidance if available
        if self.use_llm and self.llm_emotion_service:
            try:
                # Get learned feature guidance with moderate confidence
                feature_ranges = self.llm_emotion_service.get_audio_feature_guidance(
                    emotion_lower,
                    confidence=0.6  # Moderate confidence = reasonable flexibility
                )
                if feature_ranges:
                    logger.debug(f"Using LLM-guided feature ranges for '{emotion}'")
                    return feature_ranges
            except Exception as e:
                logger.warning(f"LLM emotion guidance failed for '{emotion}': {e}, falling back to static")
        
        # Fallback to static mappings
        for emotion_type in EmotionType:
            if emotion_type.value == emotion_lower:
                return self.emotion_mappings[emotion_type]
        
        # Try to parse custom emotion
        feature_ranges = self._parse_custom_emotion(emotion_lower)
        
        if not feature_ranges:
            logger.warning(f"Unknown emotion '{emotion}', using neutral defaults")
            feature_ranges = self._get_neutral_ranges()
        
        return feature_ranges
    
    def _parse_custom_emotion(self, emotion: str) -> Dict[str, Tuple[float, float]]:
        feature_ranges = {}
        
        keywords = {
            "happy": EmotionType.HAPPY,
            "joy": EmotionType.HAPPY,
            "cheerful": EmotionType.HAPPY,
            "sad": EmotionType.SAD,
            "depressed": EmotionType.SAD,
            "melancholy": EmotionType.MELANCHOLIC,
            "melancholic": EmotionType.MELANCHOLIC,
            "energetic": EmotionType.ENERGETIC,
            "hyper": EmotionType.ENERGETIC,
            "upbeat": EmotionType.ENERGETIC,
            "calm": EmotionType.CALM,
            "relaxed": EmotionType.CALM,
            "chill": EmotionType.CALM,
            "angry": EmotionType.ANGRY,
            "rage": EmotionType.ANGRY,
            "aggressive": EmotionType.ANGRY,
            "hopeful": EmotionType.HOPEFUL,
            "optimistic": EmotionType.HOPEFUL,
            "romantic": EmotionType.ROMANTIC,
            "love": EmotionType.ROMANTIC,
            "anxious": EmotionType.ANXIOUS,
            "nervous": EmotionType.ANXIOUS,
            "peaceful": EmotionType.PEACEFUL,
            "serene": EmotionType.PEACEFUL,
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
    
    def _get_neutral_ranges(self) -> Dict[str, Tuple[float, float]]:
       
        return {
            "valence": (0.3, 0.7),
            "energy": (0.3, 0.7),
            "danceability": (0.3, 0.7),
            "tempo": (80, 140),
        }
    
    def matches_emotion(
        self,
        audio_features: Dict[str, float],
        emotion: str,
        tolerance: float = 0.1
    ) -> bool:
        
        feature_ranges = self.get_feature_ranges(emotion)
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature not in audio_features:
                continue
            
        
            min_val -= tolerance
            max_val += tolerance
            
            value = audio_features[feature]
            if not (min_val <= value <= max_val):
                return False
        
        return True
    
    def compute_emotion_score(
        self,
        audio_features: Dict[str, float],
        emotion: str
    ) -> float:
        """
        Compute how well audio features match an emotion.
        If LLM is enabled, uses contextual understanding; otherwise uses range matching.
        """
        # Use LLM-based inference if available
        if self.use_llm and self.llm_emotion_service:
            try:
                # Get emotion inference from audio features
                emotion_scores = self.llm_emotion_service.infer_emotion_from_audio_features(
                    audio_features,
                    candidates=[emotion.lower().strip()]
                )
                if emotion_scores:
                    _, score = emotion_scores[0]
                    logger.debug(f"LLM emotion score for '{emotion}': {score:.3f}")
                    return score
            except Exception as e:
                logger.warning(f"LLM emotion scoring failed: {e}, falling back to range-based")
        
        # Fallback to range-based scoring
        feature_ranges = self.get_feature_ranges(emotion)
        
        if not feature_ranges:
            return 0.5  
        
        scores = []
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature not in audio_features:
                continue
            
            value = audio_features[feature]
            
            # Score based on how well the value fits within the range
            if min_val <= value <= max_val:
                range_size = max_val - min_val
                if range_size > 0:
                    center = (min_val + max_val) / 2
                    distance_from_center = abs(value - center)
                    score = 1.0 - (distance_from_center / (range_size / 2))
                else:
                    score = 1.0
            else:
                if value < min_val:
                    distance = min_val - value
                else:
                    distance = value - max_val
                
                score = max(0.0, 1.0 - distance)
            
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def analyze_emotions(self, emotions: list) -> dict:
        """
        Analyze multiple emotions to understand their relationship.
        Only available when LLM is enabled.
        """
        if not self.use_llm or not self.llm_emotion_service:
            logger.warning("Emotion analysis requires LLM to be enabled")
            return {"error": "LLM emotion service not available"}
        
        return self.llm_emotion_service.analyze_multi_emotion_query(emotions)
    
    def find_similar_emotions(self, emotion: str, top_k: int = 3) -> list:
        """
        Find emotions similar to the given emotion.
        Only available when LLM is enabled.
        """
        if not self.use_llm or not self.llm_emotion_service:
            logger.warning("Finding similar emotions requires LLM to be enabled")
            return []
        
        return self.llm_emotion_service.find_related_emotions(emotion, top_k=top_k)
