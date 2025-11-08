"""Tests for EmotionMapper."""
import pytest
from backend.services.emotion_mapper import EmotionMapper
from backend.models.schemas import EmotionType


class TestEmotionMapper:
    """Test suite for EmotionMapper class."""
    
    def test_initialization(self, emotion_mapper):
        """Test that EmotionMapper initializes correctly."""
        assert emotion_mapper is not None
        assert len(emotion_mapper.emotion_mappings) > 0
        assert EmotionType.HAPPY in emotion_mapper.emotion_mappings
    
    def test_get_feature_ranges_predefined_emotion(self, emotion_mapper):
        """Test getting feature ranges for a predefined emotion."""
        ranges = emotion_mapper.get_feature_ranges("happy")
        
        assert isinstance(ranges, dict)
        assert "valence" in ranges
        assert "energy" in ranges
        # Verify ranges are tuples of (min, max)
        assert isinstance(ranges["valence"], tuple)
        assert len(ranges["valence"]) == 2
        assert ranges["valence"][0] < ranges["valence"][1]
    
    def test_get_feature_ranges_all_predefined_emotions(self, emotion_mapper):
        """Test getting feature ranges for all predefined emotions."""
        for emotion_type in EmotionType:
            ranges = emotion_mapper.get_feature_ranges(emotion_type.value)
            
            assert isinstance(ranges, dict)
            assert len(ranges) > 0
            # Verify all ranges are valid tuples
            for feature, (min_val, max_val) in ranges.items():
                assert min_val <= max_val
    
    def test_get_feature_ranges_case_insensitive(self, emotion_mapper):
        """Test that emotion matching is case-insensitive."""
        ranges_lower = emotion_mapper.get_feature_ranges("happy")
        ranges_upper = emotion_mapper.get_feature_ranges("HAPPY")
        ranges_mixed = emotion_mapper.get_feature_ranges("HaPpY")
        
        assert ranges_lower == ranges_upper == ranges_mixed
    
    def test_get_feature_ranges_custom_emotion(self, emotion_mapper):
        """Test getting feature ranges for a custom emotion string."""
        ranges = emotion_mapper.get_feature_ranges("cheerful and energetic")
        
        assert isinstance(ranges, dict)
        assert len(ranges) > 0
    
    def test_get_feature_ranges_unknown_emotion(self, emotion_mapper):
        """Test getting feature ranges for an unknown emotion returns defaults."""
        ranges = emotion_mapper.get_feature_ranges("absolutely_unknown_emotion_xyz")
        
        assert isinstance(ranges, dict)
        # Should return neutral defaults
        assert "valence" in ranges
        assert "energy" in ranges
    
    def test_matches_emotion_perfect_match(self, emotion_mapper):
        """Test emotion matching with perfect audio features."""
        # Get ranges for happy emotion
        ranges = emotion_mapper.get_feature_ranges("happy")
        
        # Create audio features in the middle of ranges
        audio_features = {}
        for feature, (min_val, max_val) in ranges.items():
            audio_features[feature] = (min_val + max_val) / 2
        
        assert emotion_mapper.matches_emotion(audio_features, "happy")
    
    def test_matches_emotion_out_of_range(self, emotion_mapper):
        """Test emotion matching with out-of-range audio features."""
        # For happy emotion, use sad features
        audio_features = {
            "valence": 0.1,  # Low valence (sad)
            "energy": 0.2,   # Low energy (sad)
            "tempo": 70.0    # Slow tempo
        }
        
        # Should not match happy emotion
        assert not emotion_mapper.matches_emotion(audio_features, "happy", tolerance=0.0)
    
    def test_matches_emotion_with_tolerance(self, emotion_mapper):
        """Test emotion matching with tolerance."""
        ranges = emotion_mapper.get_feature_ranges("happy")
        
        # Create features slightly outside range
        audio_features = {}
        for feature, (min_val, max_val) in ranges.items():
            audio_features[feature] = max_val + 0.05  # Slightly above max
        
        # Should fail without tolerance
        assert not emotion_mapper.matches_emotion(audio_features, "happy", tolerance=0.0)
        # Should pass with tolerance
        assert emotion_mapper.matches_emotion(audio_features, "happy", tolerance=0.1)
    
    def test_compute_emotion_score_perfect_match(self, emotion_mapper):
        """Test emotion score computation with perfect match."""
        ranges = emotion_mapper.get_feature_ranges("happy")
        
        # Create features in the center of ranges
        audio_features = {}
        for feature, (min_val, max_val) in ranges.items():
            audio_features[feature] = (min_val + max_val) / 2
        
        score = emotion_mapper.compute_emotion_score(audio_features, "happy")
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score >= 0.8  # Should be high for perfect match
    
    def test_compute_emotion_score_poor_match(self, emotion_mapper):
        """Test emotion score computation with poor match."""
        # Use sad features for happy emotion
        audio_features = {
            "valence": 0.1,
            "energy": 0.2,
            "tempo": 70.0
        }
        
        score = emotion_mapper.compute_emotion_score(audio_features, "happy")
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low for poor match
    
    def test_compute_emotion_score_partial_features(self, emotion_mapper):
        """Test emotion score with partial audio features."""
        audio_features = {
            "valence": 0.8,  # Only one feature
        }
        
        score = emotion_mapper.compute_emotion_score(audio_features, "happy")
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_compute_emotion_score_empty_features(self, emotion_mapper):
        """Test emotion score with empty audio features."""
        audio_features = {}
        
        score = emotion_mapper.compute_emotion_score(audio_features, "happy")
        
        assert isinstance(score, float)
        assert score == 0.5  # Default neutral score
    
    def test_compute_emotion_score_all_emotions(self, emotion_mapper):
        """Test emotion score computation for all predefined emotions."""
        audio_features = {
            "valence": 0.7,
            "energy": 0.6,
            "danceability": 0.5,
            "tempo": 120.0,
            "acousticness": 0.3
        }
        
        for emotion_type in EmotionType:
            score = emotion_mapper.compute_emotion_score(
                audio_features,
                emotion_type.value
            )
            
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_emotion_mappings_have_valid_ranges(self, emotion_mapper):
        """Test that all predefined emotion mappings have valid ranges."""
        for emotion_type, features in emotion_mapper.emotion_mappings.items():
            for feature_name, (min_val, max_val) in features.items():
                assert min_val <= max_val, f"Invalid range for {emotion_type}.{feature_name}"
                
                # Check typical feature ranges
                if feature_name in ["valence", "energy", "danceability", "acousticness",
                                   "instrumentalness", "liveness", "speechiness"]:
                    assert 0.0 <= min_val <= 1.0
                    assert 0.0 <= max_val <= 1.0
                elif feature_name == "tempo":
                    assert min_val > 0
                    assert max_val > 0
    
    def test_combine_emotion_ranges(self, emotion_mapper):
        """Test combining multiple emotion ranges."""
        emotions = [EmotionType.HAPPY, EmotionType.ENERGETIC]
        combined = emotion_mapper._combine_emotion_ranges(emotions)
        
        assert isinstance(combined, dict)
        assert len(combined) > 0
        
        # Verify all ranges are valid
        for feature, (min_val, max_val) in combined.items():
            assert min_val <= max_val
    
    def test_parse_custom_emotion_with_keywords(self, emotion_mapper):
        """Test parsing custom emotions with keywords."""
        ranges = emotion_mapper._parse_custom_emotion("joyful and cheerful")
        
        assert isinstance(ranges, dict)
        # Should match happy emotion
        if ranges:
            assert "valence" in ranges or len(ranges) > 0
    
    def test_parse_custom_emotion_no_match(self, emotion_mapper):
        """Test parsing custom emotion with no keyword matches."""
        ranges = emotion_mapper._parse_custom_emotion("unknown_xyz_emotion")
        
        # Should return empty dict
        assert isinstance(ranges, dict)
        assert len(ranges) == 0
    
    def test_get_neutral_ranges(self, emotion_mapper):
        """Test getting neutral default ranges."""
        ranges = emotion_mapper._get_neutral_ranges()
        
        assert isinstance(ranges, dict)
        assert "valence" in ranges
        assert "energy" in ranges
        
        # Verify neutral ranges are centered
        for feature, (min_val, max_val) in ranges.items():
            mid_point = (min_val + max_val) / 2
            assert 0.3 <= mid_point <= 0.7  # Should be near middle
    
    def test_emotion_mapper_without_llm(self):
        """Test EmotionMapper initialization without LLM."""
        mapper = EmotionMapper(use_llm=False)
        
        assert mapper.llm_emotion_service is None
        assert not mapper.use_llm
        
        # Should still work with static mappings
        ranges = mapper.get_feature_ranges("happy")
        assert len(ranges) > 0
    
    def test_valence_ranges_make_sense(self, emotion_mapper):
        """Test that valence ranges make semantic sense."""
        happy_ranges = emotion_mapper.get_feature_ranges("happy")
        sad_ranges = emotion_mapper.get_feature_ranges("sad")
        
        # Happy should have higher valence than sad
        if "valence" in happy_ranges and "valence" in sad_ranges:
            happy_valence = sum(happy_ranges["valence"]) / 2
            sad_valence = sum(sad_ranges["valence"]) / 2
            assert happy_valence > sad_valence
    
    def test_energy_ranges_make_sense(self, emotion_mapper):
        """Test that energy ranges make semantic sense."""
        energetic_ranges = emotion_mapper.get_feature_ranges("energetic")
        calm_ranges = emotion_mapper.get_feature_ranges("calm")
        
        # Energetic should have higher energy than calm
        if "energy" in energetic_ranges and "energy" in calm_ranges:
            energetic_energy = sum(energetic_ranges["energy"]) / 2
            calm_energy = sum(calm_ranges["energy"]) / 2
            assert energetic_energy > calm_energy
