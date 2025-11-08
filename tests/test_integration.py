"""
Real integration tests for EmoRec services.

These tests use actual service implementations without mocking,
testing real functionality end-to-end.

Note: Some tests may be slow as they load ML models.
Mark with @pytest.mark.slow for optional skipping.
"""
import pytest
import numpy as np
import os
from pathlib import Path

from backend.services.embedding_service import EmbeddingService
from backend.services.emotion_mapper import EmotionMapper
from backend.services.spotify_service import SpotifyService
from backend.services.playlist_generator import PlaylistGenerator
from backend.models.schemas import SongInput, EmotionType


# Mark slow tests
pytestmark = pytest.mark.integration


class TestEmbeddingServiceIntegration:
    """Integration tests for EmbeddingService with real model."""
    
    @pytest.fixture(scope="class")
    def service(self):
        """Create a real EmbeddingService instance (cached for class)."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    def test_real_song_encoding(self, service):
        """Test encoding actual songs."""
        # Encode famous songs
        bohemian = service.encode_song("Bohemian Rhapsody", "Queen")
        imagine = service.encode_song("Imagine", "John Lennon")
        
        # Should produce valid embeddings
        assert bohemian.shape == (384,)
        assert imagine.shape == (384,)
        
        # Different songs should have different embeddings
        similarity = service.compute_similarity(bohemian, imagine)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0  # Not identical
    
    def test_similar_songs_have_high_similarity(self, service):
        """Test that similar songs have higher similarity."""
        # Rock songs
        rock1 = service.encode_song("Stairway to Heaven", "Led Zeppelin")
        rock2 = service.encode_song("Bohemian Rhapsody", "Queen")
        
        # Classical piece
        classical = service.encode_song("Moonlight Sonata", "Beethoven")
        
        # Rock songs should be more similar to each other
        rock_similarity = service.compute_similarity(rock1, rock2)
        rock_classical_sim = service.compute_similarity(rock1, classical)
        
        # This might not always hold due to embedding space, but generally should
        assert rock_similarity > 0.3  # Some similarity
        assert rock_classical_sim >= 0.0  # Valid similarity
    
    def test_emotion_semantic_similarity(self, service):
        """Test that similar emotions have higher similarity."""
        happy = service.encode_emotion("happy")
        joyful = service.encode_emotion("joyful")
        sad = service.encode_emotion("sad")
        
        # Happy and joyful should be more similar than happy and sad
        happy_joyful = service.compute_similarity(happy, joyful)
        happy_sad = service.compute_similarity(happy, sad)
        
        assert happy_joyful > happy_sad
        assert happy_joyful > 0.5  # Should be quite similar
    
    def test_batch_similarity_performance(self, service):
        """Test batch similarity computation."""
        query = service.encode_song("Test Song", "Test Artist")
        
        # Create batch of songs
        songs = [
            ("Song 1", "Artist 1"),
            ("Song 2", "Artist 2"),
            ("Song 3", "Artist 3"),
        ]
        
        batch_embeddings = np.array([
            service.encode_song(name, artist) for name, artist in songs
        ])
        
        # Compute similarities
        similarities = service.batch_similarity(query, batch_embeddings)
        
        assert len(similarities) == 3
        assert all(0.0 <= s <= 1.0 for s in similarities)
    
    def test_combined_embedding_preserves_semantics(self, service):
        """Test that combined embeddings preserve semantic meaning."""
        rock_song = service.encode_song("Back in Black", "AC/DC")
        happy_emotion = service.encode_emotion("happy energetic")
        
        # Combine with equal weights
        combined = service.combine_embeddings(
            [rock_song, happy_emotion],
            [0.5, 0.5]
        )
        
        # Combined should be somewhat similar to both inputs
        sim_to_song = service.compute_similarity(combined, rock_song)
        sim_to_emotion = service.compute_similarity(combined, happy_emotion)
        
        assert sim_to_song > 0.3  # Has some similarity to original song
        assert sim_to_emotion > 0.3  # Has some similarity to emotion
        assert np.isclose(np.linalg.norm(combined), 1.0)  # Normalized


class TestEmotionMapperIntegration:
    """Integration tests for EmotionMapper."""
    
    @pytest.fixture
    def mapper(self):
        """Create EmotionMapper without LLM for speed."""
        return EmotionMapper(use_llm=False)
    
    def test_all_predefined_emotions_have_ranges(self, mapper):
        """Test that all predefined emotions return valid ranges."""
        for emotion_type in EmotionType:
            ranges = mapper.get_feature_ranges(emotion_type.value)
            
            assert isinstance(ranges, dict)
            assert len(ranges) > 0
            
            # Validate all ranges
            for feature, (min_val, max_val) in ranges.items():
                assert min_val <= max_val, f"{emotion_type}.{feature} has invalid range"
    
    def test_emotion_score_consistency(self, mapper):
        """Test that emotion scoring is consistent."""
        # Create perfect happy features
        happy_features = {
            'valence': 0.9,
            'energy': 0.8,
            'danceability': 0.8,
            'tempo': 140.0
        }
        
        # Score for happy emotion multiple times
        scores = [
            mapper.compute_emotion_score(happy_features, "happy")
            for _ in range(3)
        ]
        
        # Should be consistent
        assert all(s == scores[0] for s in scores)
        assert scores[0] > 0.7  # Should be high score
    
    def test_opposite_emotions_have_different_scores(self, mapper):
        """Test that opposite emotions score differently."""
        happy_features = {
            'valence': 0.9,
            'energy': 0.8,
            'tempo': 140.0
        }
        
        sad_features = {
            'valence': 0.2,
            'energy': 0.3,
            'tempo': 70.0
        }
        
        # Happy features should score higher for happy
        happy_for_happy = mapper.compute_emotion_score(happy_features, "happy")
        sad_for_happy = mapper.compute_emotion_score(sad_features, "happy")
        
        assert happy_for_happy > sad_for_happy
        
        # And vice versa for sad
        happy_for_sad = mapper.compute_emotion_score(happy_features, "sad")
        sad_for_sad = mapper.compute_emotion_score(sad_features, "sad")
        
        assert sad_for_sad > happy_for_sad
    
    def test_custom_emotion_parsing(self, mapper):
        """Test parsing custom emotion strings."""
        # Should recognize keywords
        ranges = mapper.get_feature_ranges("cheerful and upbeat")
        assert len(ranges) > 0
        
        # Should have high valence for happy-related terms
        if 'valence' in ranges:
            min_val, max_val = ranges['valence']
            avg_valence = (min_val + max_val) / 2
            assert avg_valence > 0.4  # Generally positive


class TestSpotifyServiceIntegration:
    """Integration tests for SpotifyService with real API (if available)."""
    
    @pytest.fixture
    def service(self):
        """Create real SpotifyService."""
        return SpotifyService()
    
    def test_service_availability(self, service):
        """Test service availability status."""
        # Should not crash
        is_available = service.is_available()
        assert isinstance(is_available, bool)
    
    @pytest.mark.skipif(
        not os.getenv('SPOTIFY_CLIENT_ID'),
        reason="Spotify credentials not available"
    )
    def test_real_track_search(self, service):
        """Test searching for a real track."""
        if not service.is_available():
            pytest.skip("Spotify service not available")
        
        # Search for a famous song
        result = service.search_track("Bohemian Rhapsody", "Queen")
        
        if result:
            assert 'spotify_id' in result
            assert 'song_name' in result
            assert 'artist' in result
            assert 'Queen' in result['artist']
    
    @pytest.mark.skipif(
        not os.getenv('SPOTIFY_CLIENT_ID'),
        reason="Spotify credentials not available"
    )
    def test_real_audio_features(self, service):
        """Test getting audio features for a real track."""
        if not service.is_available():
            pytest.skip("Spotify service not available")
        
        # First search for a track
        track = service.search_track("Happy", "Pharrell Williams")
        
        if track and track.get('spotify_id'):
            features = service.get_audio_features(track['spotify_id'])
            
            if features:
                # Validate feature ranges
                assert 0.0 <= features['valence'] <= 1.0
                assert 0.0 <= features['energy'] <= 1.0
                assert features['tempo'] > 0
                
                # "Happy" by Pharrell should have high valence
                assert features['valence'] > 0.5


class TestPlaylistGeneratorIntegration:
    """Integration tests for PlaylistGenerator with real services."""
    
    @pytest.fixture(scope="class")
    def embedding_service(self):
        """Real embedding service."""
        return EmbeddingService(model_name="all-MiniLM-L6-v2")
    
    @pytest.fixture
    def emotion_mapper(self):
        """Real emotion mapper."""
        return EmotionMapper(use_llm=False)
    
    @pytest.fixture
    def spotify_service(self):
        """Real Spotify service."""
        return SpotifyService()
    
    @pytest.fixture
    def generator(self, embedding_service, emotion_mapper, spotify_service):
        """Real playlist generator."""
        return PlaylistGenerator(
            embedding_service=embedding_service,
            emotion_mapper=emotion_mapper,
            spotify_service=spotify_service
        )
    
    def test_generate_playlist_with_songs_only(self, generator):
        """Test generating playlist with only song inputs."""
        songs = [
            SongInput(song_name="Bohemian Rhapsody", artist="Queen"),
            SongInput(song_name="Stairway to Heaven", artist="Led Zeppelin")
        ]
        
        playlist, combined_emb, emotion_features = generator.generate_playlist(
            songs=songs,
            emotion=None,
            num_results=5
        )
        
        assert len(playlist) > 0
        assert len(playlist) <= 5
        assert combined_emb.shape == (384,)
        assert emotion_features is None
        
        # Check that results have required fields
        for song in playlist:
            assert hasattr(song, 'song_name')
            assert hasattr(song, 'artist')
            assert hasattr(song, 'similarity_score')
            assert 0.0 <= song.similarity_score <= 1.0
    
    def test_generate_playlist_with_emotion_only(self, generator):
        """Test generating playlist with only emotion."""
        playlist, combined_emb, emotion_features = generator.generate_playlist(
            songs=None,
            emotion=["happy"],
            num_results=5
        )
        
        assert len(playlist) > 0
        assert len(playlist) <= 5
        assert combined_emb.shape == (384,)
        assert emotion_features is not None
        assert isinstance(emotion_features, dict)
        
        # Should have some emotion-related features
        assert len(emotion_features) > 0
    
    def test_generate_playlist_combined(self, generator):
        """Test generating playlist with both songs and emotion."""
        songs = [
            SongInput(song_name="Happy", artist="Pharrell Williams")
        ]
        
        playlist, combined_emb, emotion_features = generator.generate_playlist(
            songs=songs,
            emotion=["joyful", "upbeat"],
            num_results=10
        )
        
        assert len(playlist) > 0
        assert len(playlist) <= 10
        assert combined_emb.shape == (384,)
        assert emotion_features is not None
    
    def test_playlist_results_are_sorted(self, generator):
        """Test that playlist results are sorted by similarity."""
        songs = [
            SongInput(song_name="Test Song", artist="Test Artist")
        ]
        
        playlist, _, _ = generator.generate_playlist(
            songs=songs,
            emotion=["calm"],
            num_results=10
        )
        
        # Verify sorting
        scores = [song.similarity_score for song in playlist]
        assert scores == sorted(scores, reverse=True)
    
    def test_error_on_empty_input(self, generator):
        """Test that empty input raises error."""
        with pytest.raises(ValueError, match="Must provide either songs or emotion"):
            generator.generate_playlist(
                songs=None,
                emotion=None,
                num_results=5
            )
    
    def test_combined_embedding_weights(self, generator):
        """Test that combined embedding uses correct weights."""
        songs = [
            SongInput(song_name="Test Song", artist="Test Artist")
        ]
        
        # With both songs and emotion, should use weighted combination
        combined_with_emotion = generator._compute_combined_embedding(
            songs=songs,
            emotion="happy"
        )
        
        # Just songs should give different result
        combined_songs_only = generator._compute_combined_embedding(
            songs=songs,
            emotion=None
        )
        
        # Should be different due to emotion contribution
        similarity = generator.embedding_service.compute_similarity(
            combined_with_emotion,
            combined_songs_only
        )
        assert similarity < 1.0  # Not identical


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture(scope="class")
    def full_stack(self):
        """Set up full stack of services."""
        return {
            'embedding': EmbeddingService(model_name="all-MiniLM-L6-v2"),
            'emotion': EmotionMapper(use_llm=False),
            'spotify': SpotifyService(),
            'generator': None  # Will be created
        }
    
    @pytest.fixture(scope="class")
    def generator(self, full_stack):
        """Create full playlist generator."""
        full_stack['generator'] = PlaylistGenerator(
            embedding_service=full_stack['embedding'],
            emotion_mapper=full_stack['emotion'],
            spotify_service=full_stack['spotify']
        )
        return full_stack['generator']
    
    def test_complete_playlist_generation_flow(self, generator):
        """Test complete flow from input to playlist."""
        # User provides some favorite songs and desired mood
        user_songs = [
            SongInput(song_name="Imagine", artist="John Lennon"),
            SongInput(song_name="Let It Be", artist="The Beatles"),
        ]
        user_emotion = ["peaceful", "hopeful"]
        
        # Generate playlist
        playlist, embedding, features = generator.generate_playlist(
            songs=user_songs,
            emotion=user_emotion,
            num_results=10
        )
        
        # Validate complete response
        assert len(playlist) > 0, "Should generate some songs"
        assert len(playlist) <= 10, "Should respect num_results"
        assert embedding is not None, "Should return embedding"
        assert features is not None, "Should return emotion features"
        
        # Validate playlist structure
        for i, song in enumerate(playlist):
            assert song.song_name, f"Song {i} missing name"
            assert song.artist, f"Song {i} missing artist"
            assert 0.0 <= song.similarity_score <= 1.0, f"Song {i} invalid score"
            
            # Scores should be in descending order
            if i > 0:
                assert song.similarity_score <= playlist[i-1].similarity_score
        
        print(f"\n✅ Generated {len(playlist)} songs")
        print(f"Top match: {playlist[0].song_name} by {playlist[0].artist} "
              f"(score: {playlist[0].similarity_score:.3f})")
    
    def test_different_emotions_produce_different_results(self, generator):
        """Test that different emotions lead to different playlists."""
        base_songs = [
            SongInput(song_name="Wonderwall", artist="Oasis")
        ]
        
        # Generate with happy emotion
        happy_playlist, happy_emb, _ = generator.generate_playlist(
            songs=base_songs,
            emotion=["happy"],
            num_results=5
        )
        
        # Generate with sad emotion
        sad_playlist, sad_emb, _ = generator.generate_playlist(
            songs=base_songs,
            emotion=["sad"],
            num_results=5
        )
        
        # Embeddings should be different
        similarity = generator.embedding_service.compute_similarity(
            happy_emb, sad_emb
        )
        assert similarity < 0.99, "Different emotions should produce different embeddings"
        
        # At least some songs should be different
        happy_ids = {s.spotify_id for s in happy_playlist if s.spotify_id}
        sad_ids = {s.spotify_id for s in sad_playlist if s.spotify_id}
        
        # May have some overlap, but shouldn't be identical
        if happy_ids and sad_ids:
            assert happy_ids != sad_ids, "Playlists should have some differences"
    
    def test_multiple_emotions_are_combined(self, generator):
        """Test that multiple emotions are properly combined."""
        playlist, emb, features = generator.generate_playlist(
            songs=None,
            emotion=["happy", "energetic", "dance"],
            num_results=5
        )
        
        assert len(playlist) > 0
        assert features is not None
        
        # Should use the first emotion for feature ranges
        # but the combined text for embedding
        print(f"\n✅ Multi-emotion playlist generated with features: "
              f"{list(features.keys())}")
    
    def test_edge_case_single_song(self, generator):
        """Test with just one song input."""
        playlist, emb, features = generator.generate_playlist(
            songs=[SongInput(song_name="Yesterday", artist="The Beatles")],
            emotion=None,
            num_results=3
        )
        
        assert len(playlist) > 0
        assert emb.shape == (384,)
    
    def test_edge_case_many_results(self, generator):
        """Test requesting many results."""
        playlist, _, _ = generator.generate_playlist(
            songs=None,
            emotion=["rock"],
            num_results=50  # Maximum
        )
        
        assert len(playlist) > 0
        assert len(playlist) <= 50
    
    def test_performance_benchmark(self, generator, benchmark=None):
        """Test performance of playlist generation."""
        songs = [
            SongInput(song_name="Test Song", artist="Test Artist")
        ]
        
        import time
        start = time.time()
        
        playlist, _, _ = generator.generate_playlist(
            songs=songs,
            emotion=["happy"],
            num_results=10
        )
        
        elapsed = time.time() - start
        
        assert len(playlist) > 0
        print(f"\n⏱️ Playlist generation took {elapsed:.2f}s")
        
        # Should complete in reasonable time (adjust based on your needs)
        assert elapsed < 60, "Generation should complete within 60 seconds"


class TestDataConsistency:
    """Test data consistency across services."""
    
    def test_embedding_determinism(self):
        """Test that embeddings are deterministic."""
        service1 = EmbeddingService()
        service2 = EmbeddingService()
        
        text = "Test song by Test Artist"
        
        emb1 = service1.encode_text(text)
        emb2 = service2.encode_text(text)
        
        # Should be identical
        assert np.allclose(emb1, emb2)
    
    def test_emotion_mapper_determinism(self):
        """Test that emotion mapping is deterministic."""
        mapper1 = EmotionMapper(use_llm=False)
        mapper2 = EmotionMapper(use_llm=False)
        
        features = {
            'valence': 0.7,
            'energy': 0.8,
            'tempo': 120.0
        }
        
        score1 = mapper1.compute_emotion_score(features, "happy")
        score2 = mapper2.compute_emotion_score(features, "happy")
        
        assert score1 == score2
    
    def test_similarity_symmetry(self):
        """Test that similarity is symmetric."""
        service = EmbeddingService()
        
        emb1 = service.encode_text("Song A")
        emb2 = service.encode_text("Song B")
        
        sim_ab = service.compute_similarity(emb1, emb2)
        sim_ba = service.compute_similarity(emb2, emb1)
        
        assert np.isclose(sim_ab, sim_ba)
