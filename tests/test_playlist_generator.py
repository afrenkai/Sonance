"""Tests for PlaylistGenerator."""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from backend.services.playlist_generator import PlaylistGenerator
from backend.models.schemas import SongInput, SongResult


class TestPlaylistGenerator:
    
    def test_initialization(self, playlist_generator):
        """Test PlaylistGenerator initialization."""
        assert playlist_generator is not None
        assert playlist_generator.embedding_service is not None
        assert playlist_generator.emotion_mapper is not None
        assert playlist_generator.spotify_service is not None
    
    def test_generate_playlist_with_songs_only(
        self, playlist_generator, sample_song_inputs
    ):
        playlist, combined_emb, emotion_features = playlist_generator.generate_playlist(
            songs=sample_song_inputs,
            emotion=None,
            num_results=10
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
        assert len(playlist) <= 10
        assert isinstance(combined_emb, np.ndarray)
        assert emotion_features is None
    
    def test_generate_playlist_with_emotion_only(self, playlist_generator):
        playlist, combined_emb, emotion_features = playlist_generator.generate_playlist(
            songs=None,
            emotion=["happy"],
            num_results=10
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
        assert isinstance(combined_emb, np.ndarray)
        assert emotion_features is not None
        assert isinstance(emotion_features, dict)
    
    def test_generate_playlist_with_songs_and_emotion(
        self, playlist_generator, sample_song_inputs
    ):
        playlist, combined_emb, emotion_features = playlist_generator.generate_playlist(
            songs=sample_song_inputs,
            emotion=["energetic"],
            num_results=15
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
        assert len(playlist) <= 15
        assert isinstance(combined_emb, np.ndarray)
        assert emotion_features is not None
    
    def test_generate_playlist_without_inputs(self, playlist_generator):
        with pytest.raises(ValueError, match="Must provide either songs, artists, or emotion"):
            playlist_generator.generate_playlist(
                songs=None,
                emotion=None,
                num_results=10
            )
    
    def test_generate_playlist_result_structure(
        self, playlist_generator, sample_song_inputs
    ):
        playlist, _, _ = playlist_generator.generate_playlist(
            songs=sample_song_inputs,
            emotion=["happy"],
            num_results=5
        )
        
        for song in playlist:
            assert isinstance(song, SongResult)
            assert hasattr(song, 'song_name')
            assert hasattr(song, 'artist')
            assert hasattr(song, 'similarity_score')
            assert 0.0 <= song.similarity_score <= 1.0
    
    def test_compute_combined_embedding_songs_only(
        self, playlist_generator, sample_song_inputs
    ):
        combined = playlist_generator._compute_combined_embedding(
            songs=sample_song_inputs,
            emotion=None
        )
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (384,)
        assert np.isclose(np.linalg.norm(combined), 1.0, atol=1e-5)
    
    def test_compute_combined_embedding_emotion_only(self, playlist_generator):
        combined = playlist_generator._compute_combined_embedding(
            songs=None,
            emotion="melancholic"
        )
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (384,)
        assert np.isclose(np.linalg.norm(combined), 1.0, atol=1e-5)
    
    def test_compute_combined_embedding_both(
        self, playlist_generator, sample_song_inputs
    ):
        combined = playlist_generator._compute_combined_embedding(
            songs=sample_song_inputs,
            emotion="romantic"
        )
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (384,)
        assert np.isclose(np.linalg.norm(combined), 1.0, atol=1e-5)
    
    def test_query_songs_with_spotify(
        self, playlist_generator, sample_song_inputs
    ):
        query_embedding = np.random.randn(384).astype(np.float32)
        
        playlist = playlist_generator._query_songs_with_spotify(
            songs=sample_song_inputs,
            query_embedding=query_embedding,
            emotion="happy",
            emotion_features={"valence": (0.6, 1.0), "energy": (0.5, 1.0)},
            num_results=10
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
        assert len(playlist) <= 10
    
    def test_query_songs_with_spotify_no_seeds(self, playlist_generator):
        query_embedding = np.random.randn(384).astype(np.float32)
        
        playlist = playlist_generator._query_songs_with_spotify(
            songs=None,
            query_embedding=query_embedding,
            emotion="calm",
            emotion_features={"valence": (0.3, 0.7), "energy": (0.0, 0.4)},
            num_results=10
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
    
    def test_get_emotion_keywords(self, playlist_generator):
        keywords = playlist_generator._get_emotion_keywords("happy")
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert len(keywords) <= 5
        assert "happy" in keywords
    
    def test_get_emotion_keywords_unknown(self, playlist_generator):
        keywords = playlist_generator._get_emotion_keywords("custom_emotion_xyz")
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "custom_emotion_xyz" in keywords
    
    def test_generate_mock_results(self, playlist_generator):
        mock_results = playlist_generator._generate_mock_results(5)
        
        assert isinstance(mock_results, list)
        assert len(mock_results) == 5
        
        for result in mock_results:
            assert isinstance(result, SongResult)
            assert result.song_name
            assert result.artist
            assert 0.0 <= result.similarity_score <= 1.0
    
    def test_playlist_sorted_by_similarity(
        self, playlist_generator, sample_song_inputs
    ):
        playlist, _, _ = playlist_generator.generate_playlist(
            songs=sample_song_inputs,
            emotion=["happy"],
            num_results=10
        )
        
        scores = [song.similarity_score for song in playlist]
        assert scores == sorted(scores, reverse=True)
    
    def test_playlist_deduplication(self, playlist_generator, sample_song_inputs):
        playlist, _, _ = playlist_generator.generate_playlist(
            songs=sample_song_inputs,
            emotion=["energetic"],
            num_results=20
        )
        
        spotify_ids = [song.spotify_id for song in playlist if song.spotify_id]
        assert len(spotify_ids) == len(set(spotify_ids))
    
    def test_multiple_emotions(self, playlist_generator, sample_song_inputs):
        playlist, _, emotion_features = playlist_generator.generate_playlist(
            songs=sample_song_inputs,
            emotion=["happy", "energetic", "upbeat"],
            num_results=10
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
        assert emotion_features is not None
    
    def test_num_results_respected(self, playlist_generator, sample_song_inputs):
        for num_results in [5, 10, 20]:
            playlist, _, _ = playlist_generator.generate_playlist(
                songs=sample_song_inputs,
                emotion=["happy"],
                num_results=num_results
            )
            
            assert len(playlist) <= num_results
    
    def test_enrich_with_lyrics_flag(self, playlist_generator, sample_song_inputs):
        playlist, _, _ = playlist_generator.generate_playlist(
            songs=sample_song_inputs,
            emotion=["romantic"],
            num_results=5,
            enrich_with_lyrics=True
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
    
    def test_emotion_features_ranges(self, playlist_generator):
        _, _, emotion_features = playlist_generator.generate_playlist(
            songs=None,
            emotion=["sad"],
            num_results=5
        )
        
        assert emotion_features is not None
        for feature, value in emotion_features.items():
            if isinstance(value, tuple):
                min_val, max_val = value
                assert min_val <= max_val
    
    def test_has_llm_emotions_property(self, playlist_generator):
        assert isinstance(playlist_generator.has_llm_emotions, bool)
    
    def test_spotify_service_availability_check(
        self, embedding_service, emotion_mapper
    ):
        mock_spotify = Mock()
        mock_spotify.is_available.return_value = False
        
        generator = PlaylistGenerator(
            embedding_service=embedding_service,
            emotion_mapper=emotion_mapper,
            spotify_service=mock_spotify
        )
        
        playlist, _, _ = generator.generate_playlist(
            songs=None,
            emotion=["happy"],
            num_results=5
        )
        
        assert isinstance(playlist, list)
        assert len(playlist) > 0
