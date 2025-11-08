"""Shared fixtures for pytest tests."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient

from backend.services.embedding_service import EmbeddingService
from backend.services.emotion_mapper import EmotionMapper
from backend.services.spotify_service import SpotifyService
from backend.services.playlist_generator import PlaylistGenerator
from backend.models.schemas import SongInput, AudioFeatures


@pytest.fixture
def embedding_service():
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def emotion_mapper():
    return EmotionMapper(use_llm=False)  # Disable LLM for faster tests


@pytest.fixture
def mock_spotify_service():
    """Create a mocked SpotifyService for testing."""
    mock_service = Mock(spec=SpotifyService)
    mock_service.is_available.return_value = True
    
    # Mock search_track
    mock_service.search_track.return_value = {
        'spotify_id': 'test_track_id',
        'song_name': 'Test Song',
        'artist': 'Test Artist',
        'album': 'Test Album',
        'preview_url': 'https://test.com/preview',
        'external_url': 'https://spotify.com/track/test',
        'duration_ms': 240000,
        'popularity': 75,
        'album_image': 'https://test.com/image.jpg'
    }
    
    # Mock get_audio_features
    mock_service.get_audio_features.return_value = {
        'valence': 0.7,
        'energy': 0.8,
        'danceability': 0.6,
        'tempo': 120.0,
        'acousticness': 0.3,
        'instrumentalness': 0.1,
        'liveness': 0.2,
        'speechiness': 0.1
    }
    
    # Mock search_by_multiple_queries
    mock_service.search_by_multiple_queries.return_value = [
        {
            'spotify_id': f'track_{i}',
            'song_name': f'Song {i}',
            'artist': f'Artist {i}',
            'album': f'Album {i}',
            'preview_url': f'https://test.com/preview/{i}',
            'external_url': f'https://spotify.com/track/{i}',
            'duration_ms': 200000 + i * 1000,
            'popularity': 70 + i,
            'album_image': f'https://test.com/image{i}.jpg'
        }
        for i in range(10)
    ]
    
    # Mock get_tracks_with_features
    mock_service.get_tracks_with_features.return_value = [
        {
            'spotify_id': f'track_{i}',
            'song_name': f'Song {i}',
            'artist': f'Artist {i}',
            'album': f'Album {i}',
            'preview_url': f'https://test.com/preview/{i}',
            'external_url': f'https://spotify.com/track/{i}',
            'duration_ms': 200000 + i * 1000,
            'popularity': 70 + i,
            'album_image': f'https://test.com/image{i}.jpg',
            'audio_features': {
                'valence': 0.5 + i * 0.05,
                'energy': 0.6 + i * 0.03,
                'danceability': 0.5 + i * 0.04,
                'tempo': 110.0 + i * 5,
                'acousticness': 0.3,
                'instrumentalness': 0.1,
                'liveness': 0.2,
                'speechiness': 0.1
            }
        }
        for i in range(10)
    ]
    
    return mock_service


@pytest.fixture
def playlist_generator(embedding_service, emotion_mapper, mock_spotify_service):
    """Create a PlaylistGenerator instance for testing."""
    return PlaylistGenerator(
        embedding_service=embedding_service,
        emotion_mapper=emotion_mapper,
        spotify_service=mock_spotify_service
    )


@pytest.fixture
def sample_song_inputs():
    """Create sample SongInput objects for testing."""
    return [
        SongInput(song_name="Bohemian Rhapsody", artist="Queen"),
        SongInput(song_name="Imagine", artist="John Lennon"),
        SongInput(song_name="Hotel California", artist="Eagles")
    ]


@pytest.fixture
def sample_audio_features():
    """Create sample AudioFeatures for testing."""
    return AudioFeatures(
        valence=0.7,
        energy=0.8,
        danceability=0.6,
        tempo=120.0,
        acousticness=0.3,
        instrumentalness=0.1,
        liveness=0.2,
        speechiness=0.1
    )


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return {
        'song_embedding': np.random.randn(384).astype(np.float32),
        'emotion_embedding': np.random.randn(384).astype(np.float32),
        'combined_embedding': np.random.randn(384).astype(np.float32)
    }


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient with properly initialized app state."""
    from backend.main import app
    
    # Initialize app state manually for testing (bypassing lifespan)
    # This avoids the async context manager in TestClient
    app.state.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    app.state.emotion_mapper = EmotionMapper(use_llm=False)
    app.state.spotify_service = SpotifyService()
    
    # Mock genius service (optional)
    from unittest.mock import Mock
    app.state.genius_service = Mock()
    app.state.genius_service.is_available.return_value = False
    
    client = TestClient(app)
    return client


@pytest.fixture
def test_client_with_mock_services():
    """Create a FastAPI TestClient with mocked services for faster tests."""
    from backend.main import app
    
    # Create mock services
    mock_embedding = Mock(spec=EmbeddingService)
    mock_embedding.encode_song.return_value = np.random.randn(384).astype(np.float32)
    mock_embedding.encode_emotion.return_value = np.random.randn(384).astype(np.float32)
    mock_embedding.combine_embeddings.return_value = np.random.randn(384).astype(np.float32)
    mock_embedding.compute_similarity.return_value = 0.85
    
    mock_emotion = Mock(spec=EmotionMapper)
    mock_emotion.get_feature_ranges.return_value = {
        'valence': (0.6, 1.0),
        'energy': (0.5, 1.0)
    }
    
    mock_spotify = Mock(spec=SpotifyService)
    mock_spotify.is_available.return_value = True
    mock_spotify.search_track.return_value = {
        'spotify_id': 'test_id',
        'song_name': 'Test Song',
        'artist': 'Test Artist',
        'album': 'Test Album',
        'external_url': 'https://test.url',
        'duration_ms': 200000,
        'popularity': 80,
        'album_image': 'https://test.image'
    }
    mock_spotify.get_audio_features.return_value = {
        'valence': 0.7,
        'energy': 0.8,
        'danceability': 0.6,
        'tempo': 120.0,
        'acousticness': 0.3,
        'instrumentalness': 0.1,
        'liveness': 0.2,
        'speechiness': 0.1
    }
    
    mock_genius = Mock()
    mock_genius.is_available.return_value = False
    
    # Set mocked services on app state
    app.state.embedding_service = mock_embedding
    app.state.emotion_mapper = mock_emotion
    app.state.spotify_service = mock_spotify
    app.state.genius_service = mock_genius
    
    client = TestClient(app)
    return client
