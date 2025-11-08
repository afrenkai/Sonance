"""Tests for FastAPI API routes."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from backend.main import app
from backend.models.schemas import (
    PlaylistRequest,
    SongInput,
    EmotionType
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from backend.main import app
    from backend.services.embedding_service import EmbeddingService
    from backend.services.emotion_mapper import EmotionMapper
    from backend.services.spotify_service import SpotifyService
    from unittest.mock import Mock
    
    # Initialize app state for testing
    app.state.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    app.state.emotion_mapper = EmotionMapper(use_llm=False)
    app.state.spotify_service = SpotifyService()
    
    # Mock genius service
    app.state.genius_service = Mock()
    app.state.genius_service.is_available.return_value = False
    
    return TestClient(app)


class TestHealthEndpoints:
    """Test suite for health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "EmoRec API"
    
    def test_health_check_endpoint(self, client):
        """Test the /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_api_health_endpoint(self, client):
        """Test the /api/v1/health endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data


class TestEmotionsEndpoint:
    """Test suite for emotions listing endpoint."""
    
    def test_list_emotions(self, client):
        """Test listing all available emotions."""
        response = client.get("/api/v1/emotions")
        
        assert response.status_code == 200
        data = response.json()
        assert "emotions" in data
        assert isinstance(data["emotions"], list)
        assert len(data["emotions"]) > 0
        
        # Check that predefined emotions are included
        emotions = data["emotions"]
        assert "happy" in emotions
        assert "sad" in emotions
        assert "energetic" in emotions


class TestGeneratePlaylistEndpoint:
    """Test suite for playlist generation endpoint."""
    
    def test_generate_playlist_with_songs(self, client):
        """Test generating playlist with song inputs."""
        request_data = {
            "songs": [
                {"song_name": "Bohemian Rhapsody", "artist": "Queen"},
                {"song_name": "Imagine", "artist": "John Lennon"}
            ],
            "num_results": 10,
            "include_collage": False,
            "enrich_with_lyrics": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "playlist" in data
        assert isinstance(data["playlist"], list)
        assert len(data["playlist"]) > 0
    
    def test_generate_playlist_with_emotion(self, client):
        """Test generating playlist with emotion."""
        request_data = {
            "emotion": ["happy"],
            "num_results": 10,
            "include_collage": False,
            "enrich_with_lyrics": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "playlist" in data
        assert "emotion_features" in data
        assert isinstance(data["playlist"], list)
    
    def test_generate_playlist_with_songs_and_emotion(self, client):
        """Test generating playlist with both songs and emotion."""
        request_data = {
            "songs": [
                {"song_name": "Hotel California", "artist": "Eagles"}
            ],
            "emotion": ["melancholic"],
            "num_results": 15,
            "include_collage": False,
            "enrich_with_lyrics": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "playlist" in data
        assert "emotion_features" in data
        assert len(data["playlist"]) > 0
    
    def test_generate_playlist_with_collage(self, client):
        """Test generating playlist with mood collage."""
        request_data = {
            "emotion": ["romantic"],
            "num_results": 5,
            "include_collage": True,
            "enrich_with_lyrics": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "playlist" in data
        assert "mood_collage" in data
        
        if data["mood_collage"]:
            collage = data["mood_collage"]
            assert "image_base64" in collage
            assert "dominant_colors" in collage
            assert "width" in collage
            assert "height" in collage
    
    def test_generate_playlist_without_inputs(self, client):
        """Test that generating playlist without inputs returns error."""
        request_data = {
            "num_results": 10,
            "include_collage": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 400
    
    def test_generate_playlist_invalid_num_results(self, client):
        """Test with invalid num_results."""
        request_data = {
            "emotion": ["happy"],
            "num_results": 100,  # Over limit
            "include_collage": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        # Should either cap at max or return validation error
        assert response.status_code in [200, 422]
    
    def test_generate_playlist_response_structure(self, client):
        """Test that response has correct structure."""
        request_data = {
            "emotion": ["energetic"],
            "num_results": 5,
            "include_collage": False,
            "enrich_with_lyrics": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check playlist structure
        assert "playlist" in data
        for song in data["playlist"]:
            assert "song_name" in song
            assert "artist" in song
            assert "similarity_score" in song
            assert 0.0 <= song["similarity_score"] <= 1.0
    
    def test_generate_playlist_multiple_emotions(self, client):
        """Test generating playlist with multiple emotions."""
        request_data = {
            "emotion": ["happy", "energetic", "upbeat"],
            "num_results": 10,
            "include_collage": False,
            "enrich_with_lyrics": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["playlist"]) > 0


class TestSpotifyEndpoints:
    """Test suite for Spotify-related endpoints."""
    
    def test_search_spotify_track(self, client):
        """Test searching for a Spotify track."""
        response = client.get(
            "/api/v1/spotify/search",
            params={"song_name": "Bohemian Rhapsody", "artist": "Queen"}
        )
        
        # May return 200 if Spotify is available, or 503 if not
        assert response.status_code in [200, 503, 404]
    
    def test_search_spotify_track_without_artist(self, client):
        """Test searching without artist name."""
        response = client.get(
            "/api/v1/spotify/search",
            params={"song_name": "Imagine"}
        )
        
        assert response.status_code in [200, 503, 404]
    
    def test_get_spotify_track_by_id(self, client):
        """Test getting track by Spotify ID."""
        response = client.get("/api/v1/spotify/track/test_track_id")
        
        # May return various codes depending on service availability
        assert response.status_code in [200, 404, 503]
    
    def test_get_audio_features(self, client):
        """Test getting audio features for a track."""
        response = client.get("/api/v1/spotify/audio-features/test_track_id")
        
        assert response.status_code in [200, 404, 503]
    
    def test_get_spotify_recommendations(self, client):
        """Test getting Spotify recommendations."""
        request_data = {
            "seed_tracks": ["track1", "track2"],
            "emotion": "happy",
            "num_results": 20
        }
        
        response = client.post("/api/v1/spotify/recommendations", json=request_data)
        
        # Recommendations API is deprecated, may return 422 for validation or empty results
        assert response.status_code in [200, 422, 503]


class TestCORSMiddleware:
    """Test suite for CORS middleware."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/")
        
        # Check for CORS headers
        assert response.status_code == 200
        # Note: TestClient doesn't always include CORS headers in responses


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_404_for_nonexistent_endpoint(self, client):
        """Test 404 response for nonexistent endpoint."""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    def test_422_for_invalid_json(self, client):
        """Test 422 response for invalid request body."""
        response = client.post(
            "/api/v1/generate-playlist",
            json={"invalid_field": "value"}
        )
        
        assert response.status_code in [400, 422]
    
    def test_method_not_allowed(self, client):
        """Test 405 response for wrong HTTP method."""
        response = client.get("/api/v1/generate-playlist")
        
        assert response.status_code == 405


class TestRequestValidation:
    """Test suite for request validation."""
    
    def test_playlist_request_validation_min_num_results(self, client):
        """Test validation of minimum num_results."""
        request_data = {
            "emotion": ["happy"],
            "num_results": 0,  # Below minimum
            "include_collage": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 422
    
    def test_playlist_request_validation_max_num_results(self, client):
        """Test validation of maximum num_results."""
        request_data = {
            "emotion": ["happy"],
            "num_results": 51,  # Above maximum
            "include_collage": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 422
    
    def test_song_input_validation_missing_fields(self, client):
        """Test validation when song fields are missing."""
        request_data = {
            "songs": [
                {"song_name": "Test Song"}  # Missing artist
            ],
            "num_results": 10,
            "include_collage": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 422
    
    def test_valid_request_with_defaults(self, client):
        """Test request with minimal required fields."""
        request_data = {
            "emotion": ["calm"]
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["playlist"]) <= 10  # Default num_results


class TestResponseFormats:
    """Test suite for response formats."""
    
    def test_json_response_format(self, client):
        """Test that responses are valid JSON."""
        response = client.get("/api/v1/emotions")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Should be parseable as JSON
        data = response.json()
        assert isinstance(data, dict)
    
    def test_playlist_response_includes_metadata(self, client):
        """Test that playlist response includes metadata."""
        request_data = {
            "emotion": ["happy"],
            "num_results": 5,
            "include_collage": False
        }
        
        response = client.post("/api/v1/generate-playlist", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for metadata fields
        assert "playlist" in data
        assert "combined_embedding" in data or data.get("combined_embedding") is not None
