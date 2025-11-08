"""Tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError
from backend.models.schemas import (
    EmotionType,
    SongInput,
    PlaylistRequest,
    AudioFeatures,
    SongResult,
    MoodCollage,
    PlaylistResponse,
    HealthResponse,
    SpotifyTrackInfo
)


class TestEmotionType:
    """Test suite for EmotionType enum."""
    
    def test_emotion_type_values(self):
        """Test that all emotion types have correct values."""
        assert EmotionType.HAPPY.value == "happy"
        assert EmotionType.SAD.value == "sad"
        assert EmotionType.ENERGETIC.value == "energetic"
        assert EmotionType.CALM.value == "calm"
        assert EmotionType.ANGRY.value == "angry"
        assert EmotionType.MELANCHOLIC.value == "melancholic"
        assert EmotionType.HOPEFUL.value == "hopeful"
        assert EmotionType.ROMANTIC.value == "romantic"
        assert EmotionType.ANXIOUS.value == "anxious"
        assert EmotionType.PEACEFUL.value == "peaceful"
    
    def test_emotion_type_count(self):
        """Test that we have expected number of emotion types."""
        emotions = list(EmotionType)
        assert len(emotions) == 10


class TestSongInput:
    """Test suite for SongInput schema."""
    
    def test_song_input_valid(self):
        """Test creating a valid SongInput."""
        song = SongInput(
            song_name="Bohemian Rhapsody",
            artist="Queen"
        )
        
        assert song.song_name == "Bohemian Rhapsody"
        assert song.artist == "Queen"
        assert song.spotify_id is None
    
    def test_song_input_with_spotify_id(self):
        """Test SongInput with spotify_id."""
        song = SongInput(
            song_name="Imagine",
            artist="John Lennon",
            spotify_id="track123"
        )
        
        assert song.spotify_id == "track123"
    
    def test_song_input_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            SongInput(song_name="Test Song")  # Missing artist
        
        with pytest.raises(ValidationError):
            SongInput(artist="Test Artist")  # Missing song_name
    
    def test_song_input_empty_strings(self):
        """Test SongInput with empty strings."""
        song = SongInput(song_name="", artist="")
        
        # Should allow empty strings (validation happens elsewhere)
        assert song.song_name == ""
        assert song.artist == ""


class TestPlaylistRequest:
    """Test suite for PlaylistRequest schema."""
    
    def test_playlist_request_with_songs(self):
        """Test PlaylistRequest with songs."""
        request = PlaylistRequest(
            songs=[
                SongInput(song_name="Song 1", artist="Artist 1"),
                SongInput(song_name="Song 2", artist="Artist 2")
            ],
            num_results=10
        )
        
        assert len(request.songs) == 2
        assert request.emotion is None
        assert request.num_results == 10
    
    def test_playlist_request_with_emotion(self):
        """Test PlaylistRequest with emotion."""
        request = PlaylistRequest(
            emotion=["happy"],
            num_results=15
        )
        
        assert request.emotion == ["happy"]
        assert request.songs is None
        assert request.num_results == 15
    
    def test_playlist_request_defaults(self):
        """Test PlaylistRequest default values."""
        request = PlaylistRequest(emotion=["calm"])
        
        assert request.num_results == 10
        assert request.include_collage is True
        assert request.enrich_with_lyrics is False
    
    def test_playlist_request_num_results_validation(self):
        """Test num_results validation."""
        # Valid values
        request = PlaylistRequest(emotion=["happy"], num_results=1)
        assert request.num_results == 1
        
        request = PlaylistRequest(emotion=["happy"], num_results=50)
        assert request.num_results == 50
        
        # Invalid values
        with pytest.raises(ValidationError):
            PlaylistRequest(emotion=["happy"], num_results=0)
        
        with pytest.raises(ValidationError):
            PlaylistRequest(emotion=["happy"], num_results=51)
        
        with pytest.raises(ValidationError):
            PlaylistRequest(emotion=["happy"], num_results=-1)
    
    def test_playlist_request_with_both_songs_and_emotion(self):
        """Test PlaylistRequest with both songs and emotion."""
        request = PlaylistRequest(
            songs=[SongInput(song_name="Test", artist="Artist")],
            emotion=["energetic"],
            num_results=20
        )
        
        assert len(request.songs) == 1
        assert request.emotion == ["energetic"]
    
    def test_playlist_request_multiple_emotions(self):
        """Test PlaylistRequest with multiple emotions."""
        request = PlaylistRequest(
            emotion=["happy", "energetic", "upbeat"],
            num_results=10
        )
        
        assert len(request.emotion) == 3
    
    def test_playlist_request_boolean_flags(self):
        """Test boolean flags in PlaylistRequest."""
        request = PlaylistRequest(
            emotion=["romantic"],
            include_collage=False,
            enrich_with_lyrics=True
        )
        
        assert request.include_collage is False
        assert request.enrich_with_lyrics is True


class TestAudioFeatures:
    """Test suite for AudioFeatures schema."""
    
    def test_audio_features_valid(self):
        """Test creating valid AudioFeatures."""
        features = AudioFeatures(
            valence=0.7,
            energy=0.8,
            danceability=0.6,
            tempo=120.0,
            acousticness=0.3,
            instrumentalness=0.1,
            liveness=0.2,
            speechiness=0.1
        )
        
        assert features.valence == 0.7
        assert features.energy == 0.8
        assert features.tempo == 120.0
    
    def test_audio_features_range_validation(self):
        """Test that audio features are validated to be in [0, 1] range."""
        # Valid values
        features = AudioFeatures(
            valence=0.0,
            energy=1.0,
            danceability=0.5,
            tempo=100.0,
            acousticness=0.0,
            instrumentalness=1.0,
            liveness=0.0,
            speechiness=1.0
        )
        
        assert features.valence == 0.0
        assert features.energy == 1.0
        
        # Invalid values (out of range)
        with pytest.raises(ValidationError):
            AudioFeatures(
                valence=1.5,  # Over 1.0
                energy=0.8,
                danceability=0.6,
                tempo=120.0,
                acousticness=0.3,
                instrumentalness=0.1,
                liveness=0.2,
                speechiness=0.1
            )
        
        with pytest.raises(ValidationError):
            AudioFeatures(
                valence=0.7,
                energy=-0.1,  # Below 0.0
                danceability=0.6,
                tempo=120.0,
                acousticness=0.3,
                instrumentalness=0.1,
                liveness=0.2,
                speechiness=0.1
            )
    
    def test_audio_features_tempo_positive(self):
        """Test that tempo can be any positive value."""
        features = AudioFeatures(
            valence=0.5,
            energy=0.5,
            danceability=0.5,
            tempo=200.0,  # High tempo
            acousticness=0.5,
            instrumentalness=0.5,
            liveness=0.5,
            speechiness=0.5
        )
        
        assert features.tempo == 200.0


class TestSongResult:
    """Test suite for SongResult schema."""
    
    def test_song_result_minimal(self):
        """Test SongResult with minimal required fields."""
        song = SongResult(
            song_name="Test Song",
            artist="Test Artist",
            similarity_score=0.85
        )
        
        assert song.song_name == "Test Song"
        assert song.artist == "Test Artist"
        assert song.similarity_score == 0.85
    
    def test_song_result_with_all_fields(self):
        """Test SongResult with all fields."""
        audio_features = AudioFeatures(
            valence=0.7,
            energy=0.8,
            danceability=0.6,
            tempo=120.0,
            acousticness=0.3,
            instrumentalness=0.1,
            liveness=0.2,
            speechiness=0.1
        )
        
        song = SongResult(
            song_name="Complete Song",
            artist="Complete Artist",
            spotify_id="track123",
            similarity_score=0.92,
            audio_features=audio_features,
            album="Test Album",
            preview_url="https://preview.url",
            external_url="https://spotify.url",
            duration_ms=240000,
            popularity=85,
            album_image="https://image.url",
            genius_url="https://genius.url"
        )
        
        assert song.spotify_id == "track123"
        assert song.album == "Test Album"
        assert song.duration_ms == 240000
        assert song.popularity == 85
        assert song.genius_url == "https://genius.url"
    
    def test_song_result_similarity_score_validation(self):
        """Test similarity_score is a valid float."""
        song = SongResult(
            song_name="Test",
            artist="Artist",
            similarity_score=0.0
        )
        assert song.similarity_score == 0.0
        
        song = SongResult(
            song_name="Test",
            artist="Artist",
            similarity_score=1.0
        )
        assert song.similarity_score == 1.0


class TestMoodCollage:
    """Test suite for MoodCollage schema."""
    
    def test_mood_collage_valid(self):
        """Test creating a valid MoodCollage."""
        collage = MoodCollage(
            image_base64="base64encodedstring",
            dominant_colors=["#FF5733", "#33FF57", "#3357FF"],
            visual_params={"param1": "value1", "param2": 42},
            width=800,
            height=600
        )
        
        assert collage.image_base64 == "base64encodedstring"
        assert len(collage.dominant_colors) == 3
        assert collage.width == 800
        assert collage.height == 600
    
    def test_mood_collage_empty_colors(self):
        """Test MoodCollage with empty colors list."""
        collage = MoodCollage(
            image_base64="data",
            dominant_colors=[],
            visual_params={},
            width=800,
            height=600
        )
        
        assert len(collage.dominant_colors) == 0


class TestPlaylistResponse:
    """Test suite for PlaylistResponse schema."""
    
    def test_playlist_response_minimal(self):
        """Test PlaylistResponse with minimal fields."""
        response = PlaylistResponse(
            playlist=[
                SongResult(
                    song_name="Song 1",
                    artist="Artist 1",
                    similarity_score=0.9
                )
            ]
        )
        
        assert len(response.playlist) == 1
        assert response.mood_collage is None
        assert response.emotion_features is None
    
    def test_playlist_response_with_all_fields(self):
        """Test PlaylistResponse with all fields."""
        songs = [
            SongResult(
                song_name=f"Song {i}",
                artist=f"Artist {i}",
                similarity_score=0.9 - i * 0.1
            )
            for i in range(5)
        ]
        
        collage = MoodCollage(
            image_base64="base64string",
            dominant_colors=["#FF5733"],
            visual_params={"test": "value"},
            width=800,
            height=600
        )
        
        response = PlaylistResponse(
            playlist=songs,
            mood_collage=collage,
            emotion_features={"valence": 0.7, "energy": 0.8},
            combined_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        assert len(response.playlist) == 5
        assert response.mood_collage is not None
        assert response.emotion_features is not None
        assert len(response.combined_embedding) == 5
    
    def test_playlist_response_empty_playlist(self):
        """Test PlaylistResponse with empty playlist."""
        response = PlaylistResponse(playlist=[])
        
        assert len(response.playlist) == 0


class TestHealthResponse:
    """Test suite for HealthResponse schema."""
    
    def test_health_response_healthy(self):
        """Test HealthResponse with healthy status."""
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            services={
                "embedding_service": True,
                "emotion_mapper": True,
                "spotify_service": True
            }
        )
        
        assert response.status == "healthy"
        assert response.version == "0.1.0"
        assert response.services["spotify_service"] is True
    
    def test_health_response_degraded(self):
        """Test HealthResponse with degraded status."""
        response = HealthResponse(
            status="degraded",
            version="0.1.0",
            services={
                "embedding_service": True,
                "emotion_mapper": True,
                "spotify_service": False
            }
        )
        
        assert response.status == "degraded"
        assert response.services["spotify_service"] is False


class TestSpotifyTrackInfo:
    """Test suite for SpotifyTrackInfo schema."""
    
    def test_spotify_track_info_complete(self):
        """Test SpotifyTrackInfo with all fields."""
        track = SpotifyTrackInfo(
            spotify_id="track123",
            song_name="Test Song",
            artist="Test Artist",
            album="Test Album",
            preview_url="https://preview.url",
            external_url="https://spotify.url",
            duration_ms=240000,
            popularity=85,
            album_image="https://image.url"
        )
        
        assert track.spotify_id == "track123"
        assert track.duration_ms == 240000
        assert track.popularity == 85
    
    def test_spotify_track_info_without_optional_fields(self):
        """Test SpotifyTrackInfo without optional fields."""
        track = SpotifyTrackInfo(
            spotify_id="track123",
            song_name="Test Song",
            artist="Test Artist",
            album="Test Album",
            external_url="https://spotify.url",
            duration_ms=240000,
            popularity=85
        )
        
        assert track.preview_url is None
        assert track.album_image is None


class TestSchemaValidation:
    """Test suite for general schema validation."""
    
    def test_nested_schema_validation(self):
        """Test validation of nested schemas."""
        # Valid nested structure
        response = PlaylistResponse(
            playlist=[
                SongResult(
                    song_name="Test",
                    artist="Artist",
                    similarity_score=0.9,
                    audio_features=AudioFeatures(
                        valence=0.7,
                        energy=0.8,
                        danceability=0.6,
                        tempo=120.0,
                        acousticness=0.3,
                        instrumentalness=0.1,
                        liveness=0.2,
                        speechiness=0.1
                    )
                )
            ]
        )
        
        assert response.playlist[0].audio_features.valence == 0.7
    
    def test_json_serialization(self):
        """Test that schemas can be serialized to JSON."""
        song = SongResult(
            song_name="Test Song",
            artist="Test Artist",
            similarity_score=0.85
        )
        
        json_data = song.model_dump()
        
        assert isinstance(json_data, dict)
        assert json_data["song_name"] == "Test Song"
        assert json_data["similarity_score"] == 0.85
    
    def test_schema_with_none_values(self):
        """Test schemas with None for optional fields."""
        song = SongResult(
            song_name="Test",
            artist="Artist",
            similarity_score=0.9,
            spotify_id=None,
            audio_features=None,
            album=None
        )
        
        assert song.spotify_id is None
        assert song.audio_features is None
        assert song.album is None
