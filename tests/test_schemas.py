import pytest
from pydantic import ValidationError
from backend.models.schemas import (
    EmotionType,
    SongInput,
    PlaylistRequest,
    SongResult,
    PlaylistResponse,
    HealthResponse,
    SpotifyTrackInfo
)


class TestEmotionType:
    
    def test_emotion_type_values(self):
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
        emotions = list(EmotionType)
        assert len(emotions) == 10


class TestSongInput:
    
    def test_song_input_valid(self):
        song = SongInput(
            song_name="Bohemian Rhapsody",
            artist="Queen"
        )
        
        assert song.song_name == "Bohemian Rhapsody"
        assert song.artist == "Queen"
        assert song.spotify_id is None
    
    def test_song_input_with_spotify_id(self):
        song = SongInput(
            song_name="Imagine",
            artist="John Lennon",
            spotify_id="track123"
        )
        
        assert song.spotify_id == "track123"
    
    def test_song_input_missing_required_fields(self):
        with pytest.raises(ValidationError):
            SongInput(song_name="Test Song")
        
        with pytest.raises(ValidationError):
            SongInput(artist="Test Artist") 

    def test_song_input_empty_strings(self):
        song = SongInput(song_name="", artist="")
        
        assert song.song_name == ""
        assert song.artist == ""


class TestPlaylistRequest:
    
    def test_playlist_request_with_songs(self):
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
        request = PlaylistRequest(
            emotion=["happy"],
            num_results=15
        )
        
        assert request.emotion == ["happy"]
        assert request.songs is None
        assert request.num_results == 15
    
    def test_playlist_request_defaults(self):
        request = PlaylistRequest(emotion=["calm"])
        
        assert request.num_results == 10
        assert request.enrich_with_lyrics is False
    
    def test_playlist_request_num_results_validation(self):
        request = PlaylistRequest(emotion=["happy"], num_results=1)
        assert request.num_results == 1
        
        request = PlaylistRequest(emotion=["happy"], num_results=50)
        assert request.num_results == 50
        
        with pytest.raises(ValidationError):
            PlaylistRequest(emotion=["happy"], num_results=0)
        
        with pytest.raises(ValidationError):
            PlaylistRequest(emotion=["happy"], num_results=51)
        
        with pytest.raises(ValidationError):
            PlaylistRequest(emotion=["happy"], num_results=-1)
    
    def test_playlist_request_with_both_songs_and_emotion(self):
        request = PlaylistRequest(
            songs=[SongInput(song_name="Test", artist="Artist")],
            emotion=["energetic"],
            num_results=20
        )
        
        assert len(request.songs) == 1
        assert request.emotion == ["energetic"]
    
    def test_playlist_request_multiple_emotions(self):
        request = PlaylistRequest(
            emotion=["happy", "energetic", "upbeat"],
            num_results=10
        )
        
        assert len(request.emotion) == 3
    
    def test_playlist_request_boolean_flags(self):
        request = PlaylistRequest(
            emotion=["romantic"],
            enrich_with_lyrics=True
        )
        
        assert request.enrich_with_lyrics is True




class TestSongResult:
    
    def test_song_result_minimal(self):
        song = SongResult(
            song_name="Test Song",
            artist="Test Artist",
            similarity_score=0.85
        )
        
        assert song.song_name == "Test Song"
        assert song.artist == "Test Artist"
        assert song.similarity_score == 0.85
    
    def test_song_result_with_all_fields(self):
        song = SongResult(
            song_name="Complete Song",
            artist="Complete Artist",
            spotify_id="track123",
            similarity_score=0.92,
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


class TestPlaylistResponse:
    def test_playlist_response_minimal(self):
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
        assert response.emotion_features is None
    
    def test_playlist_response_with_all_fields(self):
        songs = [
            SongResult(
                song_name=f"Song {i}",
                artist=f"Artist {i}",
                similarity_score=0.9 - i * 0.1
            )
            for i in range(5)
        ]
        
        response = PlaylistResponse(
            playlist=songs,
            emotion_features={"valence": 0.7, "energy": 0.8},
            combined_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        assert len(response.playlist) == 5
        assert response.emotion_features is not None
        assert len(response.combined_embedding) == 5
    
    def test_playlist_response_empty_playlist(self):
        response = PlaylistResponse(playlist=[])
        
        assert len(response.playlist) == 0


class TestHealthResponse:
    
    def test_health_response_healthy(self):
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
    
    def test_spotify_track_info_complete(self):
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
    
    def test_nested_schema_validation(self):
        response = PlaylistResponse(
            playlist=[
                SongResult(
                    song_name="Test",
                    artist="Artist",
                    similarity_score=0.9
                )
            ]
        )
        
        assert response.playlist[0].similarity_score == 0.9
    
    def test_json_serialization(self):
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
        song = SongResult(
            song_name="Test",
            artist="Artist",
            similarity_score=0.9,
            spotify_id=None,
            album=None
        )
        
        assert song.spotify_id is None
        assert song.album is None
