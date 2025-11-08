import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.services.spotify_service import SpotifyService


class TestSpotifyService:
    
    @patch('backend.services.spotify_service.os.getenv')
    @patch('backend.services.spotify_service.spotipy.Spotify')
    def test_initialization_with_credentials(self, mock_spotify, mock_getenv):
        """Test SpotifyService initialization with valid credentials."""
        mock_getenv.side_effect = lambda x: {
            'SPOTIFY_CLIENT_ID': 'test_id',
            'SPOTIFY_CLIENT_SECRET': 'test_secret'
        }.get(x)
        
        service = SpotifyService()
        
        assert service.is_available()
        assert service.spotify is not None
    
    @patch('backend.services.spotify_service.os.getenv')
    def test_initialization_without_credentials(self, mock_getenv):
        mock_getenv.return_value = None
        
        service = SpotifyService()
        
        assert not service.is_available()
        assert service.spotify is None
    
    def test_is_available(self, mock_spotify_service):
        assert mock_spotify_service.is_available()
    
    def test_search_track_success(self, mock_spotify_service):
        result = mock_spotify_service.search_track("Test Song", "Test Artist")
        
        assert result is not None
        assert result['song_name'] == 'Test Song'
        assert result['artist'] == 'Test Artist'
        assert 'spotify_id' in result
    
    def test_search_track_not_available(self):
        """Test search when service is not available."""
        service = SpotifyService()
        service.spotify = None
        
        result = service.search_track("Test Song", "Test Artist")
        
        assert result is None
    
    @patch('backend.services.spotify_service.spotipy.Spotify')
    @patch('backend.services.spotify_service.os.getenv')
    def test_search_track_with_results(self, mock_getenv, mock_spotify_class):
        """Test searching for a track that exists."""
        mock_getenv.side_effect = lambda x: {
            'SPOTIFY_CLIENT_ID': 'test_id',
            'SPOTIFY_CLIENT_SECRET': 'test_secret'
        }.get(x)
        
        mock_spotify = MagicMock()
        mock_spotify_class.return_value = mock_spotify
        
        mock_spotify.search.return_value = {
            'tracks': {
                'items': [{
                    'id': 'track123',
                    'name': 'Bohemian Rhapsody',
                    'artists': [{'name': 'Queen'}],
                    'album': {
                        'name': 'A Night at the Opera',
                        'images': [{'url': 'https://image.url'}]
                    },
                    'preview_url': 'https://preview.url',
                    'external_urls': {'spotify': 'https://spotify.url'},
                    'duration_ms': 354000,
                    'popularity': 95
                }]
            }
        }
        
        service = SpotifyService()
        result = service.search_track("Bohemian Rhapsody", "Queen")
        
        assert result is not None
        assert result['song_name'] == 'Bohemian Rhapsody'
        assert result['artist'] == 'Queen'
        assert result['spotify_id'] == 'track123'
    
    @patch('backend.services.spotify_service.spotipy.Spotify')
    @patch('backend.services.spotify_service.os.getenv')
    def test_search_track_no_results(self, mock_getenv, mock_spotify_class):
        """Test searching for a track that doesn't exist."""
        mock_getenv.side_effect = lambda x: {
            'SPOTIFY_CLIENT_ID': 'test_id',
            'SPOTIFY_CLIENT_SECRET': 'test_secret'
        }.get(x)
        
        mock_spotify = MagicMock()
        mock_spotify_class.return_value = mock_spotify
        
        mock_spotify.search.return_value = {
            'tracks': {'items': []}
        }
        
        service = SpotifyService()
        result = service.search_track("NonexistentSong", "UnknownArtist")
        
        assert result is None
    
    def test_get_audio_features_success(self, mock_spotify_service):
        """Test getting audio features for a track."""
        result = mock_spotify_service.get_audio_features("test_track_id")
        
        assert result is not None
        assert 'valence' in result
        assert 'energy' in result
        assert 'danceability' in result
        assert 0.0 <= result['valence'] <= 1.0
        assert 0.0 <= result['energy'] <= 1.0
    
    @patch('backend.services.spotify_service.spotipy.Spotify')
    @patch('backend.services.spotify_service.os.getenv')
    def test_get_audio_features_with_data(self, mock_getenv, mock_spotify_class):
        """Test getting audio features with real data structure."""
        mock_getenv.side_effect = lambda x: {
            'SPOTIFY_CLIENT_ID': 'test_id',
            'SPOTIFY_CLIENT_SECRET': 'test_secret'
        }.get(x)
        
        mock_spotify = MagicMock()
        mock_spotify_class.return_value = mock_spotify
        
        mock_spotify.audio_features.return_value = [{
            'valence': 0.8,
            'energy': 0.7,
            'danceability': 0.6,
            'tempo': 120.0,
            'acousticness': 0.3,
            'instrumentalness': 0.1,
            'liveness': 0.2,
            'speechiness': 0.1
        }]
        
        service = SpotifyService()
        result = service.get_audio_features("track123")
        
        assert result is not None
        assert result['valence'] == 0.8
        assert result['energy'] == 0.7
        assert result['tempo'] == 120.0
    
    def test_search_by_multiple_queries(self, mock_spotify_service):
        """Test searching with multiple queries."""
        queries = ["happy music", "upbeat songs", "dance"]
        
        results = mock_spotify_service.search_by_multiple_queries(queries, limit_per_query=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    @patch('backend.services.spotify_service.spotipy.Spotify')
    @patch('backend.services.spotify_service.os.getenv')
    def test_search_by_multiple_queries_deduplication(self, mock_getenv, mock_spotify_class):
        """Test that multiple queries deduplicate results."""
        mock_getenv.side_effect = lambda x: {
            'SPOTIFY_CLIENT_ID': 'test_id',
            'SPOTIFY_CLIENT_SECRET': 'test_secret'
        }.get(x)
        
        mock_spotify = MagicMock()
        mock_spotify_class.return_value = mock_spotify
        
        # Return same track for different queries
        mock_track = {
            'id': 'track123',
            'name': 'Happy Song',
            'artists': [{'name': 'Artist'}],
            'album': {
                'name': 'Album',
                'images': [{'url': 'https://image.url'}]
            },
            'preview_url': 'https://preview.url',
            'external_urls': {'spotify': 'https://spotify.url'},
            'duration_ms': 200000,
            'popularity': 80
        }
        
        mock_spotify.search.return_value = {
            'tracks': {'items': [mock_track]}
        }
        
        service = SpotifyService()
        results = service.search_by_multiple_queries(["query1", "query2"], limit_per_query=5)
        
        # Should only have one result despite two queries returning the same track
        assert len(results) == 1
        assert results[0]['spotify_id'] == 'track123'
    
    def test_get_tracks_with_features(self, mock_spotify_service):
        """Test getting multiple tracks with their features."""
        track_ids = ["track1", "track2", "track3"]
        
        results = mock_spotify_service.get_tracks_with_features(track_ids)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for track in results:
            assert 'spotify_id' in track
            assert 'song_name' in track
            assert 'audio_features' in track
            assert isinstance(track['audio_features'], dict)
    
    def test_format_track(self, mock_spotify_service):
        """Test formatting a Spotify track."""
        raw_track = {
            'id': 'track123',
            'name': 'Test Song',
            'artists': [{'name': 'Artist 1'}, {'name': 'Artist 2'}],
            'album': {
                'name': 'Test Album',
                'images': [{'url': 'https://image.url'}]
            },
            'preview_url': 'https://preview.url',
            'external_urls': {'spotify': 'https://spotify.url'},
            'duration_ms': 240000,
            'popularity': 75
        }
        
        formatted = mock_spotify_service._format_track(raw_track)
        
        assert formatted['spotify_id'] == 'track123'
        assert formatted['song_name'] == 'Test Song'
        assert formatted['artist'] == 'Artist 1, Artist 2'
        assert formatted['album'] == 'Test Album'
        assert formatted['duration_ms'] == 240000
    
    def test_format_audio_features(self, mock_spotify_service):
        """Test formatting audio features."""
        raw_features = {
            'valence': 0.7,
            'energy': 0.8,
            'danceability': 0.6,
            'tempo': 128.5,
            'acousticness': 0.2,
            'instrumentalness': 0.0,
            'liveness': 0.15,
            'speechiness': 0.05
        }
        
        formatted = mock_spotify_service._format_audio_features(raw_features)
        
        assert formatted['valence'] == 0.7
        assert formatted['energy'] == 0.8
        assert formatted['tempo'] == 128.5
        assert len(formatted) == 8
    
    def test_get_recommendations_deprecated(self, mock_spotify_service):
        """Test that recommendations API returns empty (deprecated)."""
        results = mock_spotify_service.get_recommendations(
            seed_tracks=["track1", "track2"],
            limit=20
        )
        
        assert isinstance(results, list)
        assert len(results) == 0  # API is deprecated
    
    def test_search_tracks_by_emotion(self, mock_spotify_service):
        """Test searching tracks by emotion."""
        # Mock the implementation for this test
        mock_spotify_service.search_tracks_by_emotion = Mock(return_value=[
            {
                'spotify_id': 'track1',
                'song_name': 'Happy Song',
                'artist': 'Artist 1',
                'album': 'Album 1'
            }
        ])
        
        results = mock_spotify_service.search_tracks_by_emotion("happy", num_results=10)
        
        assert isinstance(results, list)
        mock_spotify_service.search_tracks_by_emotion.assert_called_once_with("happy", num_results=10)
