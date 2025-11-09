import pytest
from unittest.mock import Mock, patch, MagicMock
from backend.services.spotify_service import SpotifyService


class TestSpotifyService:
    @patch('backend.services.spotify_service.os.getenv')
    @patch('backend.services.spotify_service.spotipy.Spotify')
    def test_initialization_with_credentials(self, mock_spotify, mock_getenv):
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
        service = SpotifyService()
        service.spotify = None
        
        result = service.search_track("Test Song", "Test Artist")
        
        assert result is None
    
    @patch('backend.services.spotify_service.spotipy.Spotify')
    @patch('backend.services.spotify_service.os.getenv')
    def test_search_track_with_results(self, mock_getenv, mock_spotify_class):
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
    
   
    def test_search_by_multiple_queries(self, mock_spotify_service):
        queries = ["happy music", "upbeat songs", "dance"]
        
        results = mock_spotify_service.search_by_multiple_queries(queries, limit_per_query=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    @patch('backend.services.spotify_service.spotipy.Spotify')
    @patch('backend.services.spotify_service.os.getenv')
    def test_search_by_multiple_queries_deduplication(self, mock_getenv, mock_spotify_class):
        mock_getenv.side_effect = lambda x: {
            'SPOTIFY_CLIENT_ID': 'test_id',
            'SPOTIFY_CLIENT_SECRET': 'test_secret'
        }.get(x)
        
        mock_spotify = MagicMock()
        mock_spotify_class.return_value = mock_spotify
        
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
        
        assert len(results) == 1
        assert results[0]['spotify_id'] == 'track123'
    
    def test_get_tracks_with_features(self, mock_spotify_service):
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
