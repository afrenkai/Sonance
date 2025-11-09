import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import List, Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class SpotifyService:
    def __init__(self):
        try:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                logger.warning("Spotify credentials not found in environment variables")
                self.spotify = None
                return
            
            auth_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.spotify = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("Spotify service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify service: {e}")
            self.spotify = None
    
    def is_available(self) -> bool:
        return self.spotify is not None
    
    def get_available_genre_seeds(self) -> List[str]:
        if not self.is_available():
            logger.warning("Spotify service not available")
            return []
        
        try:
            genres = self.spotify.recommendation_genre_seeds()
            logger.info(f"Retrieved {len(genres)} available genre seeds from Spotify")
            return genres.get('genres', [])
        except Exception as e:
            logger.error(f"Error getting genre seeds: {e}")
            return []
    
    def search_artist(
        self,
        artist_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            logger.warning("Spotify service not available")
            return []
        
        try:
            results = self.spotify.search(
                q=f"artist:{artist_name}",
                type='artist',
                limit=min(limit, 50)
            )
            
            artists = []
            for artist in results['artists']['items']:
                artists.append(self._format_artist(artist))
            
            logger.info(f"Found {len(artists)} artists for: {artist_name}")
            return artists
            
        except Exception as e:
            logger.error(f"Error searching for artist: {e}")
            return []
    
    def get_artist_top_tracks(
        self,
        artist_id: str,
        market: str = 'US',
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            logger.warning("Spotify service not available")
            return []
        
        try:
            results = self.spotify.artist_top_tracks(artist_id, country=market)
            
            tracks = []
            for track in results['tracks'][:limit]:
                tracks.append(self._format_track(track))
            
            logger.info(f"Got {len(tracks)} top tracks for artist {artist_id}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting artist top tracks: {e}")
            return []
    
    def get_artist_tracks_including_collabs(
        self,
        artist_id: str,
        artist_name: str,
        limit: int = 15
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            logger.warning("Spotify service not available")
            return []
        
        try:
            all_tracks = []
            seen_ids = set()
            
            top_tracks = self.get_artist_top_tracks(artist_id, limit=10)
            for track in top_tracks:
                track_id = track['spotify_id']
                if track_id not in seen_ids:
                    seen_ids.add(track_id)
                    all_tracks.append(track)
            
            collab_query = f'artist:"{artist_name}"'
            results = self.spotify.search(q=collab_query, type='track', limit=30)
            
            for track in results['tracks']['items']:
                track_id = track['id']
                if track_id not in seen_ids:
                    artist_names = [a['name'].lower() for a in track['artists']]
                    if artist_name.lower() in artist_names:
                        seen_ids.add(track_id)
                        all_tracks.append(self._format_track(track))
                        
                        if len(all_tracks) >= limit:
                            break
            
            logger.info(
                f"Got {len(all_tracks)} tracks for artist {artist_name} "
                f"(including collaborations)"
            )
            return all_tracks[:limit]
            
        except Exception as e:
            logger.error(f"Error getting artist tracks with collabs: {e}")
            # Fallback to just top tracks
            return self.get_artist_top_tracks(artist_id, limit=limit)
    
    def get_artist_by_id(self, artist_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            return None
        
        try:
            artist = self.spotify.artist(artist_id)
            return self._format_artist(artist)
        except Exception as e:
            logger.error(f"Error getting artist by ID: {e}")
            return None
    
    def get_similar_tracks_from_seeds(
        self,
        seed_track_ids: List[str],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get similar tracks by combining:
        1. Same artist's tracks (for some guaranteed matches)
        2. Broader search queries based on artist+genre (for variety)
        
        This balances between too narrow (only same artist) and too broad (keyword matching).
        """
        if not self.is_available():
            return []
        
        try:
            similar_tracks = []
            seen_track_ids = set(seed_track_ids)
            artist_info = []
            
            # Get artists and genres from seed tracks
            for track_id in seed_track_ids[:5]:
                try:
                    track = self.spotify.track(track_id)
                    for artist in track['artists']:
                        artist_id = artist['id']
                        artist_name = artist['name']
                        
                        # Get artist's genres
                        artist_data = self.spotify.artist(artist_id)
                        genres = artist_data.get('genres', [])
                        
                        artist_info.append({
                            'id': artist_id,
                            'name': artist_name,
                            'genres': genres
                        })
                except Exception as e:
                    logger.debug(f"Could not get track {track_id}: {e}")
            
            if not artist_info:
                logger.warning("Could not extract artists from seed tracks")
                return []
            
            logger.info(f"Found {len(artist_info)} artists from seed tracks")
            
            # Strategy 1: Get some tracks from the same artist (20-30% of results)
            same_artist_limit = min(30, limit // 3)
            for artist in artist_info[:2]:  # Limit to first 2 artists
                try:
                    # Get artist's albums
                    albums = self.spotify.artist_albums(
                        artist['id'], 
                        album_type='album,single', 
                        limit=5
                    )
                    
                    for album in albums['items'][:3]:
                        try:
                            album_tracks = self.spotify.album_tracks(album['id'], limit=5)
                            for track in album_tracks['items'][:3]:
                                track_id = track['id']
                                if track_id and track_id not in seen_track_ids:
                                    full_track = self.spotify.track(track_id)
                                    similar_tracks.append(self._format_track(full_track))
                                    seen_track_ids.add(track_id)
                                    
                                    if len(similar_tracks) >= same_artist_limit:
                                        break
                        except Exception as e:
                            logger.debug(f"Could not get tracks from album: {e}")
                        
                        if len(similar_tracks) >= same_artist_limit:
                            break
                except Exception as e:
                    logger.debug(f"Could not get albums for artist: {e}")
                
                if len(similar_tracks) >= same_artist_limit:
                    break
            
            logger.info(f"Got {len(similar_tracks)} tracks from same artists")
            
            # Strategy 2: Search by artist name + genre combinations for variety
            for artist in artist_info[:3]:
                if len(similar_tracks) >= limit:
                    break
                
                # Search variations to get diverse results
                search_queries = [
                    f'artist:"{artist["name"]}"',  # More from this artist
                ]
                
                # Add genre-based searches if we have genres
                if artist['genres']:
                    # Use first 2 genres for variety
                    for genre in artist['genres'][:2]:
                        # Search by genre WITHOUT mood keywords (avoids "melancholic" problem)
                        search_queries.append(f'genre:"{genre}"')
                
                for query in search_queries:
                    if len(similar_tracks) >= limit:
                        break
                    
                    try:
                        results = self.spotify.search(q=query, type='track', limit=20)
                        for track in results['tracks']['items']:
                            track_id = track['id']
                            if track_id and track_id not in seen_track_ids:
                                similar_tracks.append(self._format_track(track))
                                seen_track_ids.add(track_id)
                                
                                if len(similar_tracks) >= limit:
                                    break
                    except Exception as e:
                        logger.debug(f"Search query '{query}' failed: {e}")
            
            logger.info(f"Found {len(similar_tracks)} total tracks (mix of same artist + genre variety)")
            return similar_tracks[:limit]
            
        except Exception as e:
            logger.error(f"Error getting similar tracks: {e}")
            return []
    
    def search_track(
        self,
        song_name: str,
        artist: Optional[str] = None,
        limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            logger.warning("Spotify service not available")
            return None
        
        try:
            query = f"track:{song_name}"
            if artist:
                query += f" artist:{artist}"
            
            results = self.spotify.search(q=query, type='track', limit=limit)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                return self._format_track(track)
            
            logger.info(f"No track found for: {song_name} by {artist}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for track: {e}")
            return None
    
    def get_track_by_id(self, track_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            return None
        
        try:
            track = self.spotify.track(track_id)
            return self._format_track(track)
        except Exception as e:
            logger.error(f"Error getting track by ID: {e}")
            return None
    
   
    def search_tracks_by_emotion(
        self,
        emotion: str,
        num_results: int = 20
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        
        try:
            results = self.spotify.search(
                q=emotion,
                type='track',
                limit=min(num_results, 50)
            )
            
            tracks = []
            for track in results['tracks']['items']:
                tracks.append(self._format_track(track))
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error searching tracks by emotion: {e}")
            return []
    
    def search_by_multiple_queries(
        self,
        queries: List[str],
        limit_per_query: int = 20,
        randomize_offset: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search by multiple queries with optional offset randomization for variety.
        
        Args:
            queries: List of search query strings
            limit_per_query: Max results per query
            randomize_offset: If True, randomly offset results to avoid always getting the same top tracks
        
        Returns:
            List of unique track dictionaries
        """
        if not self.is_available():
            return []
        
        try:
            import random
            all_tracks = []
            seen_ids = set()
            
            for query in queries:
                # Add random offset to get different results each time (more variety, less "popular only")
                offset = 0
                if randomize_offset:
                    # Random offset between 0-40 to skip the always-same popular tracks
                    offset = random.randint(0, 40)
                
                results = self.spotify.search(
                    q=query,
                    type='track',
                    limit=min(limit_per_query, 50),
                    offset=offset
                )
                
                for track in results['tracks']['items']:
                    track_id = track['id']
                    if track_id not in seen_ids:
                        seen_ids.add(track_id)
                        all_tracks.append(self._format_track(track))
            
            logger.info(f"Found {len(all_tracks)} unique tracks from {len(queries)} queries (randomize_offset={randomize_offset})")
            return all_tracks
            
        except Exception as e:
            logger.error(f"Error searching by multiple queries: {e}")
            return []
    
    def get_album_tracks(
        self,
        album_id: str
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        
        try:
            results = self.spotify.album_tracks(album_id)
            tracks = []
            
            for track in results['items']:
                full_track = self.spotify.track(track['id'])
                tracks.append(self._format_track(full_track))
            
            logger.info(f"Got {len(tracks)} tracks from album {album_id}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting album tracks: {e}")
            return []
    
    def search_albums_by_query(
        self,
        query: str,
        limit: int = 10
    ) -> List[str]:
        if not self.is_available():
            return []
        
        try:
            results = self.spotify.search(
                q=query,
                type='album',
                limit=min(limit, 50)
            )
            
            album_ids = [album['id'] for album in results['albums']['items']]
            logger.info(f"Found {len(album_ids)} albums for query: {query}")
            return album_ids
            
        except Exception as e:
            logger.error(f"Error searching albums: {e}")
            return []
    
    def get_tracks_with_features(
        self,
        track_ids: List[str]
    ) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        
        try:
            batch_size = 50
            all_results = []
            
            for i in range(0, len(track_ids), batch_size):
                batch_ids = track_ids[i:i + batch_size]
                
                tracks = self.spotify.tracks(batch_ids)
                
                for track in tracks['tracks']:
                    if track:
                        track_data = self._format_track(track)
                        all_results.append(track_data)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error getting tracks: {e}")
            return []
    
    def _format_track(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """Format Spotify track data to our schema."""
        artists = ', '.join([artist['name'] for artist in track['artists']])
        
        return {
            'spotify_id': track['id'],
            'song_name': track['name'],
            'artist': artists,
            'album': track['album']['name'],
            'preview_url': track.get('preview_url'),
            'external_url': track['external_urls']['spotify'],
            'duration_ms': track['duration_ms'],
            'popularity': track.get('popularity', 0),
            'album_image': track['album']['images'][0]['url'] if track['album']['images'] else None
        }
    
    def _format_artist(self, artist: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'spotify_id': artist['id'],
            'name': artist['name'],
            'genres': artist.get('genres', []),
            'popularity': artist.get('popularity', 0),
            'image_url': artist['images'][0]['url'] if artist.get('images') else None,
            'external_url': artist['external_urls']['spotify']
        }
