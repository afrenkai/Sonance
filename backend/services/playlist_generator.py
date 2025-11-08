import numpy as np
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
from pathlib import Path

from backend.services.embedding_service import EmbeddingService
from backend.services.emotion_mapper import EmotionMapper
from backend.services.spotify_service import SpotifyService
from backend.services.async_genius_service import AsyncGeniusService
from backend.models.schemas import SongInput, SongResult, AudioFeatures

logger = logging.getLogger(__name__)


class PlaylistGenerator:
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        emotion_mapper: EmotionMapper,
        spotify_service: Optional[SpotifyService] = None,
        genius_service: Optional[AsyncGeniusService] = None,
        songs_db_path: Optional[str] = None
    ):
        self.embedding_service = embedding_service
        self.emotion_mapper = emotion_mapper
        self.spotify_service = spotify_service
        self.genius_service = genius_service
        self.songs_db_path = songs_db_path
        self.songs_df: Optional[pd.DataFrame] = None
        
        # Check if we have LLM-based emotion understanding
        self.has_llm_emotions = (
            hasattr(emotion_mapper, 'llm_emotion_service') and 
            emotion_mapper.llm_emotion_service is not None
        )
        
        if self.has_llm_emotions:
            logger.info("Playlist generator using LLM-based contextual emotion understanding")
        
        if songs_db_path:
            self._load_songs_database()
    
    def _load_songs_database(self):
        try:
            if self.songs_db_path.endswith('.csv'):
                self.songs_df = pd.read_csv(self.songs_db_path)
                logger.info(f"Loaded {len(self.songs_df)} songs from database")
            else:
                logger.warning(f"Unsupported database format: {self.songs_db_path}")
        except Exception as e:
            logger.error(f"Failed to load songs database: {e}")
    
    def generate_playlist(
        self,
        songs: Optional[List[SongInput]] = None,
        emotion: Optional[List[str]] = None,
        num_results: int = 10,
        enrich_with_lyrics: bool = False
    ) -> tuple[List[SongResult], np.ndarray, Dict[str, Any]]:
       
        if not songs and not emotion:
            raise ValueError("Must provide either songs or emotion")
        
        # Convert emotion list to single string for processing
        emotion_str = None
        if emotion:
            emotion_str = " ".join(emotion) if isinstance(emotion, list) else emotion
        
        combined_embedding = self._compute_combined_embedding(songs, emotion_str)
        
        emotion_features = None
        if emotion_str:
            # For multiple emotions, use the first one for feature ranges
            # or combine them if needed
            primary_emotion = emotion[0] if isinstance(emotion, list) else emotion_str
            emotion_features = self.emotion_mapper.get_feature_ranges(primary_emotion)
        
        # Use Spotify service if available
        if self.spotify_service and self.spotify_service.is_available():
            playlist = self._query_songs_with_spotify(
                songs,
                combined_embedding,
                emotion_str,
                emotion_features,
                num_results,
                enrich_with_lyrics
            )
        else:
            playlist = self._query_songs(
                combined_embedding,
                emotion_str,
                emotion_features,
                num_results
            )
        
        return playlist, combined_embedding, emotion_features
    
    def _compute_combined_embedding(
        self,
        songs: Optional[List[SongInput]],
        emotion: Optional[str]
    ) -> np.ndarray:
        
        embeddings = []
        weights = []
        
       
        if songs:
            for song in songs:
                song_emb = self.embedding_service.encode_song(
                    song.song_name,
                    song.artist
                )
                embeddings.append(song_emb)
            
            song_weight = 0.7 if emotion else 1.0
            weights.extend([song_weight / len(songs)] * len(songs))
        
        if emotion:
            emotion_emb = self.embedding_service.encode_emotion(emotion)
            embeddings.append(emotion_emb)
            
            emotion_weight = 0.3 if songs else 1.0
            weights.append(emotion_weight)
        
        combined = self.embedding_service.combine_embeddings(embeddings, weights)
        
        logger.info(f"Combined {len(embeddings)} embeddings into single vector")
        return combined
    
    def _query_songs_with_spotify(
        self,
        songs: Optional[List[SongInput]],
        query_embedding: np.ndarray,
        emotion: Optional[str],
        emotion_features: Optional[Dict],
        num_results: int,
        enrich_with_lyrics: bool = False
    ) -> List[SongResult]:
        """Query songs using Spotify API for real track data."""
        try:
            # Get seed tracks from input songs
            seed_track_ids = []
            if songs:
                for song in songs[:5]:  # Max 5 seeds for Spotify
                    if song.spotify_id:
                        seed_track_ids.append(song.spotify_id)
                    else:
                        # Search for the track
                        track = self.spotify_service.search_track(song.song_name, song.artist)
                        if track and track.get('spotify_id'):
                            seed_track_ids.append(track['spotify_id'])
            
            # Prepare target audio features from emotion
            target_features = {}
            if emotion_features:
                for key, value in emotion_features.items():
                    if isinstance(value, (tuple, list)) and len(value) == 2:
                        # Use midpoint of range
                        target_features[f'target_{key}'] = (value[0] + value[1]) / 2
                    else:
                        target_features[f'target_{key}'] = value
            
                        # Get tracks from Spotify using search (recommendations API is deprecated)
            if seed_track_ids or songs:
                logger.info("Searching for tracks similar to seed songs")
                # Search for tracks related to the seed songs
                search_queries = []
                
                if songs:
                    # Create search queries from seed songs
                    for song in songs[:5]:
                        search_queries.append(f"{song.song_name} {song.artist}")
                        # Also search by artist to find similar tracks
                        search_queries.append(song.artist)
                
                spotify_tracks = self.spotify_service.search_by_multiple_queries(
                    queries=search_queries,
                    limit_per_query=15
                )
                
            elif emotion:
                # For emotion-only queries, search for tracks matching emotion
                logger.info(f"Searching for tracks matching emotion: {emotion}")
                
                # Expand search with emotion-related keywords
                emotion_keywords = self._get_emotion_keywords(emotion)
                
                spotify_tracks = self.spotify_service.search_by_multiple_queries(
                    queries=emotion_keywords,
                    limit_per_query=20
                )
            else:
                spotify_tracks = []
            
            # Get audio features for all tracks
            if spotify_tracks:
                track_ids = [t['spotify_id'] for t in spotify_tracks if t.get('spotify_id')]
                
                # Try to get audio features, but continue without them if not available
                try:
                    tracks_with_features = self.spotify_service.get_tracks_with_features(track_ids)
                except Exception as e:
                    logger.warning(f"Could not get audio features (API may be restricted): {e}")
                    tracks_with_features = spotify_tracks  # Use tracks without features
                
                # If no tracks with features, use the original tracks
                if not tracks_with_features:
                    tracks_with_features = spotify_tracks
                
                # Convert to SongResult objects and compute similarity scores
                playlist = []
                for track_data in tracks_with_features:
                    # Compute embedding similarity
                    track_embedding = self.embedding_service.encode_song(
                        track_data['song_name'],
                        track_data['artist']
                    )
                    similarity_score = float(
                        self.embedding_service.compute_similarity(query_embedding, track_embedding)
                    )
                    
                    # If we have LLM emotions and an emotion query, also compute contextual emotion match
                    if self.has_llm_emotions and emotion:
                        try:
                            llm_service = self.emotion_mapper.llm_emotion_service
                            
                            # Compute emotion similarity based on song name and artist
                            song_text = f"{track_data['song_name']} by {track_data['artist']}"
                            emotion_similarity = llm_service.compute_emotion_similarity(
                                song_text,
                                emotion,
                                context="song"
                            )
                            
                            # Blend embedding similarity and emotion similarity
                            # 40% embedding, 60% contextual emotion
                            similarity_score = (
                                similarity_score * 0.4 +
                                emotion_similarity * 0.6
                            )
                            logger.debug(
                                f"{track_data['song_name']}: emb={similarity_score:.3f}, "
                                f"emotion={emotion_similarity:.3f}"
                            )
                        except Exception as e:
                            logger.debug(f"LLM emotion scoring failed for {track_data['song_name']}: {e}")
                    
                    # Create SongResult
                    audio_features = None
                    if 'audio_features' in track_data and track_data['audio_features']:
                        audio_features = AudioFeatures(**track_data['audio_features'])
                    
                    song_result = SongResult(
                        song_name=track_data['song_name'],
                        artist=track_data['artist'],
                        spotify_id=track_data.get('spotify_id'),
                        similarity_score=similarity_score,
                        audio_features=audio_features,
                        album=track_data.get('album'),
                        preview_url=track_data.get('preview_url'),
                        external_url=track_data.get('external_url'),
                        album_image=track_data.get('album_image'),
                        popularity=track_data.get('popularity', 0),
                        duration_ms=track_data.get('duration_ms')
                    )
                    playlist.append(song_result)
                
                # Sort by similarity score
                playlist.sort(key=lambda x: x.similarity_score, reverse=True)
                
                # Optionally enrich with Genius lyrics data
                # When enabled, this RE-RANKS using 80% lyrics + 20% embeddings
                if enrich_with_lyrics and self.genius_service and self.genius_service.is_available():
                    primary_emotion = emotion.split()[0] if emotion else None
                    playlist = self._enrich_with_genius_data(playlist[:num_results * 2], primary_emotion)
                
                logger.info(f"Generated Spotify playlist with {len(playlist)} songs")
                return playlist[:num_results]
            
            logger.warning("No tracks from Spotify, falling back to mock results")
            return self._generate_mock_results(num_results)
            
        except Exception as e:
            logger.error(f"Error querying Spotify: {e}", exc_info=True)
            return self._generate_mock_results(num_results)
    
    def _query_songs(
        self,
        query_embedding: np.ndarray,
        emotion: Optional[str],
        emotion_features: Optional[Dict],
        num_results: int
    ) -> List[SongResult]:
        
        if self.songs_df is None or self.songs_df.empty:
            logger.warning("No songs database loaded, returning mock results")
            return self._generate_mock_results(num_results)
        
       
        song_embeddings = np.array([
            eval(emb) if isinstance(emb, str) else emb
            for emb in self.songs_df['embedding']
        ])
        
        similarity_scores = self.embedding_service.batch_similarity(
            query_embedding,
            song_embeddings
        )
        
        results_df = self.songs_df.copy()
        results_df['similarity_score'] = similarity_scores
        
      
        if emotion and 'audio_features' in results_df.columns:
            emotion_scores = results_df['audio_features'].apply(
                lambda features: self.emotion_mapper.compute_emotion_score(
                    features, emotion
                ) if isinstance(features, dict) else 0.5
            )
            
            results_df['combined_score'] = (
                0.6 * results_df['similarity_score'] +
                0.4 * emotion_scores
            )
        else:
            results_df['combined_score'] = results_df['similarity_score']
        
        top_results = results_df.nlargest(num_results, 'combined_score')
        
        playlist = []
        for _, row in top_results.iterrows():
            song_result = SongResult(
                song_name=row['song_name'],
                artist=row['artist'],
                spotify_id=row.get('spotify_id'),
                similarity_score=float(row['similarity_score']),
                album=row.get('album'),
                preview_url=row.get('preview_url')
            )
            
            if 'audio_features' in row and isinstance(row['audio_features'], dict):
                song_result.audio_features = AudioFeatures(**row['audio_features'])
            
            playlist.append(song_result)
        
        logger.info(f"Generated playlist with {len(playlist)} songs")
        return playlist
    
    def _enrich_with_genius_data(self, playlist: List[SongResult], target_emotion: Optional[str] = None) -> List[SongResult]:
        """
        Enrich playlist with Genius lyrics emotional analysis and RE-RANK by lyrics score.
        
        This is the primary ranking mechanism - lyrics emotion > embeddings.
        
        Args:
            playlist: List of SongResult objects
            target_emotion: The emotion we're matching against
            
        Returns:
            Re-ranked playlist based primarily on lyrics emotional match
        """
        if not self.genius_service or not self.genius_service.is_available():
            logger.info("Genius service not available, using embedding-only ranking")
            return playlist
        
        try:
            # Prepare songs for batch search
            songs_to_search = [(song.song_name, song.artist) for song in playlist]
            
            # Batch get emotional profiles with target emotion scoring
            genius_results = self.genius_service.batch_get_emotional_profiles_sync(
                songs_to_search,
                target_emotion=target_emotion,
                max_concurrent=3  # Conservative to avoid rate limiting
            )
            
            # Enrich playlist with Genius data and compute new scores
            lyrics_scored_count = 0
            for song in playlist:
                key = f"{song.song_name}|{song.artist}"
                if key in genius_results and genius_results[key]:
                    genius_data = genius_results[key]
                    
                    # Store Genius URL
                    song.genius_url = genius_data.get('url')
                    
                    # Get lyrics-based emotion match score (0.0 to 1.0)
                    lyrics_score = genius_data.get('emotion_match_score', 0.0)
                    
                    if lyrics_score > 0:
                        # Combine scores: 80% lyrics, 20% embeddings
                        # This heavily favors lyrics-based emotional match
                        original_embedding_score = song.similarity_score
                        song.similarity_score = (
                            lyrics_score * 0.8 +  # 80% weight on lyrics emotion
                            original_embedding_score * 0.2  # 20% weight on name/artist similarity
                        )
                        lyrics_scored_count += 1
                        
                        logger.debug(
                            f"{song.song_name}: lyrics={lyrics_score:.3f}, "
                            f"embedding={original_embedding_score:.3f}, "
                            f"final={song.similarity_score:.3f}"
                        )
            
            # Re-sort by new combined score (lyrics-heavy)
            playlist.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(
                f"Re-ranked playlist: {lyrics_scored_count}/{len(playlist)} songs "
                f"with lyrics-based scoring (80% lyrics, 20% embeddings)"
            )
            
        except Exception as e:
            logger.warning(f"Could not enrich with Genius data: {e}")
        
        return playlist
    
    def _get_emotion_keywords(self, emotion: str) -> List[str]:
        """Generate search keywords for an emotion."""
        emotion_keywords_map = {
            "happy": ["happy", "upbeat", "cheerful", "joyful", "party"],
            "sad": ["sad", "melancholy", "heartbreak", "emotional", "tearjerker"],
            "energetic": ["energetic", "upbeat", "pump up", "workout", "intense"],
            "calm": ["calm", "relaxing", "chill", "peaceful", "ambient"],
            "angry": ["angry", "aggressive", "intense", "rage", "metal"],
            "melancholic": ["melancholic", "nostalgic", "bittersweet", "reflective"],
            "hopeful": ["hopeful", "inspiring", "uplifting", "optimistic"],
            "romantic": ["romantic", "love", "ballad", "intimate"],
            "anxious": ["anxious", "tense", "suspenseful", "dark"],
            "peaceful": ["peaceful", "serene", "tranquil", "meditation", "ambient"],
        }
        
        emotion_lower = emotion.lower().strip()
        keywords = emotion_keywords_map.get(emotion_lower, [emotion_lower])
        
        # Also add the emotion itself if not in the list
        if emotion_lower not in keywords:
            keywords.insert(0, emotion_lower)
        
        return keywords[:5]  # Limit to 5 keywords
    
    def _get_genre_seeds_for_emotion(self, emotion: str) -> List[str]:
        """Map emotions to Spotify genre seeds (deprecated - kept for compatibility)."""
        emotion_genres = {
            "happy": ["pop", "dance", "party"],
            "sad": ["sad", "acoustic", "singer-songwriter"],
            "energetic": ["edm", "work-out", "power-pop"],
            "calm": ["ambient", "chill", "sleep"],
            "angry": ["metal", "hard-rock", "punk"],
            "melancholic": ["indie", "folk", "blues"],
            "hopeful": ["indie-pop", "alternative", "soul"],
            "romantic": ["r-n-b", "soul", "romance"],
            "anxious": ["alternative", "emo", "grunge"],
            "peaceful": ["ambient", "classical", "meditation"],
        }
        
        emotion_lower = emotion.lower().strip()
        return emotion_genres.get(emotion_lower, ["pop"])
    
    def _generate_mock_results(self, num_results: int) -> List[SongResult]:
        mock_songs = [
            ("Bohemian Rhapsody", "Queen"),
            ("Imagine", "John Lennon"),
            ("Hotel California", "Eagles"),
            ("Stairway to Heaven", "Led Zeppelin"),
            ("Hey Jude", "The Beatles"),
            ("Smells Like Teen Spirit", "Nirvana"),
            ("Billie Jean", "Michael Jackson"),
            ("Sweet Child O' Mine", "Guns N' Roses"),
            ("Come Together", "The Beatles"),
            ("Purple Haze", "Jimi Hendrix"),
        ]
        
        results = []
        for i, (song_name, artist) in enumerate(mock_songs[:num_results]):
            results.append(SongResult(
                song_name=song_name,
                artist=artist,
                similarity_score=0.9 - (i * 0.05),
            ))
        
        return results
