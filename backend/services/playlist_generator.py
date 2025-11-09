import numpy as np
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.services.embedding_service import EmbeddingService
from backend.services.emotion_mapper import EmotionMapper
from backend.services.spotify_service import SpotifyService
from backend.services.async_genius_service import AsyncGeniusService
from backend.services.llm_search_query_generator import LLMSearchQueryGenerator
from backend.models.schemas import SongInput, SongResult, ArtistInput

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
        
        self.query_generator = LLMSearchQueryGenerator(spotify_service=spotify_service)
        logger.info("Playlist generator using LLM-powered dynamic query generation with runtime genre filtering")
        
        # if spotify_service and spotify_service.is_available():
        #     success = self.query_generator.load_genre_corpus_from_spotify()
        #     if success:
        #         logger.info("Genre corpus loaded from Spotify - using runtime filtering")
        #     else:
        #         logger.info("Could not load genre corpus - using predefined genres")
        #
        self.has_llm_emotions = (
            hasattr(emotion_mapper, 'llm_emotion_service') and 
            emotion_mapper.llm_emotion_service is not None
        )
        
        if self.has_llm_emotions:
            logger.info("Playlist generator using LLM-based contextual emotion understanding")
        
        if songs_db_path:
            self._load_songs_database()

    # ------------------------
    # Helper methods for emotion search
    # ------------------------
    def _fetch_unbiased_pool_parallel(
        self,
        years: List[tuple],
        samples_per_period: int = 3,
        limit_per_query: int = 50,
        target_count: int = 2500,
        max_workers: int = 12,
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        import random
        if random_seed is not None:
            random.seed(random_seed)

        queries_to_run = []
        for year_start, year_end in years:
            for _ in range(samples_per_period):
                offset = random.randint(0, max(0, 800))
                queries_to_run.append((year_start, year_end, offset, limit_per_query))

        def _fetch(params):
            ys, ye, off, lim = params
            try:
                results = self.spotify_service.spotify.search(
                    q=f'year:{ys}-{ye}',
                    type='track',
                    limit=lim,
                    offset=off
                )
                tracks = []
                for t in results['tracks']['items']:
                    if t and t.get('id'):
                        tracks.append(self.spotify_service._format_track(t))
                return tracks
            except Exception:
                return []

        all_tracks: List[Dict[str, Any]] = []
        seen_ids = set()
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_fetch, q): q for q in queries_to_run}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception:
                    res = []
                for tr in res:
                    tid = tr.get('spotify_id')
                    if tid and tid not in seen_ids:
                        seen_ids.add(tid)
                        all_tracks.append(tr)
                        if len(all_tracks) >= target_count:
                            break
                if len(all_tracks) >= target_count:
                    break

        return all_tracks

    def _build_emotion_embedding(self, emotion: str) -> Any:
        # Build a deep contextual embedding for an emotion
        emotion_lower = (emotion or '').lower().strip()
        emotion_expansions = {
            'angry': 'rage fury aggressive hostile resentful irritated frustrated mad intense forceful confrontational rebellious defiant fierce bitter violent explosive wrathful',
            'sad': 'sorrow grief despair melancholy depressed heartbroken tearful mournful downcast miserable gloomy dejected sorrowful anguish devastated weeping crying',
            'happy': 'joyful cheerful delighted pleased content elated upbeat excited positive optimistic bright enthusiastic gleeful ecstatic jubilant radiant',
            'melancholic': 'wistful nostalgic bittersweet longing reflective pensive somber contemplative moody brooding melancholy sorrowful yearning elegiac',
            'anxious': 'worried nervous tense uneasy restless fearful stressed troubled apprehensive uncertain edgy panicked unsettled frantic',
            'calm': 'peaceful serene tranquil relaxed soothing gentle quiet still meditative placid mellow restful zen harmonious',
            'energetic': 'dynamic vibrant lively powerful intense active vigorous spirited animated charged explosive electric wild pumping',
            'romantic': 'loving tender intimate affectionate passionate devoted yearning longing sweet sensual desire amorous enchanted',
            'nostalgic': 'wistful reminiscent sentimental longing memories past reflective bittersweet yearning remembering bygone retrospective',
            'hopeful': 'optimistic aspiring uplifting encouraging positive inspiring bright promising expectant confident reassuring heartening',
            'lonely': 'isolated alone abandoned empty solitary distant separated disconnected forlorn desolate forsaken lonesome withdrawn',
            'euphoric': 'ecstatic blissful elated rapturous thrilled exhilarated overjoyed jubilant transcendent elevated intoxicated celestial',
        }

        expanded = emotion_expansions.get(emotion_lower, emotion_lower)
        context = f"Music expressing {emotion_lower}. Related feelings: {expanded}."
        emb = self.embedding_service.encode_emotion(context)
        return emb

    def _score_tracks_parallel(
        self,
        tracks: List[Dict[str, Any]],
        query_embedding: Any,
        emotion: Optional[str] = None,
        max_workers: int = 8
    ) -> List[SongResult]:
        # Deduplicate combinations first
        seen_combos = set()
        unique = []
        for t in tracks:
            combo = (t['song_name'].lower().strip(), t['artist'].lower().strip())
            if combo in seen_combos:
                continue
            seen_combos.add(combo)
            unique.append(t)

        def _score(t):
            # Normalize embeddings for stable cosine similarity
            track_emb = self.embedding_service.encode_song(t['song_name'], t['artist'])
            try:
                import numpy as _np
                track_emb = track_emb / (_np.linalg.norm(track_emb) + 1e-9)
                q_emb = query_embedding / (_np.linalg.norm(query_embedding) + 1e-9)
            except Exception:
                q_emb = query_embedding

            score = float(self.embedding_service.compute_similarity(q_emb, track_emb))
            # penalties
            literal = 0.0
            if emotion:
                for w in emotion.split():
                    if len(w) > 3 and w in t['song_name'].lower():
                        # reduce literal penalty - don't over-penalize
                        literal = 0.20
                        break
            pop = t.get('popularity', 50)
            pop_pen = (pop - 5) / 100 * 0.12 if pop > 5 else 0.0
            score -= (literal + pop_pen)
            return SongResult(
                song_name=t['song_name'],
                artist=t['artist'],
                spotify_id=t.get('spotify_id'),
                similarity_score=score,
                album=t.get('album'),
                preview_url=t.get('preview_url'),
                external_url=t.get('external_url'),
                album_image=t.get('album_image'),
                popularity=t.get('popularity', 0),
                duration_ms=t.get('duration_ms')
            )

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_score, unique))

        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
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
        artists: Optional[List[ArtistInput]] = None,
        emotion: Optional[List[str]] = None,
        num_results: int = 10,
        enrich_with_lyrics: bool = True,  # Changed default to True
        random_seed: Optional[int] = None  # For deterministic emotion search
    ) -> tuple[List[SongResult], np.ndarray, Dict[str, Any]]:
       
        if not songs and not artists and not emotion:
            raise ValueError("Must provide either songs, artists, or emotion")
        
        emotion_str = None
        emotion_list = None
        if emotion:
            if isinstance(emotion, list):
                emotion_list = emotion
                emotion_str = " ".join(emotion)
                logger.info(f"Processing multiple emotions: {emotion_list}")
            else:
                emotion_str = emotion
                emotion_list = [emotion]
        
        combined_embedding = self._compute_combined_embedding(songs, artists, emotion_str)
        
        # Extract emotion features from the emotion mapper
        # Note: Spotify audio features API is deprecated, but we keep this for legacy test compatibility
        emotion_features = None
        if emotion_str:
            try:
                # Try to get features from predefined mappings
                emotion_lower = emotion_str.lower().strip()
                from backend.models.schemas import EmotionType
                
                # Check if it's a predefined emotion
                emotion_enum = None
                for et in EmotionType:
                    if et.value.lower() == emotion_lower:
                        emotion_enum = et.value
                        break
                
                if emotion_enum and emotion_enum in self.emotion_mapper.emotion_mappings:
                    emotion_features = self.emotion_mapper.emotion_mappings[emotion_enum]
                else:
                    # Try parsing custom emotion
                    emotion_features = self.emotion_mapper._parse_custom_emotion(emotion_lower)
                
                # If no features found, create a basic feature set
                if not emotion_features:
                    emotion_features = {
                        "valence": (0.3, 0.7),
                        "energy": (0.3, 0.7)
                    }
            except Exception as e:
                logger.debug(f"Could not extract emotion features: {e}")
                # Return basic features for compatibility
                emotion_features = {
                    "valence": (0.3, 0.7),
                    "energy": (0.3, 0.7)
                }
        
        if self.spotify_service and self.spotify_service.is_available():
            playlist = self._query_songs_with_spotify(
                songs,
                artists,
                combined_embedding,
                emotion_str,
                emotion_features,
                num_results,
                enrich_with_lyrics,
                random_seed=random_seed
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
        songs: Optional[List[SongInput]] = None,
        artists: Optional[List[ArtistInput]] = None,
        emotion: Optional[str] = None
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
        
        if artists and self.spotify_service:
            logger.info(f"Fetching tracks (including collabs) for {len(artists)} artists")
            for artist in artists:
                if artist.spotify_id:
                    artist_id = artist.spotify_id
                    artist_name = artist.artist_name
                else:
                    artist_results = self.spotify_service.search_artist(artist.artist_name, limit=1)
                    if artist_results:
                        artist_id = artist_results[0]['spotify_id']
                        artist_name = artist_results[0]['name']
                    else:
                        logger.warning(f"Could not find artist: {artist.artist_name}")
                        continue
                
                # Get tracks including collaborations for this artist
                artist_tracks = self.spotify_service.get_artist_tracks_including_collabs(
                    artist_id, 
                    artist_name,
                    limit=8  # Get more tracks to better represent their style
                )
                
                if artist_tracks:
                    artist_track_embeddings = []
                    for track in artist_tracks:
                        track_emb = self.embedding_service.encode_song(
                            track['song_name'],
                            track['artist']
                        )
                        artist_track_embeddings.append(track_emb)
                    
                    artist_avg_emb = np.mean(artist_track_embeddings, axis=0)
                    embeddings.append(artist_avg_emb)
                    
                    artist_weight = 0.7 if emotion else 1.0
                    weights.append(artist_weight / len(artists))
                    logger.info(
                        f"Added embedding for artist {artist_name} based on "
                        f"{len(artist_tracks)} tracks (including collabs)"
                    )
        
        if emotion:
            emotion_emb = self.embedding_service.encode_emotion(emotion)
            embeddings.append(emotion_emb)
            
            emotion_weight = 0.3 if (songs or artists) else 1.0
            weights.append(emotion_weight)
        
        combined = self.embedding_service.combine_embeddings(embeddings, weights)
        
        logger.info(f"Combined {len(embeddings)} embeddings into single vector")
        return combined
    
    def _query_songs_with_spotify(
        self,
        songs: Optional[List[SongInput]] = None,
        artists: Optional[List[ArtistInput]] = None,
        query_embedding: np.ndarray = None,
        emotion: Optional[str] = None,
        emotion_features: Optional[Dict] = None,
        num_results: int = 10,
        enrich_with_lyrics: bool = False
        , random_seed: Optional[int] = None
    ) -> List[SongResult]:
        """Query songs using Spotify API for real track data."""
        try:
            seed_track_ids = []
            if songs:
                for song in songs[:5]:
                    if song.spotify_id:
                        seed_track_ids.append(song.spotify_id)
                    else:
                        track = self.spotify_service.search_track(song.song_name, song.artist)
                        if track and track.get('spotify_id'):
                            seed_track_ids.append(track['spotify_id'])
            
            if artists:
                for artist in artists:
                    if len(seed_track_ids) >= 5:
                        break
                    
                    artist_id = artist.spotify_id
                    artist_name = artist.artist_name
                    if not artist_id:
                        artist_results = self.spotify_service.search_artist(artist.artist_name, limit=1)
                        if artist_results:
                            artist_id = artist_results[0]['spotify_id']
                            artist_name = artist_results[0]['name']
                    
                    if artist_id:
                        # Use tracks including collabs for seed track IDs
                        artist_tracks = self.spotify_service.get_artist_tracks_including_collabs(
                            artist_id, 
                            artist_name, 
                            limit=3
                        )
                        for track in artist_tracks[:2]:
                            if len(seed_track_ids) >= 5:
                                break
                            seed_track_ids.append(track['spotify_id'])
            
            if seed_track_ids or songs or artists:
                spotify_tracks = []  # Initialize the list
                logger.info("🎵 Song/artist-based search - using hybrid strategy (same artist + genre variety)")
                
                # Get tracks using hybrid approach: same artist + genre-based variety
                # This avoids both problems: too narrow (only same artist) and keyword stuffing
                if seed_track_ids:
                    logger.info(f"🔍 Getting diverse tracks based on {len(seed_track_ids)} seed tracks")
                    try:
                        similar_tracks = self.spotify_service.get_similar_tracks_from_seeds(
                            seed_track_ids[:5],  # Max 5 seeds
                            limit=100  # Get many candidates for lyrics filtering
                        )
                        spotify_tracks.extend(similar_tracks)
                        logger.info(f"✓ Got {len(similar_tracks)} tracks (mix of same artist + genre variety)")
                        
                        if len(similar_tracks) > 0:
                            sample_tracks = [f"{t['song_name']} by {t['artist']}" for t in similar_tracks[:5]]
                            logger.info(f"Sample tracks: {sample_tracks}")
                    except Exception as e:
                        logger.error(f"❌ Hybrid strategy failed: {e}", exc_info=True)
                
                # Only supplement with genre search if we got very few results
                # SKIP keyword search entirely for song-based searches to avoid "melancholic" problem
                if len(spotify_tracks) < 30:
                    logger.warning(f"⚠️  Only got {len(spotify_tracks)} tracks from related artists")
                    
                    # Only use genre search if no songs were provided (pure emotion/artist search)
                    if not songs:
                        logger.info("📝 No seed songs provided, supplementing with genre-based search")
                        search_queries = []
                        
                        if emotion:
                            logger.info(f"Using LLM to generate queries for emotion '{emotion}'")
                            emotion_queries = self.query_generator.generate_queries_for_emotion(
                                emotion,
                                num_queries=6
                            )
                            search_queries.extend(emotion_queries)
                        elif artists:
                            logger.info("Inferring mood from artists and generating queries")
                            seed_tuples = []
                            
                            for artist in artists:
                                artist_id = artist.spotify_id
                                artist_name = artist.artist_name
                                if not artist_id:
                                    artist_results = self.spotify_service.search_artist(artist.artist_name, limit=1)
                                    if artist_results:
                                        artist_id = artist_results[0]['spotify_id']
                                        artist_name = artist_results[0]['name']
                                
                                if artist_id:
                                    artist_tracks = self.spotify_service.get_artist_tracks_including_collabs(
                                        artist_id, 
                                        artist_name, 
                                        limit=4
                                    )
                                    for track in artist_tracks[:2]:
                                        seed_tuples.append((track['song_name'], track['artist']))
                            
                            if seed_tuples:
                                seed_queries = self.query_generator.generate_queries_for_seed_songs(
                                    seed_tuples,
                                    num_queries=7
                                )
                                search_queries.extend(seed_queries)
                        
                        if search_queries:
                            additional_tracks = self.spotify_service.search_by_multiple_queries(
                                queries=search_queries,
                                limit_per_query=15
                            )
                            spotify_tracks.extend(additional_tracks)
                            logger.info(f"Added {len(additional_tracks)} tracks from genre search")
                    else:
                        logger.info("🚫 Skipping genre search for song-based query to avoid keyword matching")
                
            elif emotion:
                logger.info(f"🎭 Emotion-based search for: '{emotion}' - structured pipeline")

                # Configure broad time periods and targets
                years = [
                    (2015, 2024),
                    (2005, 2014),
                    (1990, 2004),
                    (1970, 1989),
                ]

                # 1) Fetch unbiased pool (parallel)
                spotify_tracks = self._fetch_unbiased_pool_parallel(
                    years=years,
                    samples_per_period=3,
                    limit_per_query=50,
                    target_count=2500,
                    max_workers=12,
                    random_seed=random_seed
                )

                # 2) Build emotion embedding and use it as query embedding
                query_embedding = self._build_emotion_embedding(emotion)

                # 3) Score tracks in parallel using the emotion embedding
                playlist = self._score_tracks_parallel(
                    spotify_tracks,
                    query_embedding,
                    emotion=emotion,
                    max_workers=8
                )

                logger.info(f"Generated initial scored playlist of {len(playlist)} items for emotion '{emotion}'")
                spotify_tracks = []
            
            # If 'playlist' already exists, it was created by the emotion pipeline.
            if 'playlist' in locals() and playlist:
                final_candidates = playlist
            elif spotify_tracks:
                # Use unified parallel scorer for spotify_tracks
                final_candidates = self._score_tracks_parallel(
                    spotify_tracks,
                    query_embedding,
                    emotion=emotion,
                    max_workers=8
                )
            else:
                final_candidates = []

            if not final_candidates:
                logger.warning("No tracks from Spotify, falling back to mock results")
                return self._generate_mock_results(num_results)

            # Apply artist diversity constraints and limit results
            playlist = final_candidates
            if len(playlist) > num_results:
                final_playlist = []
                artist_count = {}
                max_per_artist = 2

                for track in playlist:
                    artist = track.artist.lower()
                    if len(final_playlist) < num_results:
                        if artist_count.get(artist, 0) < max_per_artist:
                            final_playlist.append(track)
                            artist_count[artist] = artist_count.get(artist, 0) + 1

                if len(final_playlist) < num_results:
                    max_per_artist = 3
                    for track in playlist:
                        if len(final_playlist) >= num_results:
                            break
                        artist = track.artist.lower()
                        if track not in final_playlist and artist_count.get(artist, 0) < max_per_artist:
                            final_playlist.append(track)
                            artist_count[artist] = artist_count.get(artist, 0) + 1

                playlist = final_playlist

            # Enrich with lyrics if available
            if enrich_with_lyrics and self.genius_service and self.genius_service.is_available():
                primary_emotion = emotion.split()[0] if emotion else None
                candidate_multiplier = 6 if (emotion and not songs) else 4
                playlist = self._enrich_with_genius_data(
                    playlist[:num_results * candidate_multiplier],
                    primary_emotion,
                    seed_songs=songs,
                    seed_artists=artists
                )

            logger.info(f"Generated Spotify playlist with {len(playlist)} songs")
            return playlist[:num_results]
            
            logger.warning("No tracks from Spotify, falling back to mock results")
            return self._generate_mock_results(num_results)
            
        except Exception as e:
            logger.error(f"Error querying Spotify: {e}", exc_info=True)
            return self._generate_mock_results(num_results)
    
    def _get_emotion_keywords(self, emotion: str) -> List[str]:
        """
        Get relevant keywords for an emotion to improve search.
        
        Args:
            emotion: The emotion string
            
        Returns:
            List of keywords related to the emotion
        """
        emotion_keyword_map = {
            'happy': ['happy', 'joyful', 'upbeat', 'cheerful', 'bright'],
            'sad': ['sad', 'melancholic', 'sorrow', 'heartbroken', 'blue'],
            'energetic': ['energetic', 'upbeat', 'powerful', 'intense', 'dynamic'],
            'calm': ['calm', 'peaceful', 'relaxing', 'tranquil', 'soothing'],
            'angry': ['angry', 'aggressive', 'intense', 'fierce', 'rage'],
            'melancholic': ['melancholic', 'wistful', 'nostalgic', 'pensive', 'reflective'],
            'hopeful': ['hopeful', 'optimistic', 'uplifting', 'inspiring', 'positive'],
            'romantic': ['romantic', 'love', 'passionate', 'intimate', 'tender'],
            'anxious': ['anxious', 'tense', 'nervous', 'restless', 'uneasy'],
            'peaceful': ['peaceful', 'serene', 'calm', 'tranquil', 'gentle'],
        }
        
        emotion_lower = emotion.lower().strip()
        
        # Return predefined keywords if available
        if emotion_lower in emotion_keyword_map:
            return emotion_keyword_map[emotion_lower]
        
        # For unknown emotions, return the emotion itself plus some generic variations
        return [emotion, emotion_lower, f"{emotion} music", f"{emotion} song"][:5]
    
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
            
            playlist.append(song_result)
        
        logger.info(f"Generated playlist with {len(playlist)} songs")
        return playlist
    
    def _enrich_with_genius_data(
        self, 
        playlist: List[SongResult], 
        target_emotion: Optional[str] = None,
        filter_threshold: float = 0.0,
        seed_songs: Optional[List[SongInput]] = None,
        seed_artists: Optional[List[ArtistInput]] = None
    ) -> List[SongResult]:
        """
        Enrich playlist by comparing actual song lyrics content.
        
        Two modes:
        1. Song/Artist-based: Compare candidate lyrics to SEED song lyrics
        2. Mood-based: Compare candidate lyrics to each other to find cohesive theme
        
        Args:
            playlist: List of SongResult objects (candidates)
            target_emotion: Emotion for mood-based search
            seed_songs: Seed songs for song-based search
            seed_artists: Seed artists for artist-based search
            
        Returns:
            Re-ranked playlist based on lyrical similarity
        """
        if not self.genius_service or not self.genius_service.is_available():
            logger.info("Genius service not available, using title-only embeddings")
            return playlist
        
        try:
            # Determine if this is song/artist-based or mood-based search
            has_seeds = (seed_songs and len(seed_songs) > 0) or (seed_artists and len(seed_artists) > 0)
            
            if has_seeds:
                # Song/Artist-based: Compare to seed lyrics
                return self._enrich_with_seed_lyrics(playlist, seed_songs, seed_artists)
            else:
                # Mood-based: Compare candidates to emotion target
                return self._enrich_with_mood_lyrics(playlist, target_emotion)
            
        except Exception as e:
            logger.warning(f"Could not enrich with lyrics: {e}", exc_info=True)
        
        return playlist
    
    def _enrich_with_seed_lyrics(
        self,
        playlist: List[SongResult],
        seed_songs: Optional[List[SongInput]],
        seed_artists: Optional[List[ArtistInput]]
    ) -> List[SongResult]:
        """Compare candidate lyrics to SEED song lyrics."""
        
        # First, get lyrics for seed songs
        seed_tuples = []
        if seed_songs:
            seed_tuples.extend([(s.song_name, s.artist) for s in seed_songs])
        
        # For artists, get their top tracks
        if seed_artists and self.spotify_service:
            for artist in seed_artists:
                artist_id = artist.spotify_id
                artist_name = artist.artist_name
                if not artist_id:
                    artist_results = self.spotify_service.search_artist(artist.artist_name, limit=1)
                    if artist_results:
                        artist_id = artist_results[0]['spotify_id']
                        artist_name = artist_results[0]['name']
                
                if artist_id:
                    artist_tracks = self.spotify_service.get_artist_tracks_including_collabs(
                        artist_id, artist_name, limit=3
                    )
                    for track in artist_tracks[:2]:
                        seed_tuples.append((track['song_name'], track['artist']))
        
        if not seed_tuples:
            logger.info("No seed songs to compare lyrics against")
            return playlist
        
        logger.info(f"Fetching lyrics for {len(seed_tuples)} SEED songs...")
        seed_genius_results = self.genius_service.batch_get_lyrics_sync(
            seed_tuples,
            max_concurrent=5
        )
        
        # Create embeddings from seed lyrics
        seed_lyrics_embeddings = []
        for song_name, artist in seed_tuples:
            key = f"{song_name}|{artist}"
            if key in seed_genius_results and seed_genius_results[key]:
                lyrics = seed_genius_results[key].get('lyrics')
                if lyrics:
                    lyrics_emb = self.embedding_service.encode_text(lyrics[:2000])
                    seed_lyrics_embeddings.append(lyrics_emb)
        
        if not seed_lyrics_embeddings:
            logger.warning(
                f"⚠️  Could not fetch lyrics for ANY seed songs! "
                f"Tried: {[f'{s}|{a}' for s, a in seed_tuples]}"
            )
            logger.warning("Falling back to title-based matching only")
            return playlist
        
        # Create target profile from seed lyrics
        target_profile = np.mean(seed_lyrics_embeddings, axis=0)
        target_profile = target_profile / np.linalg.norm(target_profile)
        
        logger.info(f"Created target profile from {len(seed_lyrics_embeddings)} seed song lyrics")
        
        # Now get lyrics for candidate songs
        candidate_tuples = [(song.song_name, song.artist) for song in playlist]
        logger.info(f"Fetching lyrics for {len(candidate_tuples)} candidate songs...")
        
        candidate_genius_results = self.genius_service.batch_get_lyrics_sync(
            candidate_tuples,
            max_concurrent=5
        )
        
        # Re-score candidates based on lyrics similarity to seed profile
        lyrics_scored = 0
        no_lyrics_count = 0
        
        for song in playlist:
            key = f"{song.song_name}|{song.artist}"
            if key in candidate_genius_results and candidate_genius_results[key]:
                genius_data = candidate_genius_results[key]
                
                song.genius_url = genius_data.get('genius_url')
                lyrics = genius_data.get('lyrics')
                
                if lyrics:
                    original_score = song.similarity_score
                    
                    # Compare candidate lyrics to seed profile
                    candidate_lyrics_emb = self.embedding_service.encode_text(lyrics[:2000])
                    lyrics_similarity = float(
                        self.embedding_service.compute_similarity(
                            target_profile,
                            candidate_lyrics_emb
                        )
                    )
                    
                    # Blend: 80% lyrics content, 20% original
                    song.similarity_score = lyrics_similarity * 0.8 + original_score * 0.2
                    lyrics_scored += 1
                    
                    logger.debug(
                        f"✓ {song.song_name}: title={original_score:.3f}, "
                        f"lyrics_vs_seeds={lyrics_similarity:.3f}, "
                        f"blended={song.similarity_score:.3f}"
                    )
                else:
                    no_lyrics_count += 1
            else:
                no_lyrics_count += 1
        
        # Re-sort by new scores
        playlist.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(
            f"✓ Re-ranked by comparing to SEED lyrics: {lyrics_scored}/{len(candidate_tuples)} "
            f"candidates had lyrics ({no_lyrics_count} had no lyrics), "
            f"using 80% lyrics + 20% title weighting"
        )
        
        if lyrics_scored == 0:
            logger.warning("⚠️  No candidates had lyrics! Keeping title-based ranking only")
        
        return playlist
    
    def _enrich_with_mood_lyrics(
        self, 
        playlist: List[SongResult],
        target_emotion: Optional[str] = None
    ) -> List[SongResult]:
        """
        Compare candidate lyrics semantically to the target emotion using advanced analysis.
        
        Now uses:
        1. Multithreaded lyrics fetching for speed
        2. Deeper semantic prompts for better matching
        3. LLM emotion service for nuanced understanding (if available)
        """
        
        songs_to_search = [(song.song_name, song.artist) for song in playlist]
        
        logger.info(f"🎭 Fetching lyrics for {len(songs_to_search)} songs (multithreaded) for deep emotional analysis...")
        
        # Use higher concurrency for faster processing
        genius_results = self.genius_service.batch_get_lyrics_sync(
            songs_to_search,
            max_concurrent=10  # Increased from 5 for speed
        )
        
        # Store lyrics-based embeddings
        lyrics_embeddings = {}
        lyrics_found = 0
        
        for song in playlist:
            key = f"{song.song_name}|{song.artist}"
            if key in genius_results and genius_results[key]:
                genius_data = genius_results[key]
                
                song.genius_url = genius_data.get('genius_url')
                lyrics = genius_data.get('lyrics')
                
                if lyrics:
                    # Create embedding from lyrics content only
                    lyrics_embedding = self.embedding_service.encode_text(lyrics[:2000])
                    lyrics_embeddings[key] = lyrics_embedding
                    lyrics_found += 1
        
        if not lyrics_embeddings:
            logger.warning("⚠️  No lyrics found for mood search, keeping original ranking")
            return playlist
        
        logger.info(f"✓ Found lyrics for {lyrics_found}/{len(songs_to_search)} songs")
        
        # Create semantic target based on the emotion
        if target_emotion:
            # DEEPER SEMANTIC PROMPT with multi-faceted emotional understanding
            emotion_expansions = {
                'angry': 'rage fury aggressive hostile resentful irritated frustrated mad intense forceful confrontational rebellious defiant fierce bitter violent explosive wrathful indignant furious livid enraged incensed outraged',
                'sad': 'sorrow grief despair melancholy depressed heartbroken tearful mournful downcast miserable gloomy dejected sorrowful anguish devastated despondent forlorn woeful disconsolate desolate crying weeping lamenting',
                'happy': 'joyful cheerful delighted pleased content elated upbeat excited positive optimistic bright enthusiastic gleeful ecstatic jubilant merry blissful radiant exuberant thrilled overjoyed buoyant sunny',
                'melancholic': 'wistful nostalgic bittersweet longing reflective pensive somber contemplative moody brooding melancholy sorrowful yearning reminiscent sentimental mournful elegiac plaintive rueful regretful',
                'anxious': 'worried nervous tense uneasy restless fearful stressed troubled apprehensive uncertain edgy panicked unsettled jittery agitated paranoid frantic desperate overwhelmed distressed',
                'calm': 'peaceful serene tranquil relaxed soothing gentle quiet still meditative placid mellow restful composed zen harmonious balanced centered grounded subdued soft ethereal',
                'energetic': 'dynamic vibrant lively powerful intense active vigorous spirited animated charged explosive electric wild frenzied pulsing driving forceful adrenaline pumping',
                'romantic': 'loving tender intimate affectionate passionate devoted yearning longing sweet sensual desire amorous enchanted infatuated enamored ardent fond doting adoring',
                'nostalgic': 'wistful reminiscent sentimental longing memories past reflective bittersweet yearning remembering bygone yesteryear throwback retrospective reminiscing recollection memoir',
                'hopeful': 'optimistic aspiring uplifting encouraging positive inspiring bright promising expectant confident reassuring heartening buoyant anticipatory forward-looking believing faithful',
                'lonely': 'isolated alone abandoned empty solitary distant separated disconnected forlorn desolate forsaken deserted friendless lonesome withdrawn alienated estranged solitary',
                'euphoric': 'ecstatic blissful elated rapturous thrilled exhilarated overjoyed jubilant transcendent elevated intoxicated high flying heavenly celestial sublime divine',
            }
            
            # Get expansion for this emotion (or use the emotion itself)
            emotion_lower = target_emotion.lower().strip()
            expanded_terms = emotion_expansions.get(emotion_lower, target_emotion)
            
            # Create DEEP multi-layered semantic context
            emotion_context = f"""
            Emotional theme: {target_emotion}. 
            Related feelings and moods: {expanded_terms}.
            Song lyrics expressing {target_emotion} emotions: {expanded_terms}.
            Musical themes conveying {target_emotion} sentiments and atmosphere.
            The emotional landscape of {target_emotion}: {expanded_terms}.
            Lyrical content that captures the essence of feeling {target_emotion}.
            Words, imagery, and metaphors associated with {target_emotion}: {expanded_terms}.
            The deep emotional resonance and psychological state of {target_emotion}.
            """
            
            emotion_embedding = self.embedding_service.encode_text(emotion_context)
            # normalize
            emotion_embedding = emotion_embedding / (np.linalg.norm(emotion_embedding) + 1e-9)
            
            logger.info(f"🎯 Created deep semantic target for '{target_emotion}' with {len(expanded_terms.split())} expanded terms")
            
            # Also create a collective mood profile for secondary ranking
            all_lyrics_embs = list(lyrics_embeddings.values())
            collective_profile = np.mean(all_lyrics_embs, axis=0)
            collective_profile = collective_profile / np.linalg.norm(collective_profile)
            
            # If LLM emotion service is available, use it for additional scoring
            use_llm = (
                hasattr(self, 'has_llm_emotions') and 
                self.has_llm_emotions and
                hasattr(self.emotion_mapper, 'llm_emotion_service')
            )
            
            if use_llm:
                logger.info("🤖 Using LLM emotion service for enhanced emotional understanding")
            
            # Re-score based on BOTH emotion target AND collective coherence
            scored_count = 0
            for song in playlist:
                key = f"{song.song_name}|{song.artist}"
                if key in lyrics_embeddings:
                    original_score = song.similarity_score
                    
                    # Normalize candidate lyrics embedding
                    try:
                        candidate_emb = lyrics_embeddings[key]
                        candidate_emb = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-9)
                    except Exception:
                        candidate_emb = lyrics_embeddings[key]

                    # Compare lyrics to emotion target (embedding-based)
                    emotion_similarity = float(
                        self.embedding_service.compute_similarity(
                            emotion_embedding,
                            candidate_emb
                        )
                    )

                    # Compare lyrics to collective mood (for coherence)
                    collective_similarity = float(
                        self.embedding_service.compute_similarity(
                            collective_profile,
                            candidate_emb
                        )
                    )

                    # LLM-based emotional nuance scoring (if available)
                    llm_similarity = 0.0
                    if use_llm:
                        try:
                            llm_service = self.emotion_mapper.llm_emotion_service
                            lyrics_text = genius_results[key].get('lyrics', '')[:1000]
                            llm_similarity = llm_service.compute_emotion_similarity(
                                lyrics_text,
                                target_emotion,
                                context="lyrics"
                            )
                        except Exception as e:
                            logger.debug(f"LLM scoring failed: {e}")
                            llm_similarity = 0.0

                    # Weighted combination: prioritize lyrics heavily when available
                    if use_llm and llm_similarity > 0:
                        song.similarity_score = (
                            emotion_similarity * 0.50 +
                            llm_similarity * 0.30 +
                            collective_similarity * 0.10 +
                            original_score * 0.10
                        )
                    else:
                        song.similarity_score = (
                            emotion_similarity * 0.70 +
                            collective_similarity * 0.15 +
                            original_score * 0.15
                        )

                    scored_count += 1
                    logger.debug(
                        f"✓ {song.song_name}: emotion={emotion_similarity:.3f}, "
                        f"llm={llm_similarity:.3f}, coherence={collective_similarity:.3f}, "
                        f"final={song.similarity_score:.3f}"
                    )
        else:
            # No emotion specified - use collective coherence only
            all_lyrics_embs = list(lyrics_embeddings.values())
            target_lyrics_profile = np.mean(all_lyrics_embs, axis=0)
            target_lyrics_profile = target_lyrics_profile / np.linalg.norm(target_lyrics_profile)
            
            logger.info(f"Created collective mood profile from {len(all_lyrics_embs)} songs")
            
            scored_count = 0
            for song in playlist:
                key = f"{song.song_name}|{song.artist}"
                if key in lyrics_embeddings:
                    original_score = song.similarity_score
                    
                    lyrics_similarity = float(
                        self.embedding_service.compute_similarity(
                            target_lyrics_profile,
                            lyrics_embeddings[key]
                        )
                    )
                    
                    song.similarity_score = lyrics_similarity * 0.80 + original_score * 0.20
                    scored_count += 1
        
        # Re-sort by new scores
        playlist.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(
            f"✓ Re-ranked by lyrical semantic analysis: {scored_count}/{len(songs_to_search)} songs analyzed"
        )
        
        return playlist
    
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
