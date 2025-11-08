import aiohttp
import asyncio
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)
load_dotenv()


class AsyncGeniusService:
    
    def __init__(self):
        self.access_token = os.getenv('GENIUS_ACCESS_TOKEN')
        self.base_url = "https://api.genius.com"
        self.cache = {}  
        
        if not self.access_token:
            logger.warning("Genius access token not found")
            
    def is_available(self) -> bool:
        return self.access_token is not None
    
    def _get_cache_key(self, song_name: str, artist: str) -> str:
        return hashlib.md5(f"{song_name}|{artist}".encode()).hexdigest()
    
    async def search_song_async(
        self, 
        session: aiohttp.ClientSession,
        song_name: str, 
        artist: str
    ) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            return None

        cache_key = self._get_cache_key(song_name, artist)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            query = f"{song_name} {artist}"
            url = f"{self.base_url}/search"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {"q": query}
            
            async with session.get(
                url, 
                headers=headers, 
                params=params,
                timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                hits = data.get('response', {}).get('hits', [])
                
                if not hits:
                    return None
                

                result = hits[0].get('result', {})
                song_data = {
                    'title': result.get('title'),
                    'artist': result.get('primary_artist', {}).get('name'),
                    'url': result.get('url'),
                    'song_id': result.get('id')
                }
                
                self.cache[cache_key] = song_data
                return song_data
                
        except Exception as e:
            logger.debug(f"Error searching Genius: {e}")
            return None
    
    def extract_emotional_keywords(self, lyrics: str) -> Dict[str, int]:
        if not lyrics:
            return {}
        
        emotion_keywords = {
            'happy': ['happy', 'joy', 'smile', 'laugh', 'celebrate', 'bright', 'sunshine', 
                      'good', 'wonderful', 'amazing', 'fantastic', 'cheerful', 'delight',
                      'fun', 'party', 'dancing', 'excited', 'glad', 'blessed'],
            'sad': ['sad', 'cry', 'tear', 'lonely', 'heartbreak', 'miss', 'lost', 'pain',
                    'hurt', 'broken', 'empty', 'alone', 'sorrow', 'grief', 'blue', 'down',
                    'depressed', 'misery', 'suffering', 'ache'],
            'love': ['love', 'heart', 'together', 'forever', 'kiss', 'embrace', 'darling',
                     'baby', 'dear', 'sweet', 'romance', 'passion', 'desire', 'need',
                     'want', 'adore', 'cherish', 'devotion'],
            'angry': ['angry', 'rage', 'hate', 'fight', 'scream', 'mad', 'burn',
                      'fury', 'violent', 'destroy', 'break', 'smash', 'kill', 'blood',
                      'war', 'enemy', 'revenge', 'fire'],
            'hopeful': ['hope', 'dream', 'believe', 'faith', 'future', 'tomorrow', 'rise',
                        'better', 'new', 'change', 'light', 'shine', 'star', 'wish',
                        'possible', 'believe', 'trust', 'prayer'],
            'nostalgic': ['remember', 'memory', 'past', 'yesterday', 'used to', 'back when',
                          'once', 'before', 'old', 'time', 'moment', 'ago', 'reminisce',
                          'recall', 'forgotten', 'history'],
            'energetic': ['run', 'dance', 'move', 'jump', 'wild', 'alive', 'fire',
                          'fast', 'quick', 'rush', 'power', 'energy', 'strong', 'loud',
                          'intense', 'explosive', 'electric', 'pumped'],
            'calm': ['calm', 'peace', 'quiet', 'still', 'gentle', 'soft', 'breathe',
                     'slow', 'rest', 'relax', 'tranquil', 'serene', 'silent', 'sleep',
                     'whisper', 'soothe', 'ease'],
            'melancholic': ['melancholy', 'wistful', 'bittersweet', 'longing', 'yearning',
                            'regret', 'lament', 'mourn', 'fade', 'dusk', 'autumn', 'rain'],
            'romantic': ['romantic', 'lover', 'intimate', 'tender', 'gentle', 'close',
                         'touch', 'hold', 'warm', 'soft', 'beautiful', 'eyes'],
            'anxious': ['anxious', 'worry', 'fear', 'scared', 'nervous', 'panic', 'stress',
                        'tension', 'pressure', 'uncertain', 'doubt', 'restless', 'uneasy'],
            'peaceful': ['peaceful', 'harmony', 'balance', 'zen', 'meditation', 'nature',
                         'ocean', 'breeze', 'sunset', 'morning', 'stillness']
        }
        
        lyrics_lower = lyrics.lower()
        scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(lyrics_lower.count(word) for word in keywords)
            if count > 0:
                scores[emotion] = count
        
        return scores
    
    def compute_emotion_match_score(
        self, 
        emotional_keywords: Dict[str, int],
        target_emotion: str
    ) -> float:
        if not emotional_keywords:
            return 0.0
        
        target_emotion = target_emotion.lower().strip()
        
        if target_emotion in emotional_keywords:
            direct_score = emotional_keywords[target_emotion]
            total_keywords = sum(emotional_keywords.values())
            
            normalized = direct_score / max(total_keywords, 1)
            return min(1.0, normalized * 2.0)  

        related_emotions = {
            'happy': ['love', 'hopeful', 'energetic'],
            'sad': ['melancholic', 'nostalgic', 'anxious'],
            'energetic': ['happy', 'angry'],
            'calm': ['peaceful', 'hopeful'],
            'angry': ['energetic', 'anxious'],
            'melancholic': ['sad', 'nostalgic'],
            'hopeful': ['happy', 'peaceful'],
            'romantic': ['love', 'happy', 'peaceful'],
            'anxious': ['sad', 'angry'],
            'peaceful': ['calm', 'hopeful']
        }

        if target_emotion in related_emotions:
            related_score = 0.0
            for related in related_emotions[target_emotion]:
                if related in emotional_keywords:
                    related_score += emotional_keywords[related] * 0.3  
            
            total_keywords = sum(emotional_keywords.values())
            if total_keywords > 0:
                return min(0.7, related_score / total_keywords)  
        
        return 0.0
    
    async def get_song_with_emotional_profile(
        self,
        session: aiohttp.ClientSession,
        song_name: str,
        artist: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get song metadata with emotional profile from Genius.
        
        NOTE: Uses lyricsgenius library in background for actual lyrics extraction.
        This method only returns emotional analysis, not lyrics content.
        
        Args:
            session: aiohttp client session
            song_name: Name of the song
            artist: Artist name
            
        Returns:
            Dict with emotional profile or None
        """
        if not self.is_available():
            return None
        
        cache_key = self._get_cache_key(song_name, artist)
        if cache_key in self.cache:
            cached = self.cache.get(cache_key)
            if cached and 'emotional_profile' in cached:
                return cached
        
        try:
            song_data = await self.search_song_async(session, song_name, artist)
            if not song_data:
                return None
            
            # For now, return basic data
            # In production, you'd use lyricsgenius library here to get lyrics
            # and compute emotional profile
            return song_data
            
        except Exception as e:
            logger.debug(f"Error getting emotional profile: {e}")
            return None
    
    async def get_lyrics_async(
        self,
        session: aiohttp.ClientSession,
        song_url: str
    ) -> Optional[str]:
        """
        Get lyrics for a song from Genius web scraping.
        
        Note: This requires web scraping which may be unreliable.
        For production, consider using lyricsgenius library.
        
        Args:
            session: aiohttp client session
            song_url: Genius song URL
            
        Returns:
            Lyrics text or None
        """
        # This would require web scraping the Genius website
        # For now, return None to keep it simple
        # In production, use lyricsgenius library or similar
        return None
    
    async def batch_get_emotional_profiles(
        self,
        songs: List[Tuple[str, str]],
        target_emotion: Optional[str] = None,
        max_concurrent: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get emotional profiles for multiple songs concurrently.
        Uses lyricsgenius in thread pool to avoid blocking.
        
        Args:
            songs: List of (song_name, artist) tuples
            target_emotion: Optional target emotion for scoring
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping "song_name|artist" to emotional profile with scores
        """
        if not self.is_available():
            return {}
        
        # Import lyricsgenius here to avoid issues if not installed
        try:
            import lyricsgenius
        except ImportError:
            logger.warning("lyricsgenius not installed, skipping lyrics analysis")
            return {}
        
        results = {}
        
        # Initialize genius client
        genius = lyricsgenius.Genius(
            self.access_token,
            timeout=5,
            retries=1,
            sleep_time=0.2,
            verbose=False,
            remove_section_headers=True
        )
        
        async def get_profile_for_song(song_name: str, artist: str):
            """Get emotional profile for a single song."""
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                song = await loop.run_in_executor(
                    None, 
                    lambda: genius.search_song(song_name, artist)
                )
                
                if not song or not song.lyrics:
                    return None
                
                # Extract emotional keywords (not storing lyrics)
                emotional_keywords = self.extract_emotional_keywords(song.lyrics)
                
                # Compute emotion match score if target provided
                emotion_score = 0.0
                if target_emotion and emotional_keywords:
                    emotion_score = self.compute_emotion_match_score(
                        emotional_keywords,
                        target_emotion
                    )
                
                # Determine dominant emotion
                dominant_emotion = None
                if emotional_keywords:
                    dominant_emotion = max(emotional_keywords.items(), key=lambda x: x[1])[0]
                
                return {
                    'song_id': song.id,
                    'genius_url': song.url,
                    'emotional_keywords': emotional_keywords,
                    'dominant_emotion': dominant_emotion,
                    'emotion_match_score': emotion_score,
                    'word_count': len(song.lyrics.split()),
                    'has_lyrics': True
                }
                
            except Exception as e:
                logger.debug(f"Could not get profile for {song_name}: {e}")
                return None
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_with_semaphore(song_name: str, artist: str):
            async with semaphore:
                return await get_profile_for_song(song_name, artist)
        
        # Process all songs
        tasks = [get_with_semaphore(song_name, artist) for song_name, artist in songs]
        
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=15.0  # 15 second total timeout
            )
            
            for (song_name, artist), response in zip(songs, responses):
                if isinstance(response, dict) and response:
                    key = f"{song_name}|{artist}"
                    results[key] = response
                    # Cache the result
                    cache_key = self._get_cache_key(song_name, artist)
                    self.cache[cache_key] = response
                    
        except asyncio.TimeoutError:
            logger.warning("Batch emotional profile extraction timed out")
        
        logger.info(f"Got emotional profiles for {len(results)}/{len(songs)} songs")
        return results
    
    def batch_get_emotional_profiles_sync(
        self,
        songs: List[Tuple[str, str]],
        target_emotion: Optional[str] = None,
        max_concurrent: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Synchronous wrapper for batch_get_emotional_profiles.
        
        Args:
            songs: List of (song_name, artist) tuples
            target_emotion: Optional target emotion for scoring
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping "song_name|artist" to emotional profile
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.batch_get_emotional_profiles(songs, target_emotion, max_concurrent)
        )
    
    async def batch_search_songs(
        self,
        songs: List[Tuple[str, str]],
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Search for multiple songs concurrently.
        
        Args:
            songs: List of (song_name, artist) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping "song_name|artist" to song data
        """
        if not self.is_available():
            return {}
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def search_with_semaphore(song_name: str, artist: str):
                async with semaphore:
                    return await self.search_song_async(session, song_name, artist)
            
            # Create tasks for all songs
            tasks = [
                search_with_semaphore(song_name, artist)
                for song_name, artist in songs
            ]
            
            # Wait for all tasks with timeout
            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=10.0  # 10 second total timeout
                )
                
                # Process results
                for (song_name, artist), response in zip(songs, responses):
                    if isinstance(response, dict):
                        key = f"{song_name}|{artist}"
                        results[key] = response
                        
            except asyncio.TimeoutError:
                logger.warning("Batch search timed out")
        
        return results
    
    def batch_search_songs_sync(
        self,
        songs: List[Tuple[str, str]],
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Synchronous wrapper for batch_search_songs.
        
        Args:
            songs: List of (song_name, artist) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping "song_name|artist" to song data
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.batch_search_songs(songs, max_concurrent)
        )
