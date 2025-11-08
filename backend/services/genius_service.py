import lyricsgenius
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import re
from functools import lru_cache

logger = logging.getLogger(__name__)
load_dotenv()


class GeniusService:
    """Service for interacting with Genius API for lyrics data."""
    
    def __init__(self):
        """Initialize Genius client with credentials from environment."""
        try:
            access_token = os.getenv('GENIUS_ACCESS_TOKEN')
            
            if not access_token:
                logger.warning("Genius access token not found in environment variables")
                self.genius = None
                return
            
            self.genius = lyricsgenius.Genius(
                access_token,
                timeout=5,  # 5 second timeout
                retries=1,   # Only retry once
                sleep_time=0.1,  # Minimal delay between requests
                verbose=False,
                remove_section_headers=True
            )
            logger.info("Genius service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Genius service: {e}")
            self.genius = None
    
    def is_available(self) -> bool:
        """Check if Genius service is available."""
        return self.genius is not None
    
    @lru_cache(maxsize=500)
    def search_song(self, song_name: str, artist: str) -> Optional[Dict[str, Any]]:
        """
        Search for a song on Genius (cached).
        
        Args:
            song_name: Name of the song
            artist: Artist name
            
        Returns:
            Song metadata dict or None if not found
        """
        if not self.is_available():
            return None
        
        try:
            # Search for the song
            song = self.genius.search_song(song_name, artist)
            
            if not song:
                logger.debug(f"Song not found on Genius: {song_name} by {artist}")
                return None
            
            return {
                'title': song.title,
                'artist': song.artist,
                'url': song.url,
                'song_id': song.id
            }
            
        except Exception as e:
            logger.debug(f"Error searching Genius: {e}")
            return None
    
    def extract_emotional_keywords(self, lyrics: str) -> Dict[str, int]:
        """
        Extract emotional keywords from lyrics without storing full lyrics.
        
        Args:
            lyrics: Song lyrics text
            
        Returns:
            Dictionary of emotion categories and their frequency
        """
        if not lyrics:
            return {}
        
        # Emotion keyword groups (not copyrighted - just classification words)
        emotion_keywords = {
            'happy': ['happy', 'joy', 'smile', 'laugh', 'celebrate', 'bright', 'sun'],
            'sad': ['sad', 'cry', 'tear', 'lonely', 'heartbreak', 'miss', 'lost'],
            'love': ['love', 'heart', 'together', 'forever', 'kiss', 'embrace'],
            'angry': ['angry', 'rage', 'hate', 'fight', 'scream', 'mad'],
            'hopeful': ['hope', 'dream', 'believe', 'faith', 'future', 'tomorrow'],
            'nostalgic': ['remember', 'memory', 'past', 'yesterday', 'used to'],
        }
        
        lyrics_lower = lyrics.lower()
        scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(lyrics_lower.count(word) for word in keywords)
            if count > 0:
                scores[emotion] = count
        
        return scores
    
    @lru_cache(maxsize=200)
    def get_song_emotional_profile(
        self, 
        song_name: str, 
        artist: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get emotional profile of a song based on lyrics analysis (cached).
        
        NOTE: Only returns emotional analysis, not the actual lyrics content.
        
        Args:
            song_name: Name of the song
            artist: Artist name
            
        Returns:
            Emotional profile dict or None if not available
        """
        if not self.is_available():
            return None
        
        try:
            song = self.genius.search_song(song_name, artist)
            
            if not song or not song.lyrics:
                return None
            
            # Extract emotional themes without storing lyrics
            emotional_keywords = self.extract_emotional_keywords(song.lyrics)
            
            # Get word count as a complexity metric
            word_count = len(song.lyrics.split())
            
            # Determine dominant emotion
            dominant_emotion = None
            if emotional_keywords:
                dominant_emotion = max(emotional_keywords.items(), key=lambda x: x[1])[0]
            
            return {
                'song_id': song.id,
                'genius_url': song.url,
                'emotional_keywords': emotional_keywords,
                'dominant_emotion': dominant_emotion,
                'word_count': word_count,
                'has_lyrics': True
            }
            
        except Exception as e:
            logger.debug(f"Could not get emotional profile: {e}")
            return None
    
    def batch_get_emotional_profiles(
        self,
        songs: list[tuple[str, str]],
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get emotional profiles for multiple songs efficiently.
        
        Args:
            songs: List of (song_name, artist) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary mapping "song_name|artist" to emotional profile
        """
        if not self.is_available():
            return {}
        
        profiles = {}
        
        # Process in batches to avoid rate limiting
        for i, (song_name, artist) in enumerate(songs[:max_concurrent]):
            key = f"{song_name}|{artist}"
            try:
                profile = self.get_song_emotional_profile(song_name, artist)
                if profile:
                    profiles[key] = profile
            except Exception as e:
                logger.debug(f"Error getting profile for {song_name}: {e}")
                continue
        
        return profiles
