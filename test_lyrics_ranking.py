#!/usr/bin/env python3
"""Test lyrics-based emotional ranking"""

import logging
from backend.services.embedding_service import EmbeddingService
from backend.services.emotion_mapper import EmotionMapper
from backend.services.spotify_service import SpotifyService
from backend.services.async_genius_service import AsyncGeniusService
from backend.services.playlist_generator import PlaylistGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing services...")
    embedding_service = EmbeddingService()
    emotion_mapper = EmotionMapper()
    spotify_service = SpotifyService()
    genius_service = AsyncGeniusService()
    
    logger.info(f"Spotify available: {spotify_service.is_available()}")
    logger.info(f"Genius available: {genius_service.is_available()}")
    
    playlist_generator = PlaylistGenerator(
        embedding_service=embedding_service,
        emotion_mapper=emotion_mapper,
        spotify_service=spotify_service,
        genius_service=genius_service
    )
    
    print("\n" + "="*60)
    print("TEST 1: WITHOUT Lyrics Enrichment (Embeddings Only)")
    print("="*60)
    playlist1, _, _ = playlist_generator.generate_playlist(
        songs=None,
        emotion=["sad"],
        num_results=5,
        enrich_with_lyrics=False
    )
    
    print(f"\nGenerated {len(playlist1)} songs (embeddings-only):")
    for i, song in enumerate(playlist1, 1):
        print(f"{i}. {song.song_name} - {song.artist}")
        print(f"   Score: {song.similarity_score:.3f}")
    
    print("\n" + "="*60)
    print("TEST 2: WITH Lyrics Enrichment (80% Lyrics + 20% Embeddings)")
    print("="*60)
    
    if genius_service.is_available():
        playlist2, _, _ = playlist_generator.generate_playlist(
            songs=None,
            emotion=["sad"],
            num_results=5,
            enrich_with_lyrics=True
        )
        
        print(f"\nGenerated {len(playlist2)} songs (lyrics-heavy):")
        for i, song in enumerate(playlist2, 1):
            print(f"{i}. {song.song_name} - {song.artist}")
            print(f"   Score: {song.similarity_score:.3f}")
            if hasattr(song, 'genius_url') and song.genius_url:
                print(f"   Genius: {song.genius_url}")
    else:
        print("\nGenius service not available - add GENIUS_ACCESS_TOKEN to .env")

if __name__ == "__main__":
    main()
