#!/usr/bin/env python3
"""Test emotion-based playlist generation"""

import asyncio
import logging
from backend.services.embedding_service import EmbeddingService
from backend.services.emotion_mapper import EmotionMapper
from backend.services.spotify_service import SpotifyService
from backend.services.playlist_generator import PlaylistGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing services...")
    embedding_service = EmbeddingService()
    emotion_mapper = EmotionMapper()
    spotify_service = SpotifyService()
    
    if not spotify_service.is_available():
        logger.error("Spotify service not available")
        return
    
    playlist_generator = PlaylistGenerator(
        embedding_service=embedding_service,
        emotion_mapper=emotion_mapper,
        spotify_service=spotify_service
    )
    
    logger.info("Generating playlist for 'happy' emotion...")
    playlist, combined_embedding, emotion_features = playlist_generator.generate_playlist(
        songs=None,
        emotion=["happy"],
        num_results=5
    )
    
    logger.info(f"\nGenerated playlist with {len(playlist)} songs:")
    for i, song in enumerate(playlist, 1):
        print(f"{i}. {song.song_name} - {song.artist} (score: {song.similarity_score:.3f})")

if __name__ == "__main__":
    main()
