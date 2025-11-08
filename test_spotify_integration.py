"""
Test script to verify Spotify integration
Run this after setting up .env with Spotify credentials
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.spotify_service import SpotifyService
from backend.services.embedding_service import EmbeddingService
from backend.services.emotion_mapper import EmotionMapper
from backend.services.playlist_generator import PlaylistGenerator
from backend.models.schemas import SongInput

def test_spotify_service():
    print("=" * 60)
    print("Testing Spotify Service Integration")
    print("=" * 60)
    
    # Initialize services
    print("\n1. Initializing services...")
    spotify_service = SpotifyService()
    
    if not spotify_service.is_available():
        print("❌ Spotify service not available. Check your .env file!")
        print("   Make sure you have SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET set.")
        return False
    
    print("✅ Spotify service initialized successfully")
    
    # Test search
    print("\n2. Testing track search...")
    track = spotify_service.search_track("Bohemian Rhapsody", "Queen")
    if track:
        print(f"✅ Found track: {track['song_name']} by {track['artist']}")
        print(f"   Spotify ID: {track['spotify_id']}")
        track_id = track['spotify_id']
    else:
        print("❌ Track not found")
        return False
    
    # Test audio features
    print("\n3. Testing audio features...")
    features = spotify_service.get_audio_features(track_id)
    if features:
        print("✅ Audio features retrieved:")
        print(f"   Valence: {features['valence']:.2f}")
        print(f"   Energy: {features['energy']:.2f}")
        print(f"   Danceability: {features['danceability']:.2f}")
    else:
        print("❌ Could not get audio features")
        return False
    
    # Test recommendations
    print("\n4. Testing recommendations...")
    recommendations = spotify_service.get_recommendations(
        seed_tracks=[track_id],
        limit=5,
        target_valence=0.8,
        target_energy=0.7
    )
    if recommendations:
        print(f"✅ Got {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. {rec['song_name']} by {rec['artist']}")
    else:
        print("❌ Could not get recommendations")
        return False
    
    # Test playlist generator integration
    print("\n5. Testing playlist generator with Spotify...")
    embedding_service = EmbeddingService()
    emotion_mapper = EmotionMapper()
    
    playlist_generator = PlaylistGenerator(
        embedding_service=embedding_service,
        emotion_mapper=emotion_mapper,
        spotify_service=spotify_service
    )
    
    songs = [SongInput(song_name="Imagine", artist="John Lennon")]
    emotion = ["happy"]
    
    try:
        playlist, embedding, emotion_features = playlist_generator.generate_playlist(
            songs=songs,
            emotion=emotion,
            num_results=5
        )
        
        print(f"✅ Generated playlist with {len(playlist)} songs:")
        for i, song in enumerate(playlist, 1):
            print(f"   {i}. {song.song_name} by {song.artist}")
            print(f"      Similarity: {song.similarity_score:.3f}")
            if song.spotify_id:
                print(f"      Spotify ID: {song.spotify_id}")
        
    except Exception as e:
        print(f"❌ Error generating playlist: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Spotify integration is working!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_spotify_service()
    sys.exit(0 if success else 1)
