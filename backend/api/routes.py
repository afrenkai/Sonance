from fastapi import APIRouter, HTTPException, Request
from typing import Optional, List
import logging

from backend.models.schemas import (
    PlaylistRequest,
    PlaylistResponse,
    SongResult,
    MoodCollage,
    HealthResponse
)
from backend.services.playlist_generator import PlaylistGenerator
from backend.services.mood_collage_generator import MoodCollageGenerator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate-playlist", response_model=PlaylistResponse)
async def generate_playlist(request: PlaylistRequest, fastapi_request: Request):
    """
    Generate a playlist based on user input songs and/or emotion.
    
    This endpoint:
    1. Computes embeddings for input songs
    2. Maps emotion to audio feature ranges
    3. Combines embeddings with emotion context
    4. Queries and ranks songs
    5. Optionally generates mood collage image
    """
    try:
        embedding_service = fastapi_request.app.state.embedding_service
        emotion_mapper = fastapi_request.app.state.emotion_mapper
        spotify_service = fastapi_request.app.state.spotify_service
        genius_service = getattr(fastapi_request.app.state, 'genius_service', None)
        
        playlist_generator = PlaylistGenerator(
            embedding_service=embedding_service,
            emotion_mapper=emotion_mapper,
            spotify_service=spotify_service,
            genius_service=genius_service
        )
        
        logger.info(f"Generating playlist: {len(request.songs or [])} songs, emotion: {request.emotion}")
        
        playlist, combined_embedding, emotion_features = playlist_generator.generate_playlist(
            songs=request.songs,
            emotion=request.emotion,
            num_results=request.num_results,
            enrich_with_lyrics=request.enrich_with_lyrics
        )
        
        mood_collage = None
        if request.include_collage:
            logger.info("Generating mood collage")
            collage_generator = MoodCollageGenerator()
            # Convert emotion list to string for collage generation
            emotion_str = " ".join(request.emotion) if request.emotion else None
            image_base64, dominant_colors, visual_params = collage_generator.generate_collage(
                combined_embedding,
                emotion_str
            )
            
            mood_collage = MoodCollage(
                image_base64=image_base64,
                dominant_colors=dominant_colors,
                visual_params=visual_params,
                width=collage_generator.width,
                height=collage_generator.height
            )
        
        emotion_features_list = None
        if emotion_features:
            # Convert tuples (min, max) to midpoint values for display
            emotion_features_list = {
                key: (value[0] + value[1]) / 2 if isinstance(value, (tuple, list)) and len(value) == 2 else value
                for key, value in emotion_features.items()
            }
        
        response = PlaylistResponse(
            playlist=playlist,
            mood_collage=mood_collage,
            emotion_features=emotion_features_list,
            combined_embedding=combined_embedding.tolist()[:10]  # First 10 dims for debugging
        )
        
        logger.info(f"Successfully generated playlist with {len(playlist)} songs")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating playlist: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    try:
        services = {
            "embedding_service": hasattr(request.app.state, 'embedding_service'),
            "emotion_mapper": hasattr(request.app.state, 'emotion_mapper'),
            "spotify_service": (
                hasattr(request.app.state, 'spotify_service') and 
                request.app.state.spotify_service.is_available()
            ),
        }
        
        return HealthResponse(
            status="healthy" if all(services.values()) else "degraded",
            version="0.1.0",
            services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/emotions")
async def list_emotions():
    from backend.models.schemas import EmotionType
    
    return {
        "emotions": [emotion.value for emotion in EmotionType],
        "note": "You can also use custom emotion descriptions"
    }


@router.get("/spotify/search")
async def search_spotify_track(
    song_name: str,
    artist: Optional[str] = None,
    request: Request = None
):
    """Search for a track on Spotify by name and optional artist."""
    try:
        spotify_service = request.app.state.spotify_service
        
        if not spotify_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Spotify service is not available. Check credentials."
            )
        
        track = spotify_service.search_track(song_name, artist, limit=1)
        
        if not track:
            raise HTTPException(
                status_code=404,
                detail=f"Track not found: {song_name} by {artist}"
            )
        
        return track
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching Spotify: {e}")
        raise HTTPException(status_code=500, detail="Failed to search Spotify")


@router.get("/spotify/track/{track_id}")
async def get_spotify_track(track_id: str, request: Request):
    """Get detailed information about a Spotify track."""
    try:
        spotify_service = request.app.state.spotify_service
        
        if not spotify_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Spotify service is not available"
            )
        
        track = spotify_service.get_track_by_id(track_id)
        
        if not track:
            raise HTTPException(status_code=404, detail="Track not found")
        
        return track
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting track: {e}")
        raise HTTPException(status_code=500, detail="Failed to get track")


@router.get("/spotify/audio-features/{track_id}")
async def get_audio_features(track_id: str, request: Request):
    """Get audio features for a Spotify track."""
    try:
        spotify_service = request.app.state.spotify_service
        
        if not spotify_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Spotify service is not available"
            )
        
        features = spotify_service.get_audio_features(track_id)
        
        if not features:
            raise HTTPException(
                status_code=404,
                detail="Audio features not found"
            )
        
        return features
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audio features: {e}")
        raise HTTPException(status_code=500, detail="Failed to get audio features")


@router.post("/spotify/recommendations")
async def get_spotify_recommendations(
    seed_tracks: Optional[List[str]] = None,
    emotion: Optional[str] = None,
    num_results: int = 20,
    request: Request = None
):
    """Get Spotify recommendations based on seed tracks and/or emotion."""
    try:
        spotify_service = request.app.state.spotify_service
        emotion_mapper = request.app.state.emotion_mapper
        
        if not spotify_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Spotify service is not available"
            )
        
        # Get target features from emotion if provided
        target_features = {}
        if emotion:
            emotion_features = emotion_mapper.get_feature_ranges(emotion)
            if emotion_features:
                for key, value in emotion_features.items():
                    if isinstance(value, (tuple, list)) and len(value) == 2:
                        target_features[f'target_{key}'] = (value[0] + value[1]) / 2
                    else:
                        target_features[f'target_{key}'] = value
        
        # Get recommendations
        recommendations = spotify_service.get_recommendations(
            seed_tracks=seed_tracks,
            limit=num_results,
            **target_features
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")
