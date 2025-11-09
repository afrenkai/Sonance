from fastapi import APIRouter, HTTPException, Request
from typing import Optional, List
import logging
import aiohttp

from backend.models.schemas import (
    PlaylistRequest,
    PlaylistResponse,
    SongResult,
    HealthResponse
)
from backend.services.playlist_generator import PlaylistGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/get-audio")
async def get_audio(request: Request):
    from fastapi.responses import StreamingResponse
    import asyncio

    genius_service = getattr(request.app.state, 'genius_service', None)
    if not genius_service:
        raise HTTPException(status_code=503, detail="Genius service is not available")

    song_title = request.query_params.get("song_title")
    logger.info(f"Requested song_title: {song_title}")
    artist_name = request.query_params.get("artist_name")
    if not song_title or not artist_name:
        raise HTTPException(status_code=400, detail="song_title and artist_name are required parameters")

    async def stream_ytdlp():
        import subprocess
        query = f"ytsearch:{song_title} By {artist_name}"

        cmd = ["yt-dlp", "--get-id", query]
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_id = result.stdout.strip()
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Resolved YouTube URL: {youtube_url}")
        args = [
            "yt-dlp",
            "-f", "bestaudio",
            "-o", "-",
            "--no-playlist",
            "--no-warnings",
            youtube_url
        ]
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            if proc.returncode is None:
                proc.kill()
                proc.wait()

    return StreamingResponse(stream_ytdlp(), media_type="audio/*")

@router.post("/generate-playlist", response_model=PlaylistResponse)
async def generate_playlist(request: PlaylistRequest, fastapi_request: Request):
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
        
        emotion_display = "none"
        if request.emotion:
            if isinstance(request.emotion, list):
                emotion_display = f"[{', '.join(request.emotion)}]"
            else:
                emotion_display = request.emotion
        
        logger.info(
            f"Generating playlist: {len(request.songs or [])} songs, "
            f"{len(request.artists or [])} artists, "
            f"emotion(s): {emotion_display}"
        )
        
        playlist, combined_embedding, emotion_features = playlist_generator.generate_playlist(
            songs=request.songs,
            artists=request.artists,
            emotion=request.emotion,
            num_results=request.num_results,
            enrich_with_lyrics=request.enrich_with_lyrics
        )
        
        emotion_features_list = None
        if emotion_features:
            emotion_features_list = {
                key: (value[0] + value[1]) / 2 if isinstance(value, (tuple, list)) and len(value) == 2 else value
                for key, value in emotion_features.items()
            }
        
        response = PlaylistResponse(
            playlist=playlist,
            emotion_features=emotion_features_list,
            combined_embedding=combined_embedding.tolist()[:10]
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
async def list_emotions(request: Request):
    from backend.models.schemas import EmotionType
    
    predefined = [emotion.value for emotion in EmotionType]
    
    learned = []
    try:
        emotion_mapper = request.app.state.emotion_mapper
        if hasattr(emotion_mapper, 'llm_emotion_service') and emotion_mapper.llm_emotion_service:
            learned = emotion_mapper.llm_emotion_service.get_learned_emotions()
    except Exception as e:
        logger.debug(f"Could not retrieve learned emotions: {e}")
    
    return {
        "predefined_emotions": predefined,
        "learned_emotions": learned,
        "note": "You can use any emotion word - the system will learn its meaning through AI contextual understanding",
        "multi_emotion_support": True,
        "examples": [
            "happy",
            "sad and hopeful",
            "energetic melancholic",
            "nostalgic",
            "bittersweet",
            "euphoric"
        ]
    }


@router.get("/spotify/search")
async def search_spotify_track(
    song_name: str,
    artist: Optional[str] = None,
    limit: int = 10,
    request: Request = None
):
    try:
        spotify_service = request.app.state.spotify_service
        
        if not spotify_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Spotify service is not available. Check credentials."
            )
        
        query = f"track:{song_name}"
        if artist:
            query += f" artist:{artist}"
        
        results = spotify_service.spotify.search(q=query, type='track', limit=min(limit, 50))
        
        if not results['tracks']['items']:
            return {"tracks": []}
        
        tracks = []
        for track in results['tracks']['items']:
            tracks.append(spotify_service._format_track(track))
        
        return {"tracks": tracks}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching Spotify: {e}")
        raise HTTPException(status_code=500, detail="Failed to search Spotify")


@router.get("/spotify/artist/search")
async def search_spotify_artist(
    artist_name: str,
    limit: int = 10,
    request: Request = None
):
    try:
        spotify_service = request.app.state.spotify_service
        
        if not spotify_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Spotify service is not available. Check credentials."
            )
        
        artists = spotify_service.search_artist(artist_name, limit=min(limit, 50))
        
        return {"artists": artists}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching artists on Spotify: {e}")
        raise HTTPException(status_code=500, detail="Failed to search artists")


@router.get("/spotify/artist/{artist_id}/top-tracks")
async def get_artist_top_tracks(
    artist_id: str,
    limit: int = 10,
    request: Request = None
):
    try:
        spotify_service = request.app.state.spotify_service
        
        if not spotify_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Spotify service is not available"
            )
        
        tracks = spotify_service.get_artist_top_tracks(artist_id, limit=limit)
        
        if not tracks:
            raise HTTPException(status_code=404, detail="No tracks found for this artist")
        
        return {"tracks": tracks}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting artist top tracks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get artist top tracks")


@router.get("/spotify/track/{track_id}")
async def get_spotify_track(track_id: str, request: Request):
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



