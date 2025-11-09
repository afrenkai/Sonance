import asyncio
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv
import aiohttp

logger = logging.getLogger(__name__)
load_dotenv()


class AsyncGeniusService:
    
    BASE_URL = "https://api.genius.com"
    
    def __init__(self):
        self.access_token = os.getenv('GENIUS_ACCESS_TOKEN')
        if not self.access_token:
            logger.warning("Genius access token not found")
            
    def is_available(self) -> bool:
        return self.access_token is not None
    
    async def get_song_by_id(self, song_id: int) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            logger.warning("Genius service not available - missing API token")
            return None
        
        url = f"{self.BASE_URL}/songs/{song_id}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', {}).get('song')
                    else:
                        logger.error(f"Genius API error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching song {song_id} from Genius API: {e}")
            return None
    
    async def search_song(self, song_name: str, artist: str = None) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            logger.warning("Genius service not available - missing API token")
            return None
        
        url = f"{self.BASE_URL}/search"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"q": f"{song_name} {artist}" if artist else song_name}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        hits = data.get('response', {}).get('hits', [])
                        
                        if not hits:
                            logger.debug(f"No results found for: {song_name} by {artist}")
                            return None
                        
                        song_id = hits[0]['result']['id']
                        return await self.get_song_by_id(song_id)
                    else:
                        logger.error(f"Genius API search error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error searching for song on Genius API: {e}")
            return None
    
    async def get_song_lyrics(
        self,
        song_name: str,
        artist: str
    ) -> Optional[Dict[str, Any]]:
        if not self.is_available():
            logger.warning("Genius service not available - missing API token")
            return None
        
        song_data = await self.search_song(song_name, artist)
        
        if not song_data:
            logger.debug(f"Song not found: {song_name} by {artist}")
            return None
        
        try:
            import lyricsgenius
        except ImportError:
            logger.error("lyricsgenius not installed. Install with: pip install lyricsgenius")
            return {
                'song_id': song_data.get('id'),
                'title': song_data.get('title'),
                'artist': song_data.get('primary_artist', {}).get('name'),
                'lyrics': None,
                'genius_url': song_data.get('url'),
                'album': song_data.get('album', {}).get('name') if song_data.get('album') else None,
                'release_date': song_data.get('release_date_for_display'),
                'media': song_data.get('media'),
                'song_art_image_url': song_data.get('song_art_image_url'),
                'header_image_url': song_data.get('header_image_url')
            }
        
        try:
            genius = lyricsgenius.Genius(
                self.access_token,
                timeout=5,
                retries=1,
                verbose=False,
                remove_section_headers=True
            )
            
            loop = asyncio.get_event_loop()
            song = await loop.run_in_executor(
                None,
                lambda: genius.search_song(song_name, artist)
            )
            
            lyrics = song.lyrics if song else None
            
            return {
                'song_id': song_data.get('id'),
                'title': song_data.get('title'),
                'artist': song_data.get('primary_artist', {}).get('name'),
                'lyrics': lyrics,
                'genius_url': song_data.get('url'),
                'album': song_data.get('album', {}).get('name') if song_data.get('album') else None,
                'release_date': song_data.get('release_date_for_display'),
                'media': song_data.get('media'),
                'song_art_image_url': song_data.get('song_art_image_url'),
                'header_image_url': song_data.get('header_image_url')
            }
            
        except Exception as e:
            logger.error(f"Error fetching lyrics for {song_name} by {artist}: {e}")
            return None
    
    def get_song_lyrics_sync(
        self,
        song_name: str,
        artist: str
    ) -> Optional[Dict[str, Any]]:
   
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_song_lyrics(song_name, artist)
        )
    
    async def batch_get_lyrics(
        self,
        songs: List[Tuple[str, str]],
        max_concurrent: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        if not self.is_available():
            return {}
        
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_with_semaphore(song_name: str, artist: str):
            async with semaphore:
                return await self.get_song_lyrics(song_name, artist)
        
        tasks = [
            get_with_semaphore(song_name, artist)
            for song_name, artist in songs
        ]
        
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0
            )
            
            for (song_name, artist), response in zip(songs, responses):
                if isinstance(response, dict) and response:
                    key = f"{song_name}|{artist}"
                    results[key] = response
                    
        except asyncio.TimeoutError:
            logger.warning("Batch lyrics fetch timed out")
        
        logger.info(f"Fetched lyrics for {len(results)}/{len(songs)} songs")
        return results
    
    def batch_get_lyrics_sync(
        self,
        songs: List[Tuple[str, str]],
        max_concurrent: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.batch_get_lyrics(songs, max_concurrent)
        )


# if __name__ == "__main__":
#     async def test_song():
#         service = AsyncGeniusService()
#
#         if not service.is_available():
#             print("Genius API token not configured")
#             return
#
#         song_name = "Chandelier"
#         artist = "Sia"
#
#         print(f"Fetching data for '{song_name}' by {artist}...\n")
#         result = await service.get_song_lyrics(song_name, artist)
#
#         if result:
#             print(f" Found: {result['title']} by {result['artist']}")
#             print(f"  Song ID: {result.get('song_id')}")
#             print(f"  Genius URL: {result['genius_url']}")
#             print(f"  Album: {result.get('album', 'N/A')}")
#             print(f"  Release date: {result.get('release_date', 'N/A')}")
#             print(f"  Song art: {result.get('song_art_image_url', 'N/A')}")
#             print(f"\n  Media:")
#             if result.get('media'):
#                 for media_item in result['media']:
#                     print(f"    - Provider: {media_item.get('provider')}")
#                     print(f"      Type: {media_item.get('type')}")
#                     print(f"      URL: {media_item.get('url')}")
#             else:
#                 print("    No media found")
#
#             if result.get('lyrics'):
#                 print(f"\n  Lyrics length: {len(result['lyrics'])} characters")
#         else:
#             print("Song not found")
#
#     asyncio.run(test_song())
