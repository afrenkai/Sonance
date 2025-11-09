import asyncio
import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from difflib import SequenceMatcher
from dotenv import load_dotenv
import aiohttp

logger = logging.getLogger(__name__)
load_dotenv()


class AsyncGeniusService:
    
    BASE_URL = "https://api.genius.com"
    
    def __init__(self, use_embeddings: bool = False):
        self.access_token = os.getenv('GENIUS_ACCESS_TOKEN')
        if not self.access_token:
            logger.warning("Genius access token not found")
        
        # Embeddings support (optional)
        self.use_embeddings = use_embeddings
        self.encoder = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embeddings model loaded successfully")
            except ImportError:
                logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
                self.use_embeddings = False
            except Exception as e:
                logger.error(f"Error loading embeddings model: {e}")
                self.use_embeddings = False
            
    def is_available(self) -> bool:
        return self.access_token is not None
    
    def compute_match_score(
        self,
        result: Dict[str, Any],
        query_song: str,
        query_artist: Optional[str] = None
    ) -> float:
        """
        Compute match score between search result and query using string similarity.
        Returns a score between 0 and 1, where 1 is a perfect match.
        """
        result_title = result.get('title', '').lower()
        result_artist = result.get('primary_artist', {}).get('name', '').lower()
        
        query_song_lower = query_song.lower()
        query_artist_lower = query_artist.lower() if query_artist else ''
        
        # Title similarity (main component)
        title_sim = SequenceMatcher(None, result_title, query_song_lower).ratio()
        
        # Artist similarity (if provided)
        artist_sim = 1.0  # Default if no artist provided
        if query_artist:
            artist_sim = SequenceMatcher(None, result_artist, query_artist_lower).ratio()
        
        # Weighted score: title is more important
        score = 0.6 * title_sim + 0.4 * artist_sim
        
        # Boost exact matches
        if result_title == query_song_lower:
            score += 0.2
        if query_artist and result_artist == query_artist_lower:
            score += 0.1
        
        # Ensure score doesn't exceed 1.0
        return min(score, 1.0)
    
    def compute_semantic_similarity(
        self,
        query: str,
        candidate_text: str
    ) -> float:
        """
        Compute semantic similarity using embeddings.
        Returns a score between 0 and 1.
        """
        if not self.use_embeddings or not self.encoder:
            return 0.0
        
        try:
            import numpy as np
            
            query_embedding = self.encoder.encode(query, convert_to_numpy=True)
            candidate_embedding = self.encoder.encode(candidate_text, convert_to_numpy=True)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            
            # Convert from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
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
        """
        Search for a song on Genius API with improved matching.
        Uses match scoring to find the best result from all candidates.
        """
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
                        
                        # Score all candidates and pick the best match
                        best_match = None
                        best_score = 0.0
                        
                        for hit in hits:
                            result = hit.get('result', {})
                            score = self.compute_match_score(result, song_name, artist)
                            
                            logger.debug(f"Candidate: '{result.get('title')}' by '{result.get('primary_artist', {}).get('name')}' - Score: {score:.3f}")
                            
                            if score > best_score:
                                best_score = score
                                best_match = result
                        
                        if best_match:
                            logger.info(f"Best match: '{best_match.get('title')}' by '{best_match.get('primary_artist', {}).get('name')}' (score: {best_score:.3f})")
                            song_id = best_match.get('id')
                            return await self.get_song_by_id(song_id)
                        
                        return None
                    else:
                        logger.error(f"Genius API search error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error searching for song on Genius API: {e}")
            return None
    
    async def semantic_search_song(self, song_name: str, artist: str = None) -> Optional[Dict[str, Any]]:
        """
        Search for a song using embeddings-based semantic similarity.
        Falls back to regular search if embeddings are not available.
        Combines string matching with semantic similarity for best results.
        """
        if not self.is_available():
            logger.warning("Genius service not available - missing API token")
            return None
        
        # Fall back to regular search if embeddings not enabled
        if not self.use_embeddings:
            logger.debug("Embeddings not enabled, using regular search")
            return await self.search_song(song_name, artist)
        
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
                        
                        # Prepare query for embedding
                        query = f"{song_name} {artist}" if artist else song_name
                        
                        # Score all candidates with both methods
                        best_match = None
                        best_combined_score = 0.0
                        
                        for hit in hits:
                            result = hit.get('result', {})
                            
                            # String-based score
                            string_score = self.compute_match_score(result, song_name, artist)
                            
                            # Semantic similarity score
                            candidate_text = f"{result.get('title', '')} {result.get('primary_artist', {}).get('name', '')}"
                            semantic_score = self.compute_semantic_similarity(query, candidate_text)
                            
                            # Combined score (weighted average)
                            combined_score = 0.6 * string_score + 0.4 * semantic_score
                            
                            logger.debug(
                                f"Candidate: '{result.get('title')}' by '{result.get('primary_artist', {}).get('name')}' - "
                                f"String: {string_score:.3f}, Semantic: {semantic_score:.3f}, Combined: {combined_score:.3f}"
                            )
                            
                            if combined_score > best_combined_score:
                                best_combined_score = combined_score
                                best_match = result
                        
                        if best_match:
                            logger.info(
                                f"Best semantic match: '{best_match.get('title')}' by '{best_match.get('primary_artist', {}).get('name')}' "
                                f"(combined score: {best_combined_score:.3f})"
                            )
                            song_id = best_match.get('id')
                            return await self.get_song_by_id(song_id)
                        
                        return None
                    else:
                        logger.error(f"Genius API search error: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
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
