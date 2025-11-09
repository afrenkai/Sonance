import asyncio
import os
import logging
import re
import math
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
    
    @staticmethod
    def normalize_query(text: str) -> str:
        """
        Normalize a search query by removing special characters and extra whitespace.
        
        Theory: Text normalization reduces surface form variations while preserving
        semantic content, improving recall in information retrieval systems.
        """
        if not text:
            return ""
        
        # Remove content in parentheses (often features, versions, etc.)
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Remove content in brackets
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute Levenshtein (edit) distance between two strings.
        
        Theory: Edit distance measures minimum number of single-character edits
        (insertions, deletions, substitutions) needed to transform s1 into s2.
        Uses dynamic programming with O(m*n) time and space complexity.
        
        Algorithm:
            Let dp[i][j] = edit distance between s1[0:i] and s2[0:j]
            Base: dp[0][j] = j, dp[i][0] = i
            Recurrence: dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution (cost=0 if s1[i]==s2[j], else 1)
            )
        """
        if len(s1) < len(s2):
            return AsyncGeniusService.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # Previous row of distances
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            # Current row of distances
            current_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                
                current_row.append(min(insertions, deletions, substitutions))
            
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
        """
        Compute normalized Levenshtein similarity (0 to 1).
        
        Theory: Normalization by maximum possible distance allows comparison
        across string pairs of different lengths. Returns 1 - (distance / max_length).
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        distance = AsyncGeniusService.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    @staticmethod
    def compute_bm25_score(query_terms: List[str], document: str, 
                          k1: float = 1.5, b: float = 0.75, 
                          avgdl: float = 20.0) -> float:
        """
        Compute BM25 (Best Match 25) ranking score.
        
        Theory: BM25 is a probabilistic ranking function that estimates
        relevance of documents to a query. It's the state-of-the-art
        for lexical matching in information retrieval.
        
        Parameters:
            k1: Controls term frequency saturation (typical: 1.2-2.0)
            b: Controls length normalization (0=no norm, 1=full norm)
            avgdl: Average document length in corpus
        
        Algorithm:
            BM25(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
            
            where:
            - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
            - f(qi,D) = frequency of term qi in document D
            - |D| = document length
            - N = total documents (simplified to 10 for single-doc scoring)
        """
        if not query_terms or not document:
            return 0.0
        
        document_lower = document.lower()
        doc_length = len(document_lower.split())
        
        score = 0.0
        N = 10  # Assumed corpus size (simplified)
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Term frequency in document
            tf = document_lower.count(term_lower)
            
            if tf == 0:
                continue
            
            # Document frequency (simplified: assume term appears in 1 doc)
            df = 1
            
            # IDF component
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            # Length normalization
            length_norm = 1 - b + b * (doc_length / avgdl)
            
            # BM25 score component for this term
            term_score = idf * (tf * (k1 + 1)) / (tf + k1 * length_norm)
            score += term_score
        
        return score
    
    def generate_query_variants(self, song_name: str, artist: Optional[str] = None) -> List[str]:
        """
        Generate multiple query variants for improved recall.
        
        Theory: Query expansion increases recall by generating semantically
        equivalent or related queries, compensating for vocabulary mismatch
        between user queries and indexed documents.
        """
        variants = []
        
        # Original query
        if artist:
            variants.append(f"{song_name} {artist}")
        else:
            variants.append(song_name)
        
        # Normalized query (without parentheses, brackets, etc.)
        normalized_song = self.normalize_query(song_name)
        if normalized_song != song_name:
            if artist:
                variants.append(f"{normalized_song} {artist}")
            else:
                variants.append(normalized_song)
        
        # With "by" separator
        if artist:
            variants.append(f"{song_name} by {artist}")
            variants.append(f"{normalized_song} by {artist}")
        
        # Artist - Song format
        if artist:
            variants.append(f"{artist} - {song_name}")
            variants.append(f"{artist} - {normalized_song}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)
        
        return unique_variants
    
    def compute_match_score(
        self,
        result: Dict[str, Any],
        query_song: str,
        query_artist: Optional[str] = None,
        use_fuzzy: bool = True,
        use_bm25: bool = True
    ) -> float:
        """
        Compute match score between search result and query using ensemble of methods.
        Returns a score between 0 and 1, where 1 is a perfect match.
        
        Theory: Ensemble methods combine multiple weak learners to create a stronger
        predictor. We combine:
        1. SequenceMatcher (gestalt pattern matching)
        2. Levenshtein distance (edit distance)
        3. BM25 (probabilistic ranking)
        
        Each contributes to the final score with learned/tuned weights.
        """
        result_title = result.get('title', '').lower()
        result_artist = result.get('primary_artist', {}).get('name', '').lower()
        
        query_song_lower = query_song.lower()
        query_artist_lower = query_artist.lower() if query_artist else ''
        
        # 1. SequenceMatcher similarity (gestalt pattern matching)
        title_sim_seq = SequenceMatcher(None, result_title, query_song_lower).ratio()
        
        artist_sim_seq = 1.0
        if query_artist:
            artist_sim_seq = SequenceMatcher(None, result_artist, query_artist_lower).ratio()
        
        # Base score from SequenceMatcher
        base_score = 0.6 * title_sim_seq + 0.4 * artist_sim_seq
        
        # 2. Levenshtein similarity (fuzzy matching)
        fuzzy_component = 0.0
        if use_fuzzy:
            title_sim_lev = self.normalized_levenshtein_similarity(result_title, query_song_lower)
            artist_sim_lev = 1.0
            if query_artist:
                artist_sim_lev = self.normalized_levenshtein_similarity(result_artist, query_artist_lower)
            
            fuzzy_component = 0.6 * title_sim_lev + 0.4 * artist_sim_lev
        
        # 3. BM25 scoring
        bm25_component = 0.0
        if use_bm25:
            query_terms = query_song_lower.split()
            if query_artist:
                query_terms.extend(query_artist_lower.split())
            
            document = f"{result_title} {result_artist}"
            bm25_raw = self.compute_bm25_score(query_terms, document)
            
            # Normalize BM25 score to [0, 1] range (empirical max ~10)
            bm25_component = min(bm25_raw / 10.0, 1.0)
        
        # Ensemble combination with weights
        # Base gets 50%, fuzzy 25%, BM25 25%
        weights = [0.50, 0.25, 0.25]
        components = [base_score, fuzzy_component, bm25_component]
        
        score = sum(w * c for w, c in zip(weights, components))
        
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
    
    async def advanced_search_song(
        self,
        song_name: str,
        artist: str = None,
        min_confidence: float = 0.6,
        use_query_expansion: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Advanced search with query expansion and confidence filtering.
        
        Theory: Multi-stage retrieval pipeline:
        1. Query expansion generates alternative formulations
        2. Parallel search across variants increases recall
        3. Confidence threshold filtering reduces false positives
        4. Best result aggregation improves precision
        
        Parameters:
            min_confidence: Minimum score threshold (0-1) to accept results
            use_query_expansion: Whether to try multiple query formulations
        
        Returns best match above confidence threshold, or None
        """
        if not self.is_available():
            logger.warning("Genius service not available - missing API token")
            return None
        
        queries = [f"{song_name} {artist}" if artist else song_name]
        
        if use_query_expansion:
            queries = self.generate_query_variants(song_name, artist)
            logger.debug(f"Generated {len(queries)} query variants: {queries}")
        
        best_match = None
        best_score = 0.0
        best_query = None
        
        url = f"{self.BASE_URL}/search"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Try each query variant
        for query in queries:
            try:
                params = {"q": query}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            hits = data.get('response', {}).get('hits', [])
                            
                            if not hits:
                                continue
                            
                            # Score all candidates from this query
                            for hit in hits:
                                result = hit.get('result', {})
                                score = self.compute_match_score(result, song_name, artist)
                                
                                if score > best_score:
                                    best_score = score
                                    best_match = result
                                    best_query = query
                        
            except Exception as e:
                logger.warning(f"Error with query '{query}': {e}")
                continue
        
        # Apply confidence threshold
        if best_match and best_score >= min_confidence:
            logger.info(
                f"Advanced search found: '{best_match.get('title')}' by "
                f"'{best_match.get('primary_artist', {}).get('name')}' "
                f"(score: {best_score:.3f}, query: '{best_query}')"
            )
            song_id = best_match.get('id')
            return await self.get_song_by_id(song_id)
        elif best_match:
            logger.warning(
                f"Best match below confidence threshold: {best_score:.3f} < {min_confidence}"
            )
        
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
