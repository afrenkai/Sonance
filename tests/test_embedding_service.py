"""Tests for EmbeddingService."""
import pytest
import numpy as np
from backend.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService class."""
    
    def test_initialization(self, embedding_service):
        """Test that EmbeddingService initializes correctly."""
        assert embedding_service.model is not None
        assert embedding_service.embedding_dim > 0
        assert embedding_service.embedding_dim == 384  # all-MiniLM-L6-v2 dimension
    
    def test_encode_text_single_string(self, embedding_service):
        """Test encoding a single text string."""
        text = "This is a test sentence"
        embedding = embedding_service.encode_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.all(embedding == 0)  # Should not be all zeros
    
    def test_encode_text_list(self, embedding_service):
        """Test encoding multiple text strings."""
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = embedding_service.encode_text(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        # Verify each embedding is different
        assert not np.array_equal(embeddings[0], embeddings[1])
    
    def test_encode_song(self, embedding_service):
        """Test encoding a song with name and artist."""
        song_name = "Bohemian Rhapsody"
        artist = "Queen"
        
        embedding = embedding_service.encode_song(song_name, artist)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.all(embedding == 0)
    
    def test_encode_song_with_lyrics(self, embedding_service):
        """Test encoding a song with lyrics included."""
        song_name = "Imagine"
        artist = "John Lennon"
        lyrics = "Imagine there's no heaven, it's easy if you try"
        
        embedding_with_lyrics = embedding_service.encode_song(song_name, artist, lyrics)
        embedding_without_lyrics = embedding_service.encode_song(song_name, artist)
        
        assert isinstance(embedding_with_lyrics, np.ndarray)
        assert embedding_with_lyrics.shape == (384,)
        # Embeddings should be different when lyrics are included
        assert not np.allclose(embedding_with_lyrics, embedding_without_lyrics)
    
    def test_encode_emotion(self, embedding_service):
        """Test encoding an emotion string."""
        emotion = "happy"
        
        embedding = embedding_service.encode_emotion(emotion)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.all(embedding == 0)
    
    def test_encode_different_emotions(self, embedding_service):
        """Test that different emotions produce different embeddings."""
        happy_emb = embedding_service.encode_emotion("happy")
        sad_emb = embedding_service.encode_emotion("sad")
        
        assert not np.allclose(happy_emb, sad_emb)
    
    def test_combine_embeddings_equal_weights(self, embedding_service):
        """Test combining embeddings with equal weights."""
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        emb3 = np.random.randn(384).astype(np.float32)
        
        embeddings = [emb1, emb2, emb3]
        combined = embedding_service.combine_embeddings(embeddings)
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (384,)
        # Should be normalized
        assert np.isclose(np.linalg.norm(combined), 1.0, atol=1e-6)
    
    def test_combine_embeddings_with_weights(self, embedding_service):
        """Test combining embeddings with custom weights."""
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        
        embeddings = [emb1, emb2]
        weights = [0.7, 0.3]
        
        combined = embedding_service.combine_embeddings(embeddings, weights)
        
        assert isinstance(combined, np.ndarray)
        assert combined.shape == (384,)
        assert np.isclose(np.linalg.norm(combined), 1.0, atol=1e-6)
    
    def test_combine_embeddings_invalid_weights(self, embedding_service):
        """Test that invalid weights raise an error."""
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        
        embeddings = [emb1, emb2]
        invalid_weights = [0.5, 0.3]  # Don't sum to 1.0
        
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            embedding_service.combine_embeddings(embeddings, invalid_weights)
    
    def test_combine_embeddings_mismatched_lengths(self, embedding_service):
        """Test that mismatched lengths raise an error."""
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        
        embeddings = [emb1, emb2]
        weights = [0.5, 0.3, 0.2]  # Too many weights
        
        with pytest.raises(ValueError, match="Number of weights must match"):
            embedding_service.combine_embeddings(embeddings, weights)
    
    def test_combine_embeddings_empty_list(self, embedding_service):
        """Test that empty embeddings list raises an error."""
        with pytest.raises(ValueError, match="No embeddings provided"):
            embedding_service.combine_embeddings([])
    
    def test_compute_similarity_identical(self, embedding_service):
        """Test similarity between identical embeddings."""
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        similarity = embedding_service.compute_similarity(emb, emb)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert np.isclose(similarity, 1.0, atol=1e-5)
    
    def test_compute_similarity_different(self, embedding_service):
        """Test similarity between different embeddings."""
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = np.random.randn(384).astype(np.float32)
        
        similarity = embedding_service.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0
    
    def test_compute_similarity_normalized_range(self, embedding_service):
        """Test that similarity is normalized to [0, 1] range."""
        # Create embeddings that would have negative cosine similarity
        emb1 = np.ones(384)
        emb2 = -np.ones(384)
        
        similarity = embedding_service.compute_similarity(emb1, emb2)
        
        # Use tolerance for floating point precision
        assert -1e-10 <= similarity <= 1.0
        assert similarity < 0.2  # Should be close to 0
    
    def test_batch_similarity(self, embedding_service):
        """Test computing similarity for multiple embeddings."""
        query_emb = np.random.randn(384).astype(np.float32)
        batch_embs = np.random.randn(5, 384).astype(np.float32)
        
        similarities = embedding_service.batch_similarity(query_emb, batch_embs)
        
        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (5,)
        assert np.all((similarities >= 0) & (similarities <= 1))
    
    def test_batch_similarity_single_embedding(self, embedding_service):
        """Test batch similarity with a single embedding."""
        query_emb = np.random.randn(384).astype(np.float32)
        batch_embs = np.random.randn(1, 384).astype(np.float32)
        
        similarities = embedding_service.batch_similarity(query_emb, batch_embs)
        
        assert similarities.shape == (1,)
        assert 0.0 <= similarities[0] <= 1.0
    
    def test_encode_song_consistency(self, embedding_service):
        """Test that encoding the same song twice gives identical results."""
        song_name = "Test Song"
        artist = "Test Artist"
        
        emb1 = embedding_service.encode_song(song_name, artist)
        emb2 = embedding_service.encode_song(song_name, artist)
        
        assert np.allclose(emb1, emb2)
    
    def test_encode_emotion_consistency(self, embedding_service):
        """Test that encoding the same emotion twice gives identical results."""
        emotion = "joyful"
        
        emb1 = embedding_service.encode_emotion(emotion)
        emb2 = embedding_service.encode_emotion(emotion)
        
        assert np.allclose(emb1, emb2)
