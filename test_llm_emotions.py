"""
Test script demonstrating LLM-based contextual emotion understanding
vs. static keyword-based matching.
"""

import logging
from backend.services.llm_emotion_service import LLMEmotionService
from backend.services.emotion_mapper import EmotionMapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_emotion_similarity():
    """Test contextual emotion understanding."""
    print("\n" + "="*70)
    print("CONTEXTUAL EMOTION UNDERSTANDING TEST")
    print("="*70)
    
    llm_service = LLMEmotionService()
    
    # Test cases: songs with their actual vibe
    test_songs = [
        ("Happy", "Pharrell Williams"),
        ("Sad", "XXXTentacion"),
        ("Lose Yourself", "Eminem"),
        ("Clair de Lune", "Claude Debussy"),
        ("Bodies", "Drowning Pool"),
        ("Somewhere Over the Rainbow", "Israel Kamakawiwoʻole"),
        ("Stairway to Heaven", "Led Zeppelin"),
        ("Toxic", "Britney Spears"),
    ]
    
    emotions_to_test = ["happy", "sad", "energetic", "calm", "angry", "hopeful", "romantic"]
    
    print("\nTesting song-emotion matches (higher = better match):\n")
    
    for song, artist in test_songs:
        print(f"\n🎵 {song} by {artist}")
        song_text = f"{song} by {artist}"
        
        scores = []
        for emotion in emotions_to_test:
            score = llm_service.compute_emotion_similarity(song_text, emotion, context="song")
            scores.append((emotion, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 3
        for i, (emotion, score) in enumerate(scores[:3]):
            bar = "█" * int(score * 30)
            print(f"  {i+1}. {emotion:12s} {score:.3f} {bar}")


def test_audio_feature_inference():
    """Test inferring emotions from audio features."""
    print("\n" + "="*70)
    print("AUDIO FEATURE EMOTION INFERENCE TEST")
    print("="*70)
    
    llm_service = LLMEmotionService()
    
    # Test different audio feature profiles
    test_profiles = [
        {
            "name": "High Energy Happy Dance Track",
            "features": {
                "valence": 0.9,
                "energy": 0.95,
                "danceability": 0.88,
                "tempo": 128
            }
        },
        {
            "name": "Melancholic Acoustic Ballad",
            "features": {
                "valence": 0.2,
                "energy": 0.3,
                "acousticness": 0.9,
                "tempo": 70
            }
        },
        {
            "name": "Aggressive High-Energy Rock",
            "features": {
                "valence": 0.3,
                "energy": 0.95,
                "loudness": -5,
                "tempo": 150
            }
        },
        {
            "name": "Peaceful Ambient Soundscape",
            "features": {
                "valence": 0.6,
                "energy": 0.15,
                "acousticness": 0.8,
                "instrumentalness": 0.9,
                "tempo": 60
            }
        }
    ]
    
    print("\nInferring emotions from audio features:\n")
    
    for profile in test_profiles:
        print(f"\n🎼 {profile['name']}")
        print(f"   Features: {profile['features']}")
        
        emotion_scores = llm_service.infer_emotion_from_audio_features(
            profile['features']
        )
        
        # Show top 3
        print("   Top emotions:")
        for i, (emotion, score) in enumerate(emotion_scores[:3]):
            bar = "█" * int(score * 30)
            print(f"     {i+1}. {emotion:12s} {score:.3f} {bar}")


def test_multi_emotion_analysis():
    """Test analyzing multiple emotions together."""
    print("\n" + "="*70)
    print("MULTI-EMOTION ANALYSIS TEST")
    print("="*70)
    
    llm_service = LLMEmotionService()
    
    test_cases = [
        ["happy", "energetic"],  # Harmonious
        ["sad", "hopeful"],      # Bittersweet
        ["happy", "angry"],      # Conflicting
        ["calm", "peaceful"],    # Very similar
        ["romantic", "hopeful", "peaceful"]  # Triple blend
    ]
    
    print("\nAnalyzing emotion combinations:\n")
    
    for emotions in test_cases:
        print(f"\n🎭 {' + '.join(emotions)}")
        
        analysis = llm_service.analyze_multi_emotion_query(emotions)
        
        print(f"   Blended emotion: {analysis['blended_emotion']} "
              f"(confidence: {analysis['blend_confidence']:.3f})")
        print(f"   Coherent: {'✓ Yes' if analysis['is_coherent'] else '✗ No'}")
        
        if analysis['harmonies']:
            print(f"   Harmonies: {len(analysis['harmonies'])} pairs work well together")
        
        if analysis['conflicts']:
            print(f"   ⚠️  Conflicts detected:")
            for e1, e2, sim in analysis['conflicts']:
                print(f"      - {e1} vs {e2} (similarity: {sim:.2f})")


def test_related_emotions():
    """Test finding related emotions."""
    print("\n" + "="*70)
    print("RELATED EMOTIONS TEST")
    print("="*70)
    
    llm_service = LLMEmotionService()
    
    emotions_to_explore = ["happy", "sad", "energetic", "melancholic"]
    
    print("\nFinding similar emotions:\n")
    
    for emotion in emotions_to_explore:
        print(f"\n💭 Emotions similar to '{emotion}':")
        
        related = llm_service.find_related_emotions(emotion, top_k=3)
        
        for i, (rel_emotion, similarity) in enumerate(related):
            bar = "█" * int((similarity + 1) * 15)  # Normalize from -1,1 to 0,30
            print(f"   {i+1}. {rel_emotion:12s} (sim: {similarity:.3f}) {bar}")


def compare_static_vs_llm():
    """Compare static rule-based vs LLM-based emotion mapping."""
    print("\n" + "="*70)
    print("STATIC vs LLM COMPARISON")
    print("="*70)
    
    # Initialize both mappers
    static_mapper = EmotionMapper(use_llm=False)
    llm_mapper = EmotionMapper(use_llm=True)
    
    # Test audio features for an upbeat pop song
    test_features = {
        "valence": 0.85,
        "energy": 0.75,
        "danceability": 0.80,
        "tempo": 120,
        "acousticness": 0.1
    }
    
    emotions = ["happy", "sad", "energetic", "calm", "hopeful"]
    
    print("\nScoring test track against emotions:")
    print(f"Track features: {test_features}\n")
    
    print(f"{'Emotion':<15} {'Static Score':<15} {'LLM Score':<15} {'Difference'}")
    print("-" * 60)
    
    for emotion in emotions:
        static_score = static_mapper.compute_emotion_score(test_features, emotion)
        llm_score = llm_mapper.compute_emotion_score(test_features, emotion)
        diff = llm_score - static_score
        
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{emotion:<15} {static_score:.3f}           {llm_score:.3f}           "
              f"{arrow} {abs(diff):.3f}")


def main():
    """Run all tests."""
    print("\n" + "🎵" * 35)
    print("LLM-BASED EMOTION UNDERSTANDING DEMO")
    print("🎵" * 35)
    
    try:
        test_emotion_similarity()
        test_audio_feature_inference()
        test_multi_emotion_analysis()
        test_related_emotions()
        compare_static_vs_llm()
        
        print("\n" + "="*70)
        print("✨ All tests completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
