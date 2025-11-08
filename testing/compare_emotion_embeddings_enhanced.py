from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"  # Better semantic model
RANDOM_SEED: int = 42
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_WORDS: List[str] = [
    "happy",
    "sad",
    "angry",
    "calm",
    "energetic",
    "melancholic",
    "joyful",
    "peaceful",
    "excited",
    "relaxed",
    "nostalgic",
    "uplifting",
    "moody",
    "chill",
]

EMOTION_ANTONYMS: List[Tuple[str, str]] = [
    ("happy", "sad"),
    ("energetic", "calm"),
    ("joyful", "melancholic"),
    ("excited", "relaxed"),
    ("angry", "peaceful"),
    ("uplifting", "moody"),
]

CONTEXT_TYPES: Dict[str, str] = {
    "isolated": "{word}",
    "feeling_phrase": "feeling {word} and uplifted",
    "music_description": "This music sounds {word} and makes me feel {word}.",
    "spotify_search": "I want to find {word} songs on Spotify to match my {word} mood.",
    "emotion_statement": "I'm feeling {word} today. I need {word} music.",
    "playlist_context": "Create a playlist with {word} vibes. Play some {word} tracks.",
    "song_request": "Songs that make you feel {word}",
    "playlist_name": "The {word} playlist",
    "music_recommendation": "{word} music recommendations",
    "artist_style": "Artists known for {word} songs",
    "mood_match": "Music to match my {word} feelings",
}


def _ensure_output_dirs(base_dir: Path) -> Tuple[Path, Path]:
    output_dir = base_dir / "emotion_output_enhanced"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, plots_dir


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sentence_embedding(model: SentenceTransformer, text: str) -> np.ndarray:
    with torch.no_grad():
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))


def plot_pca_3d(points: Dict[str, np.ndarray], title: str, save_path: Path) -> None:
    labels = list(points.keys())
    X = np.stack([points[k] for k in labels], axis=0)
    n_components = min(3, X.shape[0], X.shape[1])
    if n_components < 3:
        return
    
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    X_transformed = pca.fit_transform(X)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
    
    for i, label in enumerate(labels):
        x, y, z = X_transformed[i, 0], X_transformed[i, 1], X_transformed[i, 2]
        ax.scatter(x, y, z, s=200, c=[colors[i]], alpha=0.8, edgecolors='black', linewidth=2)
        ax.text(x, y, z, label, fontsize=9, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    total_var = sum(pca.explained_variance_ratio_[:3])
    ax.text2D(0.05, 0.95, f'Total variance: {total_var:.1%}', 
              transform=ax.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Path
) -> None:
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Context Type', fontsize=12)
    plt.ylabel('Context Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_cross_emotion_similarities(
    emotion_words: List[str],
    context_type: str,
    similarities: Dict[Tuple[str, str], float],
    save_path: Path
) -> None:
    n = len(emotion_words)
    matrix = np.zeros((n, n))
    
    for i, word1 in enumerate(emotion_words):
        for j, word2 in enumerate(emotion_words):
            if i == j:
                matrix[i, j] = 1.0
            else:
                key = (word1, word2) if (word1, word2) in similarities else (word2, word1)
                matrix[i, j] = similarities.get(key, 0.0)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        vmin=0.0,
        vmax=1.0,
        xticklabels=emotion_words,
        yticklabels=emotion_words,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f'Emotion Word Similarities - {context_type}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Emotion Word', fontsize=12)
    plt.ylabel('Emotion Word', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_antonym_analysis(
    antonym_pairs: List[Tuple[str, str]],
    context_types: List[str],
    similarities: Dict[str, Dict[Tuple[str, str], float]],
    save_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(antonym_pairs))
    width = 0.8 / len(context_types)
    
    for i, context in enumerate(context_types):
        sims = [similarities[context].get(pair, 0.0) for pair in antonym_pairs]
        offset = (i - len(context_types) / 2) * width + width / 2
        ax.bar(x + offset, sims, width, label=context, alpha=0.8)
    
    ax.set_xlabel('Antonym Pairs', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Antonym Distinctiveness Across Contexts\n(Lower = Better Separation)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a} vs {b}' for a, b in antonym_pairs], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main() -> None:
    set_seed(RANDOM_SEED)
    base_dir = Path(__file__).resolve().parent
    output_dir, plots_dir = _ensure_output_dirs(base_dir)

    print(f"Using device: {DEVICE}")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("Model loaded successfully!\n")

    # Storage structures
    embeddings_out: Dict[str, Dict[str, List[float]]] = {}
    similarity_analysis: Dict[str, Dict] = {}
    
    print("=" * 80)
    print("ANALYZING EMOTION WORDS IN DIFFERENT CONTEXTS (Sentence Transformers)")
    print("=" * 80)
    
    for emotion in EMOTION_WORDS:
        print(f"\n{'='*60}")
        print(f"Processing emotion: '{emotion}'")
        print(f"{'='*60}")
        
        context_embeddings: Dict[str, np.ndarray] = {}
        
        # Get embeddings for each context type
        for context_name, template in CONTEXT_TYPES.items():
            text = template.format(word=emotion)
            vec = get_sentence_embedding(model, text)
            context_embeddings[context_name] = vec
            print(f"  ✓ Generated embedding for '{context_name}' context")
        
        # Store embeddings
        embeddings_out[emotion] = {
            k: v.tolist() for k, v in context_embeddings.items()
        }
        
        # Compute pairwise similarities between contexts
        context_names = list(CONTEXT_TYPES.keys())
        n_contexts = len(context_names)
        sim_matrix = np.zeros((n_contexts, n_contexts))
        
        similarities_detail = {}
        for i, ctx1 in enumerate(context_names):
            for j, ctx2 in enumerate(context_names):
                sim = cosine_similarity(context_embeddings[ctx1], context_embeddings[ctx2])
                sim_matrix[i, j] = sim
                if i < j:  # Only store upper triangle
                    similarities_detail[f"{ctx1}_vs_{ctx2}"] = sim
        
        # Analyze stability: how similar is each context to isolated?
        isolated_vec = context_embeddings["isolated"]
        context_stability = {}
        for ctx_name, ctx_vec in context_embeddings.items():
            if ctx_name != "isolated":
                sim = cosine_similarity(isolated_vec, ctx_vec)
                context_stability[ctx_name] = sim
        
        similarity_analysis[emotion] = {
            "pairwise_similarities": similarities_detail,
            "stability_vs_isolated": context_stability,
            "avg_similarity_to_isolated": np.mean(list(context_stability.values())),
            "min_similarity_to_isolated": min(context_stability.values()),
            "max_similarity_to_isolated": max(context_stability.values()),
        }
        
        # Print summary
        print(f"\n  Similarity to isolated word:")
        for ctx, sim in sorted(context_stability.items(), key=lambda x: x[1], reverse=True):
            print(f"    {ctx:25s}: {sim:.4f}")
        
        # Plot 3D PCA for this emotion
        plot_pca_3d(
            context_embeddings,
            title=f"Context Comparison: '{emotion}'",
            save_path=plots_dir / f"pca_{emotion}.png"
        )
        
        # Plot similarity heatmap
        plot_similarity_heatmap(
            sim_matrix,
            context_names,
            title=f"Context Similarities: '{emotion}'",
            save_path=plots_dir / f"heatmap_{emotion}.png"
        )
    
    # Cross-emotion analysis: compare different emotion words in the same context
    print(f"\n{'='*80}")
    print("CROSS-EMOTION ANALYSIS")
    print(f"{'='*80}")
    
    cross_emotion_sims: Dict[str, Dict[Tuple[str, str], float]] = {}
    
    for context_name, template in CONTEXT_TYPES.items():
        print(f"\nAnalyzing '{context_name}' context:")
        context_sims = {}
        
        for i, emotion1 in enumerate(EMOTION_WORDS):
            text1 = template.format(word=emotion1)
            vec1 = get_sentence_embedding(model, text1)
            
            for j, emotion2 in enumerate(EMOTION_WORDS):
                if i >= j:
                    continue
                    
                text2 = template.format(word=emotion2)
                vec2 = get_sentence_embedding(model, text2)
                
                sim = cosine_similarity(vec1, vec2)
                context_sims[(emotion1, emotion2)] = sim
        
        cross_emotion_sims[context_name] = context_sims
        
        # Print some examples
        sorted_pairs = sorted(context_sims.items(), key=lambda x: x[1], reverse=True)
        print(f"  Most similar pair: {sorted_pairs[0][0]} = {sorted_pairs[0][1]:.4f}")
        print(f"  Least similar pair: {sorted_pairs[-1][0]} = {sorted_pairs[-1][1]:.4f}")
        
        # Plot cross-emotion heatmap
        plot_cross_emotion_similarities(
            EMOTION_WORDS,
            context_name,
            context_sims,
            plots_dir / f"cross_emotion_{context_name}.png"
        )
    
    # Antonym analysis: validate that opposite emotions are well-separated
    print(f"\n{'='*80}")
    print("ANTONYM DISTINCTIVENESS ANALYSIS")
    print(f"{'='*80}\n")
    
    antonym_analysis: Dict[str, Dict[Tuple[str, str], float]] = {}
    
    for context_name, template in CONTEXT_TYPES.items():
        context_antonym_sims = {}
        
        for word1, word2 in EMOTION_ANTONYMS:
            text1 = template.format(word=word1)
            text2 = template.format(word=word2)
            
            vec1 = get_sentence_embedding(model, text1)
            vec2 = get_sentence_embedding(model, text2)
            
            sim = cosine_similarity(vec1, vec2)
            context_antonym_sims[(word1, word2)] = sim
        
        antonym_analysis[context_name] = context_antonym_sims
        
        avg_antonym_sim = np.mean(list(context_antonym_sims.values()))
        print(f"{context_name:25s}: avg antonym similarity = {avg_antonym_sim:.4f}")
    
    # Plot antonym analysis
    plot_antonym_analysis(
        EMOTION_ANTONYMS,
        list(CONTEXT_TYPES.keys()),
        antonym_analysis,
        plots_dir / "antonym_distinctiveness.png"
    )
    
    # Overall analysis: which context is most representative?
    print(f"\n{'='*80}")
    print("OVERALL CONTEXT REPRESENTATIVENESS ANALYSIS")
    print(f"{'='*80}\n")
    
    avg_stability_by_context = {}
    for context_name in CONTEXT_TYPES.keys():
        if context_name == "isolated":
            continue
        stabilities = [
            similarity_analysis[emotion]["stability_vs_isolated"][context_name]
            for emotion in EMOTION_WORDS
        ]
        avg_stability_by_context[context_name] = {
            "mean": np.mean(stabilities),
            "std": np.std(stabilities),
            "min": min(stabilities),
            "max": max(stabilities),
        }
    
    print("Average stability across all emotions (higher = more consistent with isolated word):")
    for context_name, stats in sorted(avg_stability_by_context.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {context_name:25s}: {stats['mean']:.4f} ± {stats['std']:.4f} (range: {stats['min']:.4f} - {stats['max']:.4f})")
    
    # Antonym separation analysis
    print(f"\n{'='*80}")
    print("ANTONYM SEPARATION BY CONTEXT")
    print(f"{'='*80}\n")
    
    avg_antonym_sep_by_context = {}
    for context_name, sims in antonym_analysis.items():
        avg_sim = np.mean(list(sims.values()))
        # Lower similarity = better separation
        separation_score = 1.0 - avg_sim
        avg_antonym_sep_by_context[context_name] = {
            "avg_similarity": avg_sim,
            "separation_score": separation_score,
        }
    
    print("Antonym separation by context (higher separation = better):")
    for context_name, stats in sorted(avg_antonym_sep_by_context.items(), 
                                     key=lambda x: x[1]["separation_score"], reverse=True):
        print(f"  {context_name:25s}: separation={stats['separation_score']:.4f} (sim={stats['avg_similarity']:.4f})")
    
    # Save all results
    with (output_dir / "embeddings.json").open("w", encoding="utf-8") as f:
        json.dump(embeddings_out, f, ensure_ascii=False, indent=2)
    
    with (output_dir / "similarity_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(similarity_analysis, f, ensure_ascii=False, indent=2)
    
    with (output_dir / "context_representativeness.json").open("w", encoding="utf-8") as f:
        json.dump(avg_stability_by_context, f, ensure_ascii=False, indent=2)
    
    with (output_dir / "cross_emotion_similarities.json").open("w", encoding="utf-8") as f:
        serializable = {
            ctx: {f"{k[0]}_vs_{k[1]}": v for k, v in sims.items()}
            for ctx, sims in cross_emotion_sims.items()
        }
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    with (output_dir / "antonym_analysis.json").open("w", encoding="utf-8") as f:
        serializable = {
            ctx: {f"{k[0]}_vs_{k[1]}": v for k, v in sims.items()}
            for ctx, sims in antonym_analysis.items()
        }
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    # Generate comprehensive summary report
    with (output_dir / "ANALYSIS_SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ENHANCED EMOTION WORD EMBEDDING ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Emotions analyzed: {len(EMOTION_WORDS)}\n")
        f.write(f"Context types: {len(CONTEXT_TYPES)}\n")
        f.write(f"Antonym pairs validated: {len(EMOTION_ANTONYMS)}\n\n")
        
        f.write("OBJECTIVE:\n")
        f.write("Validate that emotion words maintain consistent semantic meaning across\n")
        f.write("different contexts using Sentence Transformers (optimized for semantic similarity).\n")
        f.write("Also validate that antonym emotions are well-separated in embedding space.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. CONTEXT STABILITY (similarity to isolated word usage):\n\n")
        for context_name, stats in sorted(avg_stability_by_context.items(), key=lambda x: x[1]["mean"], reverse=True):
            f.write(f"   {context_name:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"   {'':27s}  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
        
        f.write("\n2. ANTONYM SEPARATION (lower similarity = better):\n\n")
        for context_name, stats in sorted(avg_antonym_sep_by_context.items(), 
                                         key=lambda x: x[1]["separation_score"], reverse=True):
            f.write(f"   {context_name:25s}: separation={stats['separation_score']:.4f} ")
            f.write(f"(similarity={stats['avg_similarity']:.4f})\n")
        
        f.write("\n3. PER-EMOTION STABILITY:\n\n")
        for emotion in EMOTION_WORDS:
            avg_sim = similarity_analysis[emotion]["avg_similarity_to_isolated"]
            min_sim = similarity_analysis[emotion]["min_similarity_to_isolated"]
            max_sim = similarity_analysis[emotion]["max_similarity_to_isolated"]
            f.write(f"   {emotion:15s}: avg={avg_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}\n")
        
        f.write("\n4. INTERPRETATION:\n\n")
        f.write("   Context Stability:\n")
        f.write("   - Values closer to 1.0 indicate high semantic consistency\n")
        f.write("   - Values above 0.85 suggest the context preserves word meaning well\n")
        f.write("   - Lower variance indicates stable performance across emotions\n\n")
        
        f.write("   Antonym Separation:\n")
        f.write("   - Similarity below 0.5 indicates good distinctiveness\n")
        f.write("   - Higher separation scores mean better emotion discrimination\n")
        f.write("   - Important for ensuring the model can distinguish opposite emotions\n\n")
        
        best_context = max(avg_stability_by_context.items(), key=lambda x: x[1]["mean"])
        best_separation = max(avg_antonym_sep_by_context.items(), key=lambda x: x[1]["separation_score"])
        
        f.write(f"   BEST CONTEXT FOR STABILITY: {best_context[0]}\n")
        f.write(f"   (Average stability: {best_context[1]['mean']:.4f})\n\n")
        
        f.write(f"   BEST CONTEXT FOR ANTONYM SEPARATION: {best_separation[0]}\n")
        f.write(f"   (Separation score: {best_separation[1]['separation_score']:.4f})\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 80 + "\n\n")
        
        stability_score = best_context[1]["mean"]
        separation_score = best_separation[1]["separation_score"]
        
        if stability_score >= 0.85 and separation_score >= 0.3:
            f.write("✓ EXCELLENT: Emotion words show high consistency AND good separation.\n")
            f.write("  Sentence Transformers provide reliable emotion embeddings for music search.\n")
        elif stability_score >= 0.75 and separation_score >= 0.2:
            f.write("✓ GOOD: Emotion words show reasonable consistency and separation.\n")
            f.write("  Suitable for use in Spotify search with monitoring.\n")
        elif stability_score >= 0.85 or separation_score >= 0.3:
            f.write("⚠ MIXED: Strong in one metric but weaker in the other.\n")
            f.write("  Consider the trade-off based on your use case.\n")
        else:
            f.write("⚠ CAUTION: Consider alternative approaches or more context.\n")
        
        f.write(f"\nModel Comparison:\n")
        f.write(f"- This analysis uses {MODEL_NAME}\n")
        f.write("- Sentence Transformers are optimized for semantic similarity\n")
        f.write("- Compare with BERT results to validate improvements\n")
    
    print(f"\n{'='*80}")
    print("ENHANCED ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Embeddings: embeddings.json")
    print(f"  - Detailed analysis: similarity_analysis.json")
    print(f"  - Context comparison: context_representativeness.json")
    print(f"  - Cross-emotion analysis: cross_emotion_similarities.json")
    print(f"  - Antonym analysis: antonym_analysis.json")
    print(f"  - Summary report: ANALYSIS_SUMMARY.txt")
    print(f"\nVisualizations saved to: {plots_dir}")
    print(f"  - 3D PCA plots per emotion: pca_<emotion>.png")
    print(f"  - Similarity heatmaps: heatmap_<emotion>.png")
    print(f"  - Cross-emotion heatmaps: cross_emotion_<context>.png")
    print(f"  - Antonym distinctiveness: antonym_distinctiveness.png")
    print("\n✨ Check ANALYSIS_SUMMARY.txt for key findings and recommendations!\n")


if __name__ == "__main__":
    main()
