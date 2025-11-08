## Spotify audio heuristics: contextual embeddings and inspection

This note explains:
- **Heuristics used in your backend**
- **Standard Spotify audio features** (0–1 range)
- **How the provided script extracts contextual token embeddings** for each heuristic using a transformer, then compares and visualizes differences across contexts.
- **Quickstart** to run the experiment.

### Heuristics found in your backend
Your backend maps emotions to audio feature ranges in `backend/services/emotion_mapper.py`. These include `valence`, `energy`, `danceability`, `acousticness`, `instrumentalness`, plus some non-0–1 features like `tempo` and `loudness`.

Relevant snippets:

```14:20:emorec/backend/services/emotion_mapper.py
            EmotionType.HAPPY: {
                "valence": (0.6, 1.0),
                "energy": (0.5, 1.0),
                "danceability": (0.5, 1.0),
                "tempo": (100, 180),
            },
```

```21:30:emorec/backend/services/emotion_mapper.py
            EmotionType.SAD: {
                "valence": (0.0, 0.4),
                "energy": (0.0, 0.5),
                "acousticness": (0.3, 1.0),
                "tempo": (60, 100),
            },
            EmotionType.ENERGETIC: {
                "energy": (0.7, 1.0),
                "danceability": (0.6, 1.0),
                "tempo": (120, 200),
            },
```

```31:41:emorec/backend/services/emotion_mapper.py
            EmotionType.CALM: {
                "valence": (0.3, 0.7),
                "energy": (0.0, 0.4),
                "acousticness": (0.4, 1.0),
                "tempo": (60, 100),
            },
            EmotionType.ANGRY: {
                "valence": (0.0, 0.3),
                "energy": (0.7, 1.0),
                "loudness": (-10, 0),
                "tempo": (120, 180),
            },
```

```43:70:emorec/backend/services/emotion_mapper.py
            EmotionType.MELANCHOLIC: {
                "valence": (0.0, 0.4),
                "energy": (0.2, 0.5),
                "acousticness": (0.4, 1.0),
                "instrumentalness": (0.0, 0.7),
            },
            EmotionType.HOPEFUL: {
                "valence": (0.4, 0.8),
                "energy": (0.4, 0.7),
                "acousticness": (0.2, 0.8),
            },
            EmotionType.ROMANTIC: {
                "valence": (0.4, 0.8),
                "energy": (0.2, 0.6),
                "acousticness": (0.3, 0.9),
                "danceability": (0.3, 0.7),
            },
            EmotionType.ANXIOUS: {
                "valence": (0.2, 0.5),
                "energy": (0.5, 0.9),
                "tempo": (100, 160),
            },
            EmotionType.PEACEFUL: {
                "valence": (0.4, 0.8),
                "energy": (0.0, 0.3),
                "acousticness": (0.5, 1.0),
                "instrumentalness": (0.2, 1.0),
            },
```

For contextual embedding analysis, we focus on the 0–1 features (as intended in your request):
- `danceability`
- `energy`
- `valence`
- `acousticness`
- `instrumentalness`
- Optionally: `liveness`, `speechiness` (standard in Spotify, even if not explicitly in your backend mappings)

### Spotify audio features (0–1)
- **danceability**: how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and regularity.
- **energy**: perceptual intensity and activity. Energetic tracks feel fast, loud, and noisy.
- **valence**: musical positiveness; high is positive/happy, low is negative/sad.
- **acousticness**: confidence measure of whether a track is acoustic.
- **instrumentalness**: likelihood that a track contains no vocals (rap/spoken word is vocal).
- **liveness**: likelihood of an audience presence; higher indicates live performance.
- **speechiness**: presence of spoken words; higher means more speech-like content.

### What the script does
- Uses a transformer (`bert-base-uncased`) to obtain contextualized token embeddings.
- For each heuristic word, compares the embedding:
  - **Isolated**: just the word by itself.
  - **In context**: several music-related sentences.
- Computes cosine similarities among these embeddings.
- Projects the set per-heuristic to 2D with PCA and saves a small plot to visualize how context moves the token’s latent representation.
- Writes out a small JSON with the raw vectors for further analysis.

### Default music-related context examples
You can change these in the script, but it ships with examples like:
- For `danceability`:
  - “This track’s danceability is through the roof; it makes crowds move.”
  - “The song’s danceability is low, feeling rigid and arrhythmic.”
- For `energy`:
  - “Explosive drums and distorted guitars push the song’s energy high.”
  - “Sparse arrangement and soft dynamics keep its energy subdued.”
- For `valence`:
  - “Major harmonies and uplifting melodies increase the track’s valence.”
  - “Somber textures and minor cadences reduce the song’s valence.”

The script includes similar pairs for `acousticness`, `instrumentalness`, `liveness`, and `speechiness`.

### Quickstart
1) Install dependencies (example):
```
pip install torch transformers scikit-learn matplotlib numpy
```
2) Run:
```
python emorec/testing/inspect_heuristic_embeddings.py
```
3) Outputs (created under `emorec/testing/output/`):
- `embeddings.json`: raw vectors per heuristic and context
- `similarities.tsv`: cosine similarities table
- `plots/<heuristic>.png`: PCA scatter per heuristic (isolated vs contexts)

### Notes
- This tool specializes semantics by placing each word into musical sentences, examining how attention/context changes the token representation.
- You can swap the model (e.g., `roberta-base`) in the script’s configuration at the top.
- If you want sentence-level comparisons instead, consider a sentence embedding model (e.g., from `sentence-transformers`), but token-level gives you the “word in context” latent directly.


