from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt


# -----------------------------
# Configuration (no argparse) [[memory:9299018]] [[memory:9301428]]
# -----------------------------
MODEL_NAME: str = "bert-base-uncased"
RANDOM_SEED: int = 42
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Heuristics to analyze (0..1 Spotify features; aligned with backend’s usage)
HEURISTICS: List[str] = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
]

# Context sentences per heuristic. You can edit or extend freely.
CONTEXTS: Dict[str, List[str]] = {
    "danceability": [
        "This track's danceability is through the roof; it makes crowds move.",
        "The song's danceability is low, feeling rigid and arrhythmic.",
    ],
    "energy": [
        "Explosive drums and distorted guitars push the song's energy high.",
        "Sparse arrangement and soft dynamics keep its energy subdued.",
    ],
    "valence": [
        "Major harmonies and uplifting melodies increase the track's valence.",
        "Somber textures and minor cadences reduce the song's valence.",
    ],
    "acousticness": [
        "Delicate fingerpicking and natural room tone raise the acousticness.",
        "Heavy processing and synthetic layers lower the track's acousticness.",
    ],
    "instrumentalness": [
        "An extended solo and no vocals boost the instrumentalness of the piece.",
        "A prominent lead singer keeps the instrumentalness of this song low.",
    ],
    "liveness": [
        "Crowd noise and mic bleed suggest high liveness in this recording.",
        "Studio isolation and overdubs keep liveness at a minimum.",
    ],
    "speechiness": [
        "Spoken interludes and rap verses increase the track's speechiness.",
        "Sustained melodies and few spoken words keep speechiness low.",
    ],
}


def _ensure_output_dirs(base_dir: Path) -> Tuple[Path, Path]:
    output_dir = base_dir / "output"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, plots_dir


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_phrase(tokenizer: AutoTokenizer, phrase: str) -> List[str]:
    # Tokenize without special tokens to match within sequences
    return tokenizer.tokenize(phrase, add_special_tokens=False)


def find_subsequence_indices(sequence: List[str], subseq: List[str]) -> List[Tuple[int, int]]:
    """
    Find all (start, end_exclusive) index ranges where subseq appears in sequence.
    """
    if not sequence or not subseq or len(subseq) > len(sequence):
        return []
    matches: List[Tuple[int, int]] = []
    first = subseq[0]
    max_start = len(sequence) - len(subseq) + 1
    for i in range(max_start):
        if sequence[i] != first:
            continue
        if sequence[i : i + len(subseq)] == subseq:
            matches.append((i, i + len(subseq)))
    return matches


def get_token_embedding_for_phrase(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    text: str,
    phrase: str,
    device: str,
) -> np.ndarray:
    """
    Return the average of last-layer hidden states over subword tokens that match 'phrase' in 'text'.
    If multiple occurrences exist, averages across all occurrences.
    """
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state: torch.Tensor = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden]

    full_tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    phrase_tokens: List[str] = tokenize_phrase(tokenizer, phrase)

    # Locate phrase subword spans
    spans = find_subsequence_indices(full_tokens, phrase_tokens)
    if not spans:
        # If we can't find an exact subsequence (rare), fallback: take the mean of all tokens of the word split
        # by stripping wordpieces "##" and matching joined form.
        joined = "".join([t.replace("##", "") for t in full_tokens])
        if phrase in joined:
            # As a conservative fallback, use CLS representation
            return last_hidden_state[0].cpu().numpy()
        # Ultimate fallback: mean over all tokens
        return last_hidden_state.mean(dim=0).cpu().numpy()

    # Collect embeddings for all occurrences and subword pieces
    collected: List[torch.Tensor] = []
    for start, end in spans:
        collected.append(last_hidden_state[start:end].mean(dim=0))
    phrase_vec: torch.Tensor = torch.stack(collected, dim=0).mean(dim=0)
    return phrase_vec.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))


def plot_pca(points: Dict[str, np.ndarray], title: str, save_path: Path) -> None:
    labels = list(points.keys())
    X = np.stack([points[k] for k in labels], axis=0)
    # Safe PCA to 2D
    n_components = min(2, X.shape[0], X.shape[1])
    if n_components < 2:
        # Not enough points; skip plotting
        return
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(5, 4))
    for i, label in enumerate(labels):
        x, y = X2[i, 0], X2[i, 1]
        plt.scatter(x, y, s=60)
        plt.text(x + 0.01, y + 0.01, label, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def main() -> None:
    set_seed(RANDOM_SEED)
    base_dir = Path(__file__).resolve().parent
    output_dir, plots_dir = _ensure_output_dirs(base_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # Storage
    embeddings_out: Dict[str, Dict[str, List[float]]] = {}
    sims_rows: List[Tuple[str, str, str, float]] = []  # (heuristic, a_label, b_label, cosine)

    for heuristic in HEURISTICS:
        # Isolated token (by itself)
        isolated_text = heuristic
        v_isolated = get_token_embedding_for_phrase(model, tokenizer, isolated_text, heuristic, DEVICE)

        # Contexts (fallback to empty list if not provided)
        context_sentences = CONTEXTS.get(heuristic, [])
        context_vectors: Dict[str, np.ndarray] = {}

        for idx, sent in enumerate(context_sentences, start=1):
            label = f"context_{idx}"
            vec = get_token_embedding_for_phrase(model, tokenizer, sent, heuristic, DEVICE)
            context_vectors[label] = vec

        # Save vectors
        embeddings_out[heuristic] = {
            "isolated": v_isolated.tolist(),
            **{k: v.tolist() for k, v in context_vectors.items()},
        }

        # Similarities: isolated vs each context, and context vs context
        for k, v in context_vectors.items():
            sims_rows.append((heuristic, "isolated", k, cosine_similarity(v_isolated, v)))
        ctx_items = list(context_vectors.items())
        for i in range(len(ctx_items)):
            for j in range(i + 1, len(ctx_items)):
                ki, vi = ctx_items[i]
                kj, vj = ctx_items[j]
                sims_rows.append((heuristic, ki, kj, cosine_similarity(vi, vj)))

        # PCA plot per heuristic
        points: Dict[str, np.ndarray] = {"isolated": v_isolated}
        points.update(context_vectors)
        plot_pca(points, title=f"PCA: {heuristic}", save_path=plots_dir / f"{heuristic}.png")

    # Write JSON with raw vectors
    with (output_dir / "embeddings.json").open("w", encoding="utf-8") as f:
        json.dump(embeddings_out, f, ensure_ascii=False, indent=2)

    # Write TSV of similarities
    with (output_dir / "similarities.tsv").open("w", encoding="utf-8") as f:
        f.write("heuristic\ta_label\tb_label\tcosine_similarity\n")
        for heuristic, a, b, cs in sims_rows:
            f.write(f"{heuristic}\t{a}\t{b}\t{cs:.6f}\n")

    # Console summary
    print("Finished computing contextual embeddings.")
    print(f"- JSON vectors: {output_dir / 'embeddings.json'}")
    print(f"- Similarities: {output_dir / 'similarities.tsv'}")
    print(f"- Plots: {plots_dir} (one PNG per heuristic)")


if __name__ == "__main__":
    main()


