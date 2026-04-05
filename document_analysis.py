import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model: SentenceTransformer | None = None


@dataclass
class AnalysisResult:
    query_name: str
    ranked_documents: List[Dict[str, object]]
    vocabulary_size: int
    document_count: int
    similar_count: int
    paraphrased_count: int
    interpretation: str
    semantic_model: str


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def load_documents_from_folder(folder_path: str) -> List[Dict[str, str]]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    documents = []
    for file_path in sorted(folder.glob("*.txt")):
        text = load_text_file(file_path).strip()
        if text:
            documents.append({"name": file_path.name, "text": text})

    if not documents:
        raise ValueError("No non-empty .txt documents were found in the folder.")

    return documents


def build_vocabulary(tokenized_documents: Sequence[Sequence[str]]) -> List[str]:
    return sorted({token for tokens in tokenized_documents for token in tokens})


def compute_tf(tokenized_documents: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> List[List[float]]:
    matrix = []
    for tokens in tokenized_documents:
        counts = Counter(tokens)
        matrix.append([float(counts[word]) for word in vocabulary])
    return matrix


def compute_df(tokenized_documents: Sequence[Sequence[str]], vocabulary: Sequence[str]) -> List[float]:
    df = []
    for word in vocabulary:
        count = sum(1 for tokens in tokenized_documents if word in tokens)
        df.append(float(count))
    return df


def compute_idf(df_vector: Sequence[float], document_count: int) -> List[float]:
    # Smoothed IDF avoids zeroing out terms that appear in every document.
    # This is especially important in small document sets where identical files
    # would otherwise produce all-zero TF-IDF vectors and a cosine score of 0.
    return [math.log((1 + document_count) / (1 + df)) + 1.0 for df in df_vector]


def compute_tfidf(tf_matrix: Sequence[Sequence[float]], idf_vector: Sequence[float]) -> List[List[float]]:
    tfidf_matrix = []
    for row in tf_matrix:
        tfidf_matrix.append([tf * idf for tf, idf in zip(row, idf_vector)])
    return tfidf_matrix


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def angle_from_cosine(cosine_value: float) -> float:
    clipped = max(-1.0, min(1.0, cosine_value))
    return math.degrees(math.acos(clipped))


def top_contributing_terms(
    vocabulary: Sequence[str],
    query_vector: Sequence[float],
    document_vector: Sequence[float],
    top_n: int = 5,
) -> List[str]:
    contributions = []
    for term, query_weight, document_weight in zip(vocabulary, query_vector, document_vector):
        contribution = query_weight * document_weight
        if contribution > 0:
            contributions.append((term, contribution))

    contributions.sort(key=lambda item: item[1], reverse=True)
    return [term for term, _ in contributions[:top_n]]


def interpret_similarity(score: float) -> str:
    if score >= 0.75:
        return "very strong relevance"
    if score >= 0.50:
        return "strong relevance"
    if score >= 0.25:
        return "moderate relevance"
    if score > 0:
        return "low relevance"
    return "no meaningful relevance"


def interpret_semantic_similarity(score: float) -> str:
    if score >= 0.85:
        return "very strong semantic match"
    if score >= 0.72:
        return "strong semantic match"
    if score >= 0.58:
        return "moderate semantic match"
    if score >= 0.40:
        return "weak semantic match"
    return "limited semantic match"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def compute_paraphrase_score(tfidf_score: float, semantic_score: float) -> float:
    semantic_component = semantic_score * 80.0
    lexical_component = tfidf_score * 15.0
    paraphrase_bonus = max(0.0, semantic_score - tfidf_score) * 35.0

    if semantic_score >= 0.90 and tfidf_score >= 0.82:
        return 98.0

    return float(clamp(semantic_component + lexical_component + paraphrase_bonus, 0.0, 100.0))


def interpret_paraphrase_score(score: float, tfidf_score: float, semantic_score: float) -> str:
    if semantic_score >= 0.90 and tfidf_score >= 0.82:
        return "Near-duplicate wording"
    if score >= 82 and tfidf_score < 0.65:
        return "Likely paraphrased"
    if score >= 82:
        return "Very strong match"
    if score >= 58:
        return "Possibly paraphrased"
    if semantic_score >= 0.72 and tfidf_score >= 0.55:
        return "Close in meaning and wording"
    if semantic_score >= 0.45:
        return "Some shared meaning"
    return "Unlikely paraphrase"


def describe_relationship(tfidf_score: float, semantic_score: float) -> str:
    if semantic_score >= 0.72 and tfidf_score < 0.50:
        return "Likely paraphrased: the documents use different wording but express strongly similar meaning."
    if semantic_score >= 0.72 and tfidf_score >= 0.50:
        return "Strong match in both wording and meaning."
    if semantic_score >= 0.58 and tfidf_score < 0.25:
        return "Possible paraphrase: semantic similarity is stronger than direct word overlap."
    if tfidf_score >= 0.50 and semantic_score < 0.58:
        return "The documents share visible wording, but the overall semantic match is weaker."
    if semantic_score >= 0.40:
        return "There is some thematic similarity, but the match is not strong."
    return "The documents do not appear closely related in wording or meaning."


def explain_paraphrase(tfidf_score: float, semantic_score: float, top_terms: Sequence[str]) -> List[str]:
    evidence = []

    if semantic_score >= 0.72:
        evidence.append(
            f"The embedding similarity is {semantic_score:.4f}, which means the two documents are close in meaning."
        )
    elif semantic_score >= 0.58:
        evidence.append(
            f"The embedding similarity is {semantic_score:.4f}, which suggests moderate semantic overlap."
        )
    else:
        evidence.append(
            f"The embedding similarity is {semantic_score:.4f}, so the semantic match is limited."
        )

    if tfidf_score < 0.30:
        evidence.append(
            f"The TF-IDF overlap is only {tfidf_score:.4f}, so the wording changed a lot even though the meaning may still align."
        )
    elif tfidf_score < 0.55:
        evidence.append(
            f"The TF-IDF overlap is {tfidf_score:.4f}, which means some words overlap, but much of the phrasing is different."
        )
    else:
        evidence.append(
            f"The TF-IDF overlap is {tfidf_score:.4f}, so the documents still share a lot of direct wording."
        )

    if top_terms:
        evidence.append(
            f"Shared weighted terms include {', '.join(top_terms)}, which shows the common language the model still found."
        )
    else:
        evidence.append(
            "There were no strong shared weighted terms, which often happens when a text is heavily reworded."
        )

    return evidence


def summarize_text(text: str, max_words: int = 28) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def generate_interpretation(query_name: str, ranked_documents: Sequence[Dict[str, object]]) -> str:
    best_match = ranked_documents[0]
    best_name = best_match["document_name"]
    paraphrase_score = best_match["paraphrase_score"]
    verdict = best_match["paraphrase_label"]
    relationship = best_match["relationship_summary"]

    return (
        f"Best match for '{query_name}' is '{best_name}'. The paraphrase gauge is "
        f"{paraphrase_score:.0f}%, classified as {verdict.lower()}. {relationship}"
    )


def analyze_documents(query_text: str, query_name: str, documents: Sequence[Dict[str, str]]) -> AnalysisResult:
    combined_documents = [{"name": query_name, "text": query_text}] + list(documents)
    tokenized_documents = [tokenize(doc["text"]) for doc in combined_documents]
    vocabulary = build_vocabulary(tokenized_documents)
    tf_matrix = compute_tf(tokenized_documents, vocabulary)
    df_vector = compute_df(tokenized_documents, vocabulary)
    idf_vector = compute_idf(df_vector, len(combined_documents))
    tfidf_matrix = compute_tfidf(tf_matrix, idf_vector)
    embedding_matrix = get_embedding_model().encode(
        [doc["text"] for doc in combined_documents],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_vector = tfidf_matrix[0]
    query_embedding = embedding_matrix[0]
    ranked_documents = []

    for index, document in enumerate(combined_documents[1:], start=1):
        document_vector = tfidf_matrix[index]
        tfidf_similarity = float(cosine_similarity(query_vector, document_vector))
        semantic_similarity = float(cosine_similarity(query_embedding, embedding_matrix[index]))
        ranking_score = float((semantic_similarity * 0.7) + (tfidf_similarity * 0.3))
        top_terms = top_contributing_terms(vocabulary, query_vector, document_vector)
        paraphrase_score = compute_paraphrase_score(tfidf_similarity, semantic_similarity)
        ranked_documents.append(
            {
                "document_name": document["name"],
                "document_summary": summarize_text(document["text"]),
                "ranking_score": ranking_score,
                "tfidf_cosine_similarity": tfidf_similarity,
                "tfidf_angle_degrees": float(angle_from_cosine(tfidf_similarity)),
                "tfidf_relevance_level": interpret_similarity(tfidf_similarity),
                "semantic_cosine_similarity": semantic_similarity,
                "semantic_angle_degrees": float(angle_from_cosine(semantic_similarity)),
                "semantic_relevance_level": interpret_semantic_similarity(semantic_similarity),
                "paraphrase_score": paraphrase_score,
                "paraphrase_label": interpret_paraphrase_score(paraphrase_score, tfidf_similarity, semantic_similarity),
                "is_similar": semantic_similarity >= 0.58 or tfidf_similarity >= 0.50,
                "is_paraphrased": paraphrase_score >= 58.0,
                "relationship_summary": describe_relationship(tfidf_similarity, semantic_similarity),
                "paraphrase_explanation": explain_paraphrase(tfidf_similarity, semantic_similarity, top_terms),
                "top_terms": top_terms,
            }
        )

    ranked_documents.sort(key=lambda item: item["ranking_score"], reverse=True)

    return AnalysisResult(
        query_name=query_name,
        ranked_documents=ranked_documents,
        vocabulary_size=len(vocabulary),
        document_count=len(documents),
        similar_count=sum(1 for item in ranked_documents if item["is_similar"]),
        paraphrased_count=sum(1 for item in ranked_documents if item["is_paraphrased"]),
        interpretation=generate_interpretation(query_name, ranked_documents),
        semantic_model=EMBEDDING_MODEL_NAME,
    )
