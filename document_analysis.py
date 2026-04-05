import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class AnalysisResult:
    query_name: str
    ranked_documents: List[Dict[str, object]]
    vocabulary_size: int
    document_count: int
    interpretation: str


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


def generate_interpretation(query_name: str, ranked_documents: Sequence[Dict[str, object]]) -> str:
    best_match = ranked_documents[0]
    best_name = best_match["document_name"]
    best_score = best_match["cosine_similarity"]
    level = best_match["relevance_level"]
    terms = best_match["top_terms"]

    if terms:
        reason = f"The strongest shared terms are: {', '.join(terms)}."
    else:
        reason = "There are no strong overlapping weighted terms."

    return (
        f"Best match for '{query_name}' is '{best_name}' with a cosine similarity of "
        f"{best_score:.4f}, which indicates {level}. {reason}"
    )


def analyze_documents(query_text: str, query_name: str, documents: Sequence[Dict[str, str]]) -> AnalysisResult:
    combined_documents = [{"name": query_name, "text": query_text}] + list(documents)
    tokenized_documents = [tokenize(doc["text"]) for doc in combined_documents]
    vocabulary = build_vocabulary(tokenized_documents)

    tf_matrix = compute_tf(tokenized_documents, vocabulary)
    df_vector = compute_df(tokenized_documents, vocabulary)
    idf_vector = compute_idf(df_vector, len(combined_documents))
    tfidf_matrix = compute_tfidf(tf_matrix, idf_vector)

    query_vector = tfidf_matrix[0]
    ranked_documents = []

    for index, document in enumerate(combined_documents[1:], start=1):
        document_vector = tfidf_matrix[index]
        similarity = cosine_similarity(query_vector, document_vector)
        ranked_documents.append(
            {
                "document_name": document["name"],
                "cosine_similarity": similarity,
                "angle_degrees": angle_from_cosine(similarity),
                "relevance_level": interpret_similarity(similarity),
                "top_terms": top_contributing_terms(vocabulary, query_vector, document_vector),
            }
        )

    ranked_documents.sort(key=lambda item: item["cosine_similarity"], reverse=True)

    return AnalysisResult(
        query_name=query_name,
        ranked_documents=ranked_documents,
        vocabulary_size=len(vocabulary),
        document_count=len(documents),
        interpretation=generate_interpretation(query_name, ranked_documents),
    )
