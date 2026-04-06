import math
import os
import re
from collections import Counter

import numpy as np


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_vocabulary(tokenized_documents):
    vocabulary = set()
    for tokens in tokenized_documents:
        for token in tokens:
            vocabulary.add(token)
    return sorted(vocabulary)


def compute_tf(tokenized_documents, vocabulary):
    tf_matrix = []
    for tokens in tokenized_documents:
        counts = Counter(tokens)
        row = [counts[word] for word in vocabulary]
        tf_matrix.append(row)
    return np.array(tf_matrix, dtype=float)


def compute_df(tokenized_documents, vocabulary):
    df = []
    for word in vocabulary:
        count = sum(1 for tokens in tokenized_documents if word in tokens)
        df.append(count)
    return np.array(df, dtype=float)


def compute_idf(df_vector, document_count):
    return np.array([math.log(document_count / df) if df != 0 else 0 for df in df_vector], dtype=float)


def compute_tfidf(tf_matrix, idf_vector):
    return tf_matrix * idf_vector


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 and norm_b == 0:
        return 1.0  
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def get_contribution_sort_key(item):
    return item[1]


def top_contributing_terms(vocabulary, query_vector, document_vector, top_n=5):
    contributions = []

    for term, query_weight, document_weight in zip(vocabulary, query_vector, document_vector):
        contribution = query_weight * document_weight
        if contribution > 0:
            contributions.append((term, contribution))

    contributions.sort(reverse=True, key=get_contribution_sort_key)
    return [term for term, _ in contributions[:top_n]]


def interpret_similarity(score):
    if score >= 0.85:
        return "very strong match"
    if score >= 0.72:
        return "strong match"
    if score >= 0.58:
        return "moderate match"
    if score >= 0.40:
        return "weak match"
    return "limited match"


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def compute_paraphrase_score(tfidf_score, bow_score):
    if bow_score >= 0.95:
        return 0.0
    if tfidf_score >= 0.65:
        return float(clamp((1.0 - tfidf_score) * 35.0, 0.0, 18.0))
    if tfidf_score >= 0.20:
        return float(60.0 + ((0.65 - tfidf_score) / 0.45) * 40.0)
    if tfidf_score >= 0.08:
        return float(25.0 + ((tfidf_score - 0.08) / 0.12) * 35.0)
    return 0.0


def interpret_paraphrase_score(tfidf_score, bow_score):
    if bow_score >= 0.95:
        return "Likely exact copy"
    if tfidf_score >= 0.65:
        return "Most likely not paraphrased"
    if tfidf_score >= 0.20:
        return "Likely paraphrased"
    if tfidf_score >= 0.08:
        return "Weak similarity"
    return "Not similar"


def describe_relationship(tfidf_score, bow_score):
    if bow_score >= 0.95:
        return "The bag-of-words cosine similarity is extremely high, so this document is likely a direct copy or near-copy."
    if tfidf_score >= 0.65:
        return "The TF-IDF score is above 65%, so this document is most likely not paraphrased."
    if tfidf_score >= 0.20:
        return "The TF-IDF score is below 65% but still meaningful, so this document is likely paraphrased."
    if tfidf_score >= 0.08:
        return "There is only weak overlap, so this may be loosely related rather than a clear paraphrase."
    return "The documents do not appear closely related in wording."


def explain_paraphrase(tfidf_score, bow_score, top_terms):
    evidence = []

    if tfidf_score >= 0.65:
        evidence.append(
            f"The TF-IDF cosine similarity is {tfidf_score:.4f}, which is above the 0.65 threshold for 'most likely not paraphrased'."
        )
    elif tfidf_score >= 0.20:
        evidence.append(
            f"The TF-IDF cosine similarity is {tfidf_score:.4f}, which is below the 0.65 threshold and supports a likely paraphrase judgment."
        )
    else:
        evidence.append(
            f"The TF-IDF cosine similarity is {tfidf_score:.4f}, so the weighted overlap is weak."
        )

    if bow_score >= 0.95:
        evidence.append(
            f"The bag-of-words cosine similarity is {bow_score:.4f}, which strongly suggests copy-paste or near-copy wording."
        )
    elif bow_score >= 0.60:
        evidence.append(
            f"The bag-of-words cosine similarity is {bow_score:.4f}, which shows the documents still share a lot of direct wording."
        )
    else:
        evidence.append(
            f"The bag-of-words cosine similarity is {bow_score:.4f}, which means the exact wording overlap is limited."
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


def summarize_text(text, max_words=28):
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def count_sentences(text):
    parts = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
    return max(1, len(parts))


def build_writing_feedback(word_count, sentence_count, average_sentence_length, lexical_diversity):
    feedback = []

    if word_count >= 140:
        feedback.append("developed length")
    elif word_count >= 80:
        feedback.append("adequate length")
    else:
        feedback.append("short length")

    if 12 <= average_sentence_length <= 28:
        feedback.append("balanced sentence flow")
    elif average_sentence_length < 12:
        feedback.append("very short sentences")
    else:
        feedback.append("long sentences")

    if lexical_diversity >= 0.55:
        feedback.append("varied vocabulary")
    elif lexical_diversity >= 0.40:
        feedback.append("moderate vocabulary variety")
    else:
        feedback.append("repetitive vocabulary")

    if sentence_count >= 5:
        feedback.append("good idea development")
    elif sentence_count >= 3:
        feedback.append("basic idea development")
    else:
        feedback.append("limited idea development")

    return ", ".join(feedback)


def compute_writing_quality_score(text, tokens):
    word_count = len(tokens)
    sentence_count = count_sentences(text)
    unique_words = len(set(tokens))
    lexical_diversity = (unique_words / word_count) if word_count else 0.0
    average_sentence_length = (word_count / sentence_count) if sentence_count else 0.0

    length_score = clamp((word_count / 180.0) * 28.0, 0.0, 28.0)
    sentence_flow_score = clamp(25.0 - abs(average_sentence_length - 20.0) * 1.2, 0.0, 25.0)
    vocabulary_score = clamp((lexical_diversity / 0.65) * 27.0, 0.0, 27.0)
    development_score = clamp((sentence_count / 6.0) * 20.0, 0.0, 20.0)
    total_score = round(length_score + sentence_flow_score + vocabulary_score + development_score, 2)

    return {
        "writing_quality_score": total_score,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "lexical_diversity": round(lexical_diversity, 4),
        "average_sentence_length": round(average_sentence_length, 2),
        "writing_feedback": build_writing_feedback(
            word_count, sentence_count, average_sentence_length, lexical_diversity
        ),
    }


def compute_uniqueness_score(tf_row, idf_vector):
    present_indices = [index for index, count in enumerate(tf_row) if count > 0]

    if not present_indices:
        return {
            "tfidf_uniqueness_score": 0.0,
            "unique_term_count": 0,
        }

    max_idf = max(idf_vector) if len(idf_vector) else 0.0
    average_idf = sum(idf_vector[index] for index in present_indices) / len(present_indices)
    uniqueness_score = (average_idf / max_idf) * 100.0 if max_idf > 0 else 0.0
    unique_term_count = sum(1 for index in present_indices if idf_vector[index] > 0)

    return {
        "tfidf_uniqueness_score": round(clamp(uniqueness_score, 0.0, 100.0), 2),
        "unique_term_count": unique_term_count,
    }


def compute_document_vector_metrics(tfidf_row, tf_row):
    tfidf_magnitude = float(np.linalg.norm(tfidf_row))
    bow_magnitude = float(np.linalg.norm(tf_row))

    return {
        "tfidf_score": round(tfidf_magnitude, 4),
        "tfidf_score_level": "tf-idf vector magnitude",
        "bow_score": round(bow_magnitude, 4),
        "bow_score_level": "bag-of-words vector magnitude",
    }


def classify_influence(influence_score):
    if influence_score >= 100.0:
        return "sobrang duper parehas"
    if influence_score > 80.0:
        return "halos parehas"
    if influence_score > 60.0:
        return "medyo parehas"
    if influence_score > 40.0:
        return "hindi masyado parehas"
    if influence_score > 20.0:
        return "konti lang ang parehas"
    return "walang kaparehas"


def generate_influence_decision(source_name, target_name, influence_score, tfidf_similarity, bow_similarity, shared_terms):
    if influence_score >= 100.0 or (tfidf_similarity >= 0.98 and bow_similarity >= 0.98):
        return "Failed. Halos xerox copy, same na same talaga."
    if influence_score > 80.0:
        return "Rewrite and resubmit. Halatang nag kopyahan."
    if influence_score > 60.0:
        return "Needs manual review. Medyo obvious pa eh, need i check."
    if influence_score > 40.0:
        return "Borderline similarity. May pagkakapareho pero pwede pa i-check."
    if influence_score > 20.0:
        return "Accept with caution. May konting pagkakapareho lang."
    return "Passed. Mabait na bata, magaling."


def describe_influence(source_name, target_name, influence_type):
    return (
        f"'{source_name}' and '{target_name}' received the flag '{influence_type}' based on their combined influence score."
    )


def get_finding_sort_key(item):
    return (
        item["influence_score"],
        item["tfidf_cosine_similarity"],
        item["bow_cosine_similarity"],
    )


def build_influence_findings(documents, vocabulary, tf_matrix, tfidf_matrix):
    findings = []

    for source_index in range(len(documents)):
        for target_index in range(source_index + 1, len(documents)):
            tfidf_similarity = cosine_similarity(tfidf_matrix[source_index], tfidf_matrix[target_index])
            bow_similarity = cosine_similarity(tf_matrix[source_index], tf_matrix[target_index])
            influence_score = round(((tfidf_similarity * 0.7) + (bow_similarity * 0.3)) * 100.0, 2)
            influence_type = classify_influence(influence_score)

            source_name = documents[source_index]["name"]
            target_name = documents[target_index]["name"]
            shared_terms = top_contributing_terms(
                vocabulary,
                tfidf_matrix[source_index],
                tfidf_matrix[target_index],
            )

            findings.append(
                {
                    "source_document": source_name,
                    "target_document": target_name,
                    "source_position": source_index + 1,
                    "target_position": target_index + 1,
                    "influence_type": influence_type,
                    "influence_score": influence_score,
                    "tfidf_cosine_similarity": round(tfidf_similarity, 4),
                    "bow_cosine_similarity": round(bow_similarity, 4),
                    "shared_terms": shared_terms,
                    "decision": generate_influence_decision(
                        source_name,
                        target_name,
                        influence_score,
                        tfidf_similarity,
                        bow_similarity,
                        shared_terms,
                    ),
                    "explanation": describe_influence(source_name, target_name, influence_type),
                }
            )

    findings.sort(reverse=True, key=get_finding_sort_key)
    return findings


def generate_influence_summary(batch_name, influence_findings):
    if not influence_findings:
        return f"No strong influence signals were detected yet for '{batch_name}'."

    top_finding = influence_findings[0]
    return (
        f"Top influence signal for '{batch_name}' is from '{top_finding['source_document']}' to "
        f"'{top_finding['target_document']}' and is flagged as {top_finding['influence_type']}."
    )

def generate_class_interpretation(batch_name, ranked_submissions):
    if not ranked_submissions:
        return f"No submissions were available for '{batch_name}'."

    top_submission = ranked_submissions[0]
    return (
        f"Top class ranking for '{batch_name}' is '{top_submission['document_name']}' with a writing quality score "
        f"of {top_submission['writing_quality_score']:.1f} and a TF-IDF uniqueness score of "
        f"{top_submission['tfidf_uniqueness_score']:.1f}."
    )


def build_group_topic_terms(member_indices, vocabulary, tfidf_matrix, top_n=4):
    weighted_terms = []

    for term_index, term in enumerate(vocabulary):
        if len(term) <= 2:
            continue

        total_weight = 0.0
        for member_index in member_indices:
            total_weight += tfidf_matrix[member_index][term_index]

        if total_weight > 0:
            weighted_terms.append((term, total_weight))

    weighted_terms.sort(reverse=True, key=get_contribution_sort_key)
    return [term for term, _ in weighted_terms[:top_n]]


def normalize_topic_term(term):
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith(("ches", "shes", "sses", "xes", "zes")) and len(term) > 4:
        return term[:-2]
    if term.endswith("s") and len(term) > 3:
        return term[:-1]
    return term


def build_topic_label(topic_terms, fallback):
    if not topic_terms:
        return fallback

    main_term = normalize_topic_term(topic_terms[0]).replace("_", " ").strip()
    if not main_term:
        return fallback

    return f"{main_term.title()} Topic Group"


def extract_name_topic_token(label):
    stem = os.path.splitext(os.path.basename(label))[0].lower()
    matches = re.findall(r"[a-z]+", stem)

    if not matches:
        return ""

    return normalize_topic_term(matches[0])


def build_group_display_label(member_indices, labels, topic_terms, fallback):
    name_tokens = set()

    for index in member_indices:
        token = extract_name_topic_token(labels[index])
        if token:
            name_tokens.add(token)

    if len(name_tokens) == 1:
        return f"{next(iter(name_tokens)).title()} Topic Group"

    return build_topic_label(topic_terms, fallback)


def group_singletons_by_topic(singleton_indices, labels, vocabulary, tfidf_matrix):
    topic_buckets = []

    for member_index in singleton_indices:
        topic_terms = build_group_topic_terms([member_index], vocabulary, tfidf_matrix)
        normalized_terms = set()
        for term in topic_terms:
            normalized_terms.add(normalize_topic_term(term))

        name_token = extract_name_topic_token(labels[member_index])
        matched_bucket = None

        for bucket in topic_buckets:
            bucket_name_token = bucket.get("name_token", "")

            if name_token and bucket_name_token and name_token == bucket_name_token:
                matched_bucket = bucket
                break

            if not name_token and not bucket_name_token and normalized_terms.intersection(bucket["normalized_terms"]):
                matched_bucket = bucket
                break

        if matched_bucket is None:
            topic_buckets.append(
                {
                    "members": [member_index],
                    "normalized_terms": set(normalized_terms),
                    "name_token": name_token,
                }
            )
        else:
            matched_bucket["members"].append(member_index)
            matched_bucket["normalized_terms"].update(normalized_terms)

    groups = []
    for bucket in topic_buckets:
        groups.append(bucket["members"])
    return groups


def build_similarity_heat_map(documents, tf_matrix, tfidf_matrix):
    labels = [document["name"] for document in documents]
    matrix_rows = []
    strongest_pair = None

    for source_index, source_name in enumerate(labels):
        row_values = []

        for target_index, target_name in enumerate(labels):
            if source_index == target_index:
                row_values.append(
                    {
                        "target_document": target_name,
                        "similarity_score": 1.0,
                        "tfidf_cosine_similarity": 1.0,
                        "bow_cosine_similarity": 1.0,
                        "similarity_level": "same document",
                    }
                )
                continue

            tfidf_similarity = cosine_similarity(tfidf_matrix[source_index], tfidf_matrix[target_index])
            bow_similarity = cosine_similarity(tf_matrix[source_index], tf_matrix[target_index])
            combined_similarity = float((tfidf_similarity * 0.7) + (bow_similarity * 0.3))

            row_values.append(
                {
                    "target_document": target_name,
                    "similarity_score": round(combined_similarity, 4),
                    "tfidf_cosine_similarity": round(tfidf_similarity, 4),
                    "bow_cosine_similarity": round(bow_similarity, 4),
                    "similarity_level": interpret_similarity(combined_similarity),
                }
            )

            if target_index > source_index:
                pair_data = {
                    "source_document": source_name,
                    "target_document": target_name,
                    "similarity_score": round(combined_similarity, 4),
                }
                if strongest_pair is None or pair_data["similarity_score"] > strongest_pair["similarity_score"]:
                    strongest_pair = pair_data

        matrix_rows.append({"document_name": source_name, "similarities": row_values})

    if strongest_pair:
        summary = (
            f"Strongest similarity is {strongest_pair['source_document']} <-> {strongest_pair['target_document']} "
            f"({strongest_pair['similarity_score']:.2f})."
        )
    else:
        summary = "Not enough submissions were provided to build a similarity heat map."

    return {
        "summary": summary,
        "matrix": {
            "labels": labels,
            "rows": matrix_rows,
        },
    }


def generate_interpretation(query_name, ranked_documents):
    best_match = ranked_documents[0]
    best_name = best_match["document_name"]
    influence_score = best_match["influence_score"]
    verdict = best_match["paraphrase_label"]
    relationship = best_match["relationship_summary"]

    return (
        f"Best match for '{query_name}' is '{best_name}'. The copycat gauge is "
        f"{influence_score:.0f}%, classified as {verdict.lower()}. {relationship}"
    )


def get_submission_sort_key(item):
    return (item["ranking_score"], item["writing_quality_score"])


def analyze_classroom_submissions(batch_name, documents):
    if not documents:
        raise ValueError("At least one student submission is required.")

    tokenized_documents = [tokenize(document["text"]) for document in documents]
    vocabulary = build_vocabulary(tokenized_documents)
    tf_matrix = compute_tf(tokenized_documents, vocabulary)
    df_vector = compute_df(tokenized_documents, vocabulary)
    idf_vector = compute_idf(df_vector, len(documents))
    tfidf_matrix = compute_tfidf(tf_matrix, idf_vector)
    influence_findings = build_influence_findings(documents, vocabulary, tf_matrix, tfidf_matrix)
    similarity_heat_map = build_similarity_heat_map(documents, tf_matrix, tfidf_matrix)

    ranked_submissions = []

    for index, document in enumerate(documents):
        quality_metrics = compute_writing_quality_score(document["text"], tokenized_documents[index])
        uniqueness_metrics = compute_uniqueness_score(tf_matrix[index], idf_vector)
        vector_metrics = compute_document_vector_metrics(tfidf_matrix[index], tf_matrix[index])
        ranking_score = round(
            (float(quality_metrics["writing_quality_score"]) * 0.65)
            + (float(uniqueness_metrics["tfidf_uniqueness_score"]) * 0.35),
            2,
        )

        ranked_submissions.append(
            {
                "document_name": document["name"],
                "document_summary": summarize_text(document["text"]),
                "ranking_score": ranking_score,
                **quality_metrics,
                **uniqueness_metrics,
                **vector_metrics,
            }
        )

    ranked_submissions.sort(reverse=True, key=get_submission_sort_key)

    return {
        "batch_name": batch_name,
        "submission_count": len(documents),
        "vocabulary_size": len(vocabulary),
        "ranked_submissions": ranked_submissions,
        "influence_findings": influence_findings,
        "similarity_heat_map": similarity_heat_map,
        "influence_summary": generate_influence_summary(batch_name, influence_findings),
        "interpretation": generate_class_interpretation(batch_name, ranked_submissions),
    }


def get_document_sort_key(item):
    return item["ranking_score"]


def analyze_documents(query_text, query_name, documents):
    combined_documents = [{"name": query_name, "text": query_text}] + list(documents)
    tokenized_documents = [tokenize(document["text"]) for document in combined_documents]
    vocabulary = build_vocabulary(tokenized_documents)
    tf_matrix = compute_tf(tokenized_documents, vocabulary)
    df_vector = compute_df(tokenized_documents, vocabulary)
    idf_vector = compute_idf(df_vector, len(combined_documents))
    tfidf_matrix = compute_tfidf(tf_matrix, idf_vector)

    query_vector = tfidf_matrix[0]
    query_bow_vector = tf_matrix[0]
    ranked_documents = []

    for index, document in enumerate(combined_documents[1:], start=1):
        document_vector = tfidf_matrix[index]
        bow_vector = tf_matrix[index]
        tfidf_similarity = cosine_similarity(query_vector, document_vector)
        bow_similarity = cosine_similarity(query_bow_vector, bow_vector)
        ranking_score = float((tfidf_similarity * 0.7) + (bow_similarity * 0.3))
        influence_score = round(ranking_score * 100.0, 2)
        top_terms = top_contributing_terms(vocabulary, query_vector, document_vector)
        paraphrase_score = compute_paraphrase_score(tfidf_similarity, bow_similarity)

        ranked_documents.append(
            {
                "document_name": document["name"],
                "document_summary": summarize_text(document["text"]),
                "ranking_score": ranking_score,
                "tfidf_cosine_similarity": tfidf_similarity,
                "tfidf_relevance_level": interpret_similarity(tfidf_similarity),
                "bow_cosine_similarity": bow_similarity,
                "bow_relevance_level": interpret_similarity(bow_similarity),
                "influence_score": influence_score,
                "paraphrase_score": paraphrase_score,
                "paraphrase_label": interpret_paraphrase_score(tfidf_similarity, bow_similarity),
                "is_similar": tfidf_similarity >= 0.20 or bow_similarity >= 0.50,
                "is_paraphrased": tfidf_similarity >= 0.20 and tfidf_similarity < 0.65 and bow_similarity < 0.95,
                "is_exact_copy": bow_similarity >= 0.95,
                "relationship_summary": describe_relationship(tfidf_similarity, bow_similarity),
                "paraphrase_explanation": explain_paraphrase(tfidf_similarity, bow_similarity, top_terms),
                "top_terms": top_terms,
            }
        )

    ranked_documents.sort(reverse=True, key=get_document_sort_key)

    return {
        "query_name": query_name,
        "ranked_documents": ranked_documents,
        "vocabulary_size": len(vocabulary),
        "document_count": len(documents),
        "similar_count": sum(1 for item in ranked_documents if item["is_similar"]),
        "paraphrased_count": sum(1 for item in ranked_documents if item["is_paraphrased"]),
        "interpretation": generate_interpretation(query_name, ranked_documents),
    }
