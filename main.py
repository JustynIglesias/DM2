import argparse
from pathlib import Path

from document_analysis import analyze_documents, load_documents_from_folder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Foundational document analysis system using TF-IDF and cosine similarity."
    )
    parser.add_argument(
        "--query-file",
        required=True,
        help="Path to the .txt file that will act as the reference or query document.",
    )
    parser.add_argument(
        "--documents-folder",
        required=True,
        help="Folder containing .txt files to compare against the query document.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    query_path = Path(args.query_file)
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    query_text = query_path.read_text(encoding="utf-8").strip()
    if not query_text:
        raise ValueError("The query document is empty.")

    documents = load_documents_from_folder(args.documents_folder)
    documents = [doc for doc in documents if doc["name"] != query_path.name]

    if not documents:
        raise ValueError("No comparison documents remain after excluding the query file.")

    result = analyze_documents(query_text, query_path.name, documents)

    print("DOCUMENT ANALYSIS FOUNDATION")
    print("-" * 60)
    print(f"Query document: {result.query_name}")
    print(f"Compared against: {result.document_count} documents")
    print(f"Vocabulary size: {result.vocabulary_size}")

    print("\nRANKED RESULTS")
    for index, item in enumerate(result.ranked_documents, start=1):
        terms = ", ".join(item["top_terms"]) if item["top_terms"] else "No strong shared terms"
        print(f"{index}. {item['document_name']}")
        print(f"   Cosine Similarity: {item['cosine_similarity']:.4f}")
        print(f"   Angle: {item['angle_degrees']:.4f} degrees")
        print(f"   Interpretation: {item['relevance_level']}")
        print(f"   Key Terms: {terms}")

    print("\nRECOMMENDATION")
    print(result.interpretation)


if __name__ == "__main__":
    main()
