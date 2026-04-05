from typing import List

from flask import Flask, jsonify, request

from document_analysis import analyze_documents


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST", "OPTIONS"])
@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 204)

    query_file = request.files.get("query_file")
    document_files: List = request.files.getlist("documents")

    if query_file is None or not query_file.filename:
        return jsonify({"error": "A query .txt file is required."}), 400

    if not document_files:
        return jsonify({"error": "At least one comparison .txt file is required."}), 400

    try:
        query_text = query_file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        return jsonify({"error": "The query file must be UTF-8 encoded text."}), 400

    if not query_text:
        return jsonify({"error": "The query document is empty."}), 400

    documents = []
    for document_file in document_files:
        if not document_file.filename:
            continue

        try:
            document_text = document_file.read().decode("utf-8").strip()
        except UnicodeDecodeError:
            return jsonify({"error": f"Invalid UTF-8 text file: {document_file.filename}"}), 400

        if document_text and document_file.filename != query_file.filename:
            documents.append({"name": document_file.filename, "text": document_text})

    if not documents:
        return jsonify({"error": "No valid comparison documents remain after filtering."}), 400

    result = analyze_documents(query_text, query_file.filename, documents)
    return jsonify(
        {
            "query_name": result.query_name,
            "document_count": result.document_count,
            "vocabulary_size": result.vocabulary_size,
            "interpretation": result.interpretation,
            "ranked_documents": result.ranked_documents,
        }
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    return jsonify({"error": f"Server error: {str(error)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
