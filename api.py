import io
import os

from flask import Flask, jsonify, request

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from document_analysis import analyze_classroom_submissions, analyze_documents


app = Flask(__name__)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def get_extension(filename):
    return os.path.splitext(filename or "")[1].lower()


def extract_text_from_upload(uploaded_file):
    filename = uploaded_file.filename or ""
    extension = get_extension(filename)

    if extension not in ALLOWED_EXTENSIONS:
        supported_formats = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type for {filename}. Use one of: {supported_formats}.")

    file_bytes = uploaded_file.read()

    if extension == ".txt":
        try:
            text = file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ValueError(f"The text file '{filename}' must be UTF-8 encoded.") from error
        return text

    if extension == ".pdf":
        if PdfReader is None:
            raise ValueError("PDF support is unavailable because 'pypdf' is not installed in the current Python environment.")
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as error:
            raise ValueError(f"Could not read PDF file: {filename}") from error
        return "\n".join(pages).strip()

    if extension == ".docx":
        if Document is None:
            raise ValueError("DOCX support is unavailable because 'python-docx' is not installed in the current Python environment.")
        try:
            document = Document(io.BytesIO(file_bytes))
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
        except Exception as error:
            raise ValueError(f"Could not read DOCX file: {filename}") from error
        return "\n".join(paragraphs).strip()

    raise ValueError(f"Unsupported file type for {filename}.")


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
    document_files = request.files.getlist("documents")
    query_text_input = request.form.get("query_text", "").strip()
    test_text_input = request.form.get("test_text", "").strip()
    test_name_input = request.form.get("test_name", "").strip() or "typed_test.txt"
    query_name_input = request.form.get("query_name", "").strip() or "typed_original.txt"

    if query_file is not None and query_file.filename:
        try:
            query_text = extract_text_from_upload(query_file)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        if not query_text:
            return jsonify({"error": "The query document is empty."}), 400

        documents = []
        for document_file in document_files:
            if not document_file.filename:
                continue

            try:
                document_text = extract_text_from_upload(document_file)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400

            if document_text and document_file.filename != query_file.filename:
                documents.append({"name": document_file.filename, "text": document_text})

        if not documents:
            return jsonify({"error": "At least one comparison .txt, .pdf, or .docx file is required."}), 400

        query_name = query_file.filename
    else:
        if not query_text_input:
            return jsonify({"error": "Typed original text is required."}), 400
        if not test_text_input:
            return jsonify({"error": "Typed test text is required."}), 400

        query_text = query_text_input
        query_name = query_name_input
        documents = [{"name": test_name_input, "text": test_text_input}]

    result = analyze_documents(query_text, query_name, documents)
    return jsonify(
        {
            "query_name": result["query_name"],
            "document_count": result["document_count"],
            "vocabulary_size": result["vocabulary_size"],
            "similar_count": result["similar_count"],
            "paraphrased_count": result["paraphrased_count"],
            "interpretation": result["interpretation"],
            "ranked_documents": result["ranked_documents"],
        }
    )


@app.route("/professor/ranking", methods=["POST", "OPTIONS"])
@app.route("/api/professor/ranking", methods=["POST", "OPTIONS"])
def professor_ranking():
    if request.method == "OPTIONS":
        return ("", 204)

    batch_name = request.form.get("batch_name", "").strip() or "Untitled class activity"
    document_files = request.files.getlist("batch_documents")

    documents = []
    for document_file in document_files:
        if not document_file.filename:
            continue

        try:
            document_text = extract_text_from_upload(document_file)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        if document_text:
            documents.append({"name": document_file.filename, "text": document_text})

    if not documents:
        return jsonify({"error": "At least one student submission .txt, .pdf, or .docx file is required."}), 400

    result = analyze_classroom_submissions(batch_name, documents)
    return jsonify(
        {
            "batch_name": result["batch_name"],
            "submission_count": result["submission_count"],
            "vocabulary_size": result["vocabulary_size"],
            "interpretation": result["interpretation"],
            "influence_summary": result["influence_summary"],
            "ranked_submissions": result["ranked_submissions"],
            "influence_findings": result["influence_findings"],
            "similarity_heat_map": result["similarity_heat_map"],
        }
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    return jsonify({"error": f"Server error: {str(error)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
