# Document Analysis Foundation

This project is a foundational document analysis system based on TF-IDF and cosine similarity, with both Python and web-based front ends.

## What it does

- Accepts one `.txt`, `.pdf`, or `.docx` query document
- Loads multiple `.txt`, `.pdf`, or `.docx` comparison documents from a folder
- Tokenizes and analyzes the text
- Builds TF, DF, IDF, and TF-IDF values
- Computes cosine similarity and angle
- Ranks the documents from most relevant to least relevant
- Produces a worded recommendation instead of only raw scores

## Files

- `main.py` - command-line entry point
- `document_analysis.py` - reusable analysis logic
- `api.py` - Flask API for the web frontend
- `frontend.py` - Tkinter desktop interface
- `web/` - Vite frontend

## Install dependency

```bash
pip install -r requirements.txt
```

## How to run

Command-line version:

```bash
python main.py --query-file sample_query.txt --documents-folder sample_docs
```

Desktop interface:

```bash
python frontend.py
```

Web version:

Terminal 1:

```bash
python api.py
```

Terminal 2:

```bash
cd web
npm install
npm run dev
```

Then open the local Vite URL shown in the terminal, usually `http://localhost:5173`.
The frontend now uses a built-in Vite proxy, so you do not need to type the API URL manually in the page.

## Why this is a good foundation

This version already goes beyond simple score printing because it:

- ranks documents
- labels relevance levels
- identifies top contributing terms
- generates an interpretation statement

It is designed so you can later extend it with:

- stopword removal
- stemming or lemmatization
- classification
- clustering
- front-end interface
- smarter recommendation rules
