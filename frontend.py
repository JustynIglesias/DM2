from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from document_analysis import analyze_documents, load_documents_from_folder


class DocumentAnalysisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Document Analysis System")
        self.root.geometry("980x680")
        self.root.minsize(900, 620)
        self.root.configure(bg="#f3efe6")

        self.query_path = tk.StringVar()
        self.folder_path = tk.StringVar()
        self.summary_text = tk.StringVar(value="Choose a query file and a folder of .txt documents to begin.")

        self._configure_styles()
        self._build_layout()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("App.TFrame", background="#f3efe6")
        style.configure("Card.TFrame", background="#fffaf0", relief="flat")
        style.configure("Header.TLabel", background="#f3efe6", foreground="#1e2a39", font=("Georgia", 20, "bold"))
        style.configure("Subheader.TLabel", background="#f3efe6", foreground="#4d5b6a", font=("Segoe UI", 10))
        style.configure("Label.TLabel", background="#fffaf0", foreground="#22313f", font=("Segoe UI", 10, "bold"))
        style.configure("Info.TLabel", background="#fffaf0", foreground="#2f3c4a", font=("Segoe UI", 10))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="App.TFrame", padding=18)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 14))

        ttk.Label(header, text="Document Analysis System", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="TF-IDF and cosine similarity with ranking, interpretation, and recommendation output.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        controls_card = ttk.Frame(container, style="Card.TFrame", padding=16)
        controls_card.pack(fill="x", pady=(0, 14))

        ttk.Label(controls_card, text="Query Document", style="Label.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls_card, textvariable=self.query_path, font=("Segoe UI", 10), width=85).grid(
            row=1, column=0, padx=(0, 10), pady=(4, 12), sticky="ew"
        )
        ttk.Button(controls_card, text="Browse File", command=self.choose_query_file, style="Primary.TButton").grid(
            row=1, column=1, pady=(4, 12), sticky="ew"
        )

        ttk.Label(controls_card, text="Documents Folder", style="Label.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls_card, textvariable=self.folder_path, font=("Segoe UI", 10), width=85).grid(
            row=3, column=0, padx=(0, 10), pady=(4, 12), sticky="ew"
        )
        ttk.Button(controls_card, text="Browse Folder", command=self.choose_documents_folder, style="Primary.TButton").grid(
            row=3, column=1, pady=(4, 12), sticky="ew"
        )

        ttk.Button(controls_card, text="Analyze Documents", command=self.run_analysis, style="Primary.TButton").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        controls_card.columnconfigure(0, weight=1)

        summary_card = ttk.Frame(container, style="Card.TFrame", padding=16)
        summary_card.pack(fill="x", pady=(0, 14))

        ttk.Label(summary_card, text="Analysis Summary", style="Label.TLabel").pack(anchor="w")
        ttk.Label(
            summary_card,
            textvariable=self.summary_text,
            style="Info.TLabel",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        results_card = ttk.Frame(container, style="Card.TFrame", padding=16)
        results_card.pack(fill="both", expand=True)

        ttk.Label(results_card, text="Ranked Results", style="Label.TLabel").pack(anchor="w")

        columns = ("rank", "document", "similarity", "angle", "relevance", "terms")
        self.tree = ttk.Treeview(results_card, columns=columns, show="headings", height=12)
        self.tree.heading("rank", text="Rank")
        self.tree.heading("document", text="Document")
        self.tree.heading("similarity", text="Cosine Similarity")
        self.tree.heading("angle", text="Angle")
        self.tree.heading("relevance", text="Interpretation")
        self.tree.heading("terms", text="Key Terms")

        self.tree.column("rank", width=60, anchor="center")
        self.tree.column("document", width=180, anchor="w")
        self.tree.column("similarity", width=130, anchor="center")
        self.tree.column("angle", width=110, anchor="center")
        self.tree.column("relevance", width=160, anchor="center")
        self.tree.column("terms", width=280, anchor="w")

        scrollbar = ttk.Scrollbar(results_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True, pady=(8, 0))
        scrollbar.pack(side="right", fill="y", pady=(8, 0))

        footer = ttk.Frame(container, style="App.TFrame")
        footer.pack(fill="x", pady=(12, 0))
        ttk.Label(
            footer,
            text="Input format supported in this foundation: .txt files",
            style="Subheader.TLabel",
        ).pack(anchor="w")

    def choose_query_file(self) -> None:
        selected_file = filedialog.askopenfilename(
            title="Select Query Document",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if selected_file:
            self.query_path.set(selected_file)

    def choose_documents_folder(self) -> None:
        selected_folder = filedialog.askdirectory(title="Select Documents Folder")
        if selected_folder:
            self.folder_path.set(selected_folder)

    def run_analysis(self) -> None:
        query_file = self.query_path.get().strip()
        documents_folder = self.folder_path.get().strip()

        if not query_file or not documents_folder:
            messagebox.showwarning("Missing Input", "Please choose both a query file and a documents folder.")
            return

        try:
            query_path = Path(query_file)
            if not query_path.exists():
                raise FileNotFoundError(f"Query file not found: {query_path}")

            query_text = query_path.read_text(encoding="utf-8").strip()
            if not query_text:
                raise ValueError("The selected query document is empty.")

            documents = load_documents_from_folder(documents_folder)
            documents = [doc for doc in documents if doc["name"] != query_path.name]

            if not documents:
                raise ValueError("No comparison documents remain after excluding the query file.")

            result = analyze_documents(query_text, query_path.name, documents)
            self._populate_results(result)
        except Exception as error:
            messagebox.showerror("Analysis Error", str(error))

    def _populate_results(self, result) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)

        for index, item in enumerate(result.ranked_documents, start=1):
            self.tree.insert(
                "",
                "end",
                values=(
                    index,
                    item["document_name"],
                    f"{item['cosine_similarity']:.4f}",
                    f"{item['angle_degrees']:.2f} deg",
                    item["relevance_level"],
                    ", ".join(item["top_terms"]) if item["top_terms"] else "No strong shared terms",
                ),
            )

        self.summary_text.set(
            f"Query document: {result.query_name} | Compared documents: {result.document_count} | "
            f"Vocabulary size: {result.vocabulary_size}\nRecommendation: {result.interpretation}"
        )


def main() -> None:
    root = tk.Tk()
    app = DocumentAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
