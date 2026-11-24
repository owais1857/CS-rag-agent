import json
import re
from pathlib import Path
import faiss
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "rawdata"
INDEX_DIR = BASE_DIR / "data" / "index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1500       # Characters (~350-450 tokens; stays inside BGE's 512-token limit)
CHUNK_OVERLAP = 250     # Bridges context across adjacent chunks
MIN_CHUNK_CHARS = 150   # Chunks smaller than this (e.g., orphan headings) merge forward

# Priority order of split boundaries: headings -> paragraphs -> lines -> sentences -> words
SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def clean_markdown(text: str) -> str:
    """Strip repeating PDF page headers, HTML tags, <br> breaks, and blank gaps."""
    # Remove repeating banners like "<mark>18</mark> Data Structures"
    text = re.sub(r"^\s*<mark>\s*\d+\s*</mark>[^\n]*\n?", "", text, flags=re.MULTILINE)
    # Replace table-cell line breaks (<br>) with a space to stop token/char bloat
    text = re.sub(r"<br\s*/?>", " ", text)
    # Strip leftover HTML underline/highlight tags
    text = re.sub(r"</?(?:u|mark)>", "", text)
    # Collapse repeated spaces/tabs without removing newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse large vertical gaps (e.g., where diagrams or footers were skipped)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(path: Path) -> list[dict]:
    """Extract PDF pages as Markdown, stripping headers/footers when supported."""
    try:
        return pymupdf4llm.to_markdown(
            str(path),
            page_chunks=True,
            header=False,
            footer=False,
        )
    except TypeError:
        # Fallback if an older pymupdf4llm version does not support header/footer kwargs
        return pymupdf4llm.to_markdown(str(path), page_chunks=True)


def load_pdfs_continuous(raw_dir: Path) -> list[dict]:
    """Merge cleaned pages per PDF into one stream while recording page character spans."""
    docs = []
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {raw_dir}")

    for path in pdf_files:
        print(f"Parsing {path.name} to continuous Markdown...")
        pages = extract_pdf_pages(path)

        parts = []
        page_spans = []  # List of (start_char, end_char, page_num)
        cursor = 0

        for page_num, page_data in enumerate(pages, start=1):
            cleaned = clean_markdown(page_data.get("text") or "")
            if not cleaned:
                continue
            start = cursor
            end = start + len(cleaned)
            page_spans.append((start, end, page_num))
            parts.append(cleaned)
            cursor = end + 2  # +2 accounts for the "\n\n" join separator

        full_text = "\n\n".join(parts)
        if full_text:
            docs.append({
                "source": path.name,
                "text": full_text,
                "page_spans": page_spans,
            })
    return docs


def resolve_page_label(
    start_idx: int,
    end_idx: int,
    page_spans: list[tuple[int, int, int]],
) -> str:
    """Map a chunk's character span back to a page ('17') or range ('17-18')."""
    pages = [
        p_num for p_start, p_end, p_num in page_spans
        if start_idx < p_end and end_idx > p_start
    ]
    if not pages:
        return "unknown"
    return str(pages[0]) if pages[0] == pages[-1] else f"{pages[0]}-{pages[-1]}"


def merge_orphan_headings(
    chunks: list[dict],
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    """Attach tiny heading-only chunks onto the start of the following chunk in the same PDF."""
    merged = []
    i = 0
    while i < len(chunks):
        curr = chunks[i]
        has_next_same_doc = (
            i + 1 < len(chunks) and chunks[i + 1]["source"] == curr["source"]
        )
        if len(curr["text"]) < min_chars and has_next_same_doc:
            nxt = chunks[i + 1]
            p1_start = curr["page"].split("-")[0]
            p2_end = nxt["page"].split("-")[-1]
            if "unknown" in (p1_start, p2_end):
                combined_page = nxt["page"] if p1_start == "unknown" else curr["page"]
            else:
                combined_page = p1_start if p1_start == p2_end else f"{p1_start}-{p2_end}"

            chunks[i + 1] = {
                **nxt,
                "page": combined_page,
                "text": f"{curr['text']}\n\n{nxt['text']}",
            }
        else:
            merged.append(curr)
        i += 1

    # Re-index chunk IDs sequentially so FAISS vector ID `i` matches chunks[i]
    for idx, chunk in enumerate(merged):
        chunk["chunk_id"] = f"chunk_{idx}"
    return merged


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Cross-page recursive chunking with exact page mapping and orphan-heading cleanup."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    chunks = []
    for doc in docs:
        split_docs = splitter.create_documents([doc["text"]])
        for piece_doc in split_docs:
            cleaned = piece_doc.page_content.strip()
            if not cleaned:
                continue
            start_idx = piece_doc.metadata.get("start_index", 0)
            if start_idx < 0:
                start_idx = 0
            end_idx = start_idx + len(piece_doc.page_content)
            page_label = resolve_page_label(start_idx, end_idx, doc["page_spans"])

            chunks.append({
                "chunk_id": "",  # Assigned sequentially after orphan merging
                "source": doc["source"],
                "page": page_label,
                "text": cleaned,
            })

    return merge_orphan_headings(chunks)


def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_pdfs_continuous(RAW_DIR)
    if not docs:
        raise RuntimeError(
            "PDFs were found, but no text was extracted. "
            "Check if your PDFs are scanned images."
        )

    chunks = chunk_documents(docs)
    total_pages = sum(len(d["page_spans"]) for d in docs)
    lengths = [len(c["text"]) for c in chunks]
    print(
        f"\nExtracted {total_pages} non-empty pages across {len(docs)} PDFs -> "
        f"{len(chunks)} chunks "
        f"(min={min(lengths)}, avg={sum(lengths)//len(lengths)}, max={max(lengths)} chars)"
    )

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    # L2-normalize vectors so FAISS Inner Product == Cosine Similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "cs.faiss"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDone! Saved {index.ntotal} vectors to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()