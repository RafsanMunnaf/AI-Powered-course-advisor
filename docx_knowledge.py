"""
docx_knowledge.py
─────────────────
Extracts the full text from the Fast Sales AI Chatbox Developer Manual (.docx)
and caches it so it can be injected into the chatbot's system prompt.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

_logger = logging.getLogger(__name__)

# Default path: same folder as this script
_DEFAULT_DOCX_PATH = Path(__file__).resolve().parent / "Fast Sales AI  Chatbox Developer Manual.docx"


import math
import re
# Optional fuzzy matching – rapidfuzz is a tiny pure‑Python library; if unavailable we fall back to exact matching.
try:
    # pyright: ignore[reportMissingImports]
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None

# Simple synonym map for common terms used in the manual
SYNONYM_MAP = {
    "refund": ["repayment", "reimburse", "reimbursement", "money back"],
    "certificate": ["cert", "certifcate", "diploma", "credential"],
    "name": ["identifier", "full name", "surname"],
    "change": ["modify", "alter", "update"],
}

def expand_query_terms(query: str) -> list[str]:
    """Return a list containing the original words plus any synonyms defined in SYNONYM_MAP."""
    terms = []
    for word in query.lower().split():
        terms.append(word)
        if word in SYNONYM_MAP:
            terms.extend(SYNONYM_MAP[word])
    # De‑duplicate while preserving order
    seen = set()
    return [t for t in terms if not (t in seen or seen.add(t))]

def fuzzy_match_score(text: str, query_terms: list[str]) -> float:
    """Return the best partial fuzzy match score (0‑100) for any term in *query_terms* against *text*.
    If rapidfuzz is unavailable we fall back to a simple 0/1 exact‑match score.
    """
    if not query_terms:
        return 0.0
    if fuzz is None:
        # Simple fallback – count exact term occurrences
        text_lc = text.lower()
        return max(text_lc.count(term) for term in query_terms)
    text_lc = text.lower()
    best = 0.0
    for term in query_terms:
        score = fuzz.partial_ratio(term, text_lc)
        if score > best:
            best = score
    return best



@lru_cache(maxsize=1)
def _extract_docx_text(docx_path: str) -> str:
    """Read every paragraph from the .docx and return as a single string, with txt fallback."""
    # Check if a text version exists next to the docx first
    txt_path = Path(docx_path).parent / "developer_manual.txt"
    if not txt_path.exists():
        txt_path = Path(docx_path).parent / "manual_text.txt"
        
    if txt_path.exists():
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
                _logger.info(
                    "Loaded developer manual from TXT fallback: %d chars", len(text)
                )
                return text
        except Exception as exc:
            _logger.warning("Failed to read TXT fallback: %s", exc)

    try:
        from docx import Document  # python-docx
    except ImportError:
        _logger.warning(
            "python-docx is not installed. Run: pip install python-docx"
        )
        return ""

    path = Path(docx_path)
    if not path.exists():
        _logger.warning("DOCX not found at %s — skipping.", path)
        return ""

    try:
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        _logger.info(
            "Loaded developer manual: %d chars from %s", len(text), path.name
        )
        return text
    except Exception as exc:
        _logger.error("Failed to read DOCX: %s", exc)
        return ""


def get_docx_knowledge_string(docx_path: Path | str | None = None) -> str:
    """Return the full developer-manual text (cached after first call)."""
    resolved = str(docx_path or _DEFAULT_DOCX_PATH)
    return _extract_docx_text(resolved)


# ── RAG Retrieval Implementation ─────────────────────────────────────────────

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'he', 'him', 'his', 
    'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'need', 'needs', 'back', 'please', 'get', 'give', 
    'want', 'wants', 'tell', 'show', 'ask', 'say', 'said'
}


def stem_word(w: str) -> str:
    if w.endswith('s') and len(w) > 3:
        if w.endswith('es'):
            if w.endswith('ies'):
                return w[:-3] + 'y'
            return w[:-2]
        return w[:-1]
    return w


class SimpleDocxRetriever:
    def __init__(self, docx_text: str):
        self.chunks = self._chunk_text(docx_text)
        self.processed_chunks = [self._tokenize(c) for c in self.chunks]
        self.doc_count = len(self.chunks)
        self.vocab = {}
        for doc in self.processed_chunks:
            unique_words = set(doc)
            for w in unique_words:
                self.vocab[w] = self.vocab.get(w, 0) + 1
                
        self.idf = {}
        for w, freq in self.vocab.items():
            self.idf[w] = math.log((self.doc_count + 1) / (freq + 0.5)) + 1.0

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        current_chunk = []
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            is_new_qa = stripped.upper().startswith("Q:") or stripped.upper().startswith("Q :")
            is_header = (
                stripped.isupper() 
                and len(stripped) > 8
                and any(word in stripped for word in ["POLICY", "LIMITATIONS", "OVERVIEW", "QUESTIONS", "GUIDELINES", "RULES", "INFORMATION", "REQUIREMENTS"])
                and not stripped.startswith("CTA")
                and "LINK" not in stripped
                and "BUTTON" not in stripped
                and "👉" not in stripped
            )
            
            if (is_new_qa or is_header) and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                
            current_chunk.append(line)
            
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return [stem_word(w) for w in words if len(w) > 1 or w.isdigit()]

    def retrieve(self, query: str, top_n: int = 5) -> list[str]:
        # Expand query with synonyms and run fuzzy matching for robustness
        expanded_terms = expand_query_terms(query)
        # Tokenize for TF‑IDF scoring (still useful for exact matches)
        query_words = [w for w in self._tokenize(query) if w not in STOP_WORDS]
        if not query_words:
            query_words = self._tokenize(query)
        scores = []
        for idx, doc in enumerate(self.processed_chunks):
            # Classic TF‑IDF score
            tfidf_score = 0.0
            for qw in query_words:
                if qw in doc:
                    tf = doc.count(qw) / len(doc)
                    idf = self.idf.get(qw, 0.0)
                    tfidf_score += tf * idf
            # Fuzzy match score on the raw chunk text
            fuzzy_score = fuzzy_match_score(self.chunks[idx], expanded_terms)
            # Blend the two scores (weight TF‑IDF slightly more)
            combined = tfidf_score * 1.5 + fuzzy_score * 0.5
            scores.append((idx, combined))
        # Highest combined scores first
        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved: list[str] = []
        for idx, score in scores[:top_n]:
            if score > 0:
                retrieved.append(self.chunks[idx])
        return retrieved


_RETRIEVER_INSTANCE: SimpleDocxRetriever | None = None
_GLOBAL_RULES: str = ""


def _init_knowledge():
    global _RETRIEVER_INSTANCE, _GLOBAL_RULES
    if _RETRIEVER_INSTANCE is not None:
        return
        
    full_text = get_docx_knowledge_string()
    parts = full_text.split("\nQ:", 1)
    
    _GLOBAL_RULES = parts[0]
    faq_part = "Q:" + parts[1] if len(parts) > 1 else ""
    
    _RETRIEVER_INSTANCE = SimpleDocxRetriever(faq_part)


def get_global_rules() -> str:
    _init_knowledge()
    return _GLOBAL_RULES


def get_relevant_chunks(query: str, top_n: int = 5) -> list[str]:
    _init_knowledge()
    if _RETRIEVER_INSTANCE is None:
        return []
    return _RETRIEVER_INSTANCE.retrieve(query, top_n)
