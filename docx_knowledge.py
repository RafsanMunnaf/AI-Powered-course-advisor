"""
docx_knowledge.py
─────────────────
Extracts the full text from the Fast Sales AI Chatbox Developer Manual (.docx)
and provides exact Q&A lookup matching to ensure 100% verbatim answers.
"""

from __future__ import annotations

import logging
import math
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

_logger = logging.getLogger(__name__)

# Default path: same folder as this script
_DEFAULT_DOCX_PATH = Path(__file__).resolve().parent / "Fast Sales AI  Chatbox Developer Manual.docx"

# Optional fuzzy matching – rapidfuzz if available
try:
    import importlib
    _rf = importlib.import_module("rapidfuzz")
    fuzz = getattr(_rf, "fuzz", None)
except ImportError:  # pragma: no cover
    fuzz = None

# Simple synonym map for common terms used in the manual
SYNONYM_MAP = {
    "refund": ["repayment", "reimburse", "reimbursement", "money back", "return"],
    "certificate": ["cert", "certificate", "diploma", "credential", "certification"],
    "name": ["identifier", "full name", "surname"],
    "change": ["modify", "alter", "update"],
    "duration": ["length", "long", "hours", "time", "how long"],
    "long": ["duration", "length", "hours", "time"],
    "access": ["enroll", "enrollment", "login", "log in", "expire", "expiration"],
    "cost": ["price", "pricing", "fee", "fees", "pay", "payment", "charge"],
    "price": ["cost", "pricing", "fee", "fees", "pay", "payment", "charge"],
    "job": ["jobs", "career", "employment", "hire", "hiring", "work", "position"],
    "course": ["courses", "program", "programs", "training", "class"],
    "cancel": ["cancellation", "cancelled", "canceled"],
    "book": ["books", "textbook", "textbooks", "reading material"],
    "partner": ["partners", "partnership", "affiliate", "dealer", "dealership"],
    "community": ["forum", "group", "discussion"],
    "expire": ["expiration", "expired", "expiring", "expires", "access"],
}


def expand_query_terms(query: str) -> list[str]:
    """Return a list containing the original words plus any synonyms defined in SYNONYM_MAP."""
    terms = []
    for word in query.lower().split():
        terms.append(word)
        if word in SYNONYM_MAP:
            terms.extend(SYNONYM_MAP[word])
    seen = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


def fuzzy_match_score(text: str, query_terms: list[str]) -> float:
    """Return the best partial fuzzy match score (0‑100) for any term in *query_terms* against *text*."""
    if not query_terms:
        return 0.0
    _fuzz = fuzz
    if _fuzz is None:
        text_lc = text.lower()
        return float(max((text_lc.count(term) for term in query_terms), default=0))
    text_lc = text.lower()
    best = 0.0
    for term in query_terms:
        score = float(_fuzz.partial_ratio(term, text_lc))
        if score > best:
            best = score
    return best


@lru_cache(maxsize=1)
def _extract_docx_text(docx_path: str) -> str:
    """Read every paragraph and table from the .docx and return as a single string, with txt fallback."""
    txt_path = Path(docx_path).parent / "developer_manual.txt"
    if not txt_path.exists():
        txt_path = Path(docx_path).parent / "manual_text.txt"

    if txt_path.exists():
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
                _logger.info("Loaded developer manual from TXT fallback: %d chars", len(text))
                return text
        except Exception as exc:
            _logger.warning("Failed to read TXT fallback: %s", exc)

    try:
        from docx import Document  # python-docx
        from docx.text.paragraph import Paragraph
        from docx.table import Table
    except ImportError:
        _logger.warning("python-docx is not installed. Run: pip install python-docx")
        return ""

    path = Path(docx_path)
    if not path.exists():
        _logger.warning("DOCX not found at %s — skipping.", path)
        return ""

    try:
        doc = Document(str(path))
        lines = []
        for block in doc.element.body:
            if block.tag.endswith('p'):
                p = Paragraph(block, doc)
                if p.text.strip():
                    lines.append(p.text.strip())
            elif block.tag.endswith('tbl'):
                t = Table(block, doc)
                table_lines = []
                for row in t.rows:
                    row_str = ' | '.join(cell.text.strip().replace('\n', ' ') for cell in row.cells)
                    if row_str.strip():
                        table_lines.append(row_str)
                if table_lines:
                    lines.append('\n' + '\n'.join(table_lines) + '\n')

        text = "\n".join(lines)
        _logger.info("Loaded developer manual (including tables): %d chars from %s", len(text), path.name)
        return text
    except Exception as exc:
        _logger.error("Failed to read DOCX: %s", exc)
        return ""


def get_docx_knowledge_string(docx_path: Path | str | None = None) -> str:
    """Return the full developer-manual text (cached after first call)."""
    resolved = str(docx_path or _DEFAULT_DOCX_PATH)
    return _extract_docx_text(resolved)


# ── Q&A Extraction and Verbatim Matching Engine ──────────────────────────────

def extract_qa_pairs(full_text: str) -> list[dict[str, str]]:
    """
    Parses full document text into structured Q&A objects:
    [{ 'question': '...', 'answer': '...' }]
    """
    qa_list = []
    lines = full_text.splitlines()
    current_q = None
    current_a_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if re.match(r"^Q\s*:\s*", stripped, re.IGNORECASE):
            if current_q and current_a_lines:
                qa_list.append({
                    "question": current_q,
                    "answer": "\n".join(current_a_lines).strip()
                })
            current_q = re.sub(r"^Q\s*:\s*", "", stripped, flags=re.IGNORECASE).strip()
            current_a_lines = []
        elif current_q is not None and re.match(r"^A\s*:\s*", stripped, re.IGNORECASE):
            a_first_line = re.sub(r"^A\s*:\s*", "", stripped, flags=re.IGNORECASE).strip()
            current_a_lines = [a_first_line]
        elif current_q is not None and len(current_a_lines) > 0:
            # Skip CTA instruction lines — they are not part of the answer content
            if (
                stripped.startswith("👉")
                or stripped.startswith("CTA")
                or stripped.startswith("RECOMMENDED CTA")
            ):
                continue  # Skip CTA lines, do NOT terminate the answer

            # Stop accumulating answer lines if we hit developer instructions or section overviews
            upper_s = stripped.upper()
            if (
                "AI CHATBOT TRAINING INSTRUCTIONS" in upper_s
                or "TOPIC OVERVIEW" in upper_s
                or upper_s.startswith("THE CHATBOT SHOULD")
                or upper_s.startswith("THE AI SHOULD")
                or upper_s.startswith("CAREER GROWTH TOPIC OVERVIEW")
                or upper_s.startswith("DEALERSHIP LIABILITY & EMPLOYMENT DISCLAIMER")
            ):
                qa_list.append({
                    "question": current_q,
                    "answer": "\n".join(current_a_lines).strip()
                })
                current_q = None
                current_a_lines = []
                continue

            current_a_lines.append(stripped)

    if current_q and current_a_lines:
        qa_list.append({
            "question": current_q,
            "answer": "\n".join(current_a_lines).strip()
        })

    return qa_list


def _clean_str(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


_QA_CACHE: list[dict[str, str]] | None = None


def get_qa_pairs() -> list[dict[str, str]]:
    global _QA_CACHE
    if _QA_CACHE is None:
        text = get_docx_knowledge_string()
        _QA_CACHE = extract_qa_pairs(text)
    return _QA_CACHE


PRICE_KEYWORDS = {"price", "cost", "fee", "tuition", "pricing", "how much", "pay", "charge"}


def find_exact_qa_match(query: str, threshold: float = 0.65) -> dict[str, str] | None:
    """
    Checks if query matches any Q: in the developer manual.
    If matched, returns dict containing {'question': ..., 'answer': ...} for verbatim output.
    Uses keyword-overlap scoring to prevent wrong matches when questions share generic phrases.
    """
    qa_pairs = get_qa_pairs()
    q_norm = _clean_str(query)
    if not q_norm or len(q_norm) < 3:
        return None

    query_lower = query.lower()
    q_has_price = any(k in query_lower for k in PRICE_KEYWORDS)

    # Extract important keywords from the query (non-stop words)
    query_keywords = {
        stem_word(w) for w in re.findall(r'\b[a-z0-9]+\b', query_lower)
        if w not in STOP_WORDS and len(w) > 2
    }

    best_match = None
    best_score = 0.0

    for qa in qa_pairs:
        target_q = qa["question"].lower()
        target_norm = _clean_str(target_q)
        if not target_norm:
            continue

        target_has_price = any(k in target_q for k in PRICE_KEYWORDS)
        if q_has_price and not target_has_price:
            # If user query is asking about price/cost, do not match a non-pricing QA pair
            continue

        # Exact normalized match
        if q_norm == target_norm:
            return qa

        # Direct containment check: if the question is a clear substring of user's query
        if (q_norm in target_norm or target_norm in q_norm):
            shorter_len = len(min(q_norm, target_norm, key=len))
            ratio = shorter_len / float(len(max(q_norm, target_norm, key=len)))
            if shorter_len > 10 and ratio > 0.5:
                return qa

        # SequenceMatcher ratio
        seq_score = SequenceMatcher(None, q_norm, target_norm).ratio()

        # rapidfuzz token sort ratio if available (more reliable than token_set_ratio)
        _fuzz = fuzz
        if _fuzz is not None:
            fuzz_score = float(_fuzz.ratio(query_lower, target_q)) / 100.0
            seq_score = max(seq_score, fuzz_score)

        # Keyword overlap bonus
        if query_keywords:
            target_keywords = {
                stem_word(w) for w in re.findall(r'\b[a-z0-9]+\b', target_q)
                if w not in STOP_WORDS and len(w) > 2
            }
            overlap = len(query_keywords & target_keywords)
            keyword_ratio = overlap / len(query_keywords) if query_keywords else 0.0
        else:
            keyword_ratio = 0.0

        # Combined score: sequence similarity (60%) + keyword overlap (40%)
        combined_score = (seq_score * 0.6) + (keyword_ratio * 0.4)

        if combined_score > best_score:
            best_score = combined_score
            best_match = qa

    if best_score >= threshold:
        return best_match

    return None


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
        expanded_terms = expand_query_terms(query)
        query_words = [w for w in self._tokenize(query) if w not in STOP_WORDS]
        if not query_words:
            query_words = self._tokenize(query)
        scores = []
        for idx, doc in enumerate(self.processed_chunks):
            tfidf_score = 0.0
            for qw in query_words:
                if qw in doc:
                    tf = doc.count(qw) / len(doc) if len(doc) > 0 else 0.0
                    idf = self.idf.get(qw, 0.0)
                    tfidf_score += tf * idf
            fuzzy_score = fuzzy_match_score(self.chunks[idx], expanded_terms)
            combined = tfidf_score * 1.5 + fuzzy_score * 0.5
            scores.append((idx, combined))
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
