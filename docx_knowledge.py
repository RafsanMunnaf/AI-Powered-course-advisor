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


@lru_cache(maxsize=1)
def _extract_docx_text(docx_path: str) -> str:
    """Read every paragraph from the .docx and return as a single string."""
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
