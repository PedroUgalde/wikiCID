from __future__ import annotations

import math
import re
import unicodedata
from urllib.parse import urlparse

from nltk.stem.snowball import SnowballStemmer
from simplemma import lemmatize

_stemmer_es = SnowballStemmer("spanish")
_token_re = re.compile(r"\w+", re.UNICODE)
_url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "-")


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def clean_url(raw: str | float | None) -> str:
    if _is_missing(raw):
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return ""
    s = s.split()[0]
    if not s.startswith(("http://", "https://")):
        s = "https://" + s.lstrip("/")
    try:
        parsed = urlparse(s)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def lemmatize_and_stem_spanish(text: str) -> str:
    """Huella lexical para deduplicación (español)."""
    if not text:
        return ""
    t = strip_accents(text.lower())
    t = _url_re.sub(" ", t)
    t = normalize_whitespace(t)
    out: list[str] = []
    for w in _token_re.findall(t):
        if len(w) < 2:
            continue
        lemma = lemmatize(w, lang="es") or w
        out.append(_stemmer_es.stem(lemma))
    return " ".join(out)


def text_for_embedding(parts: list[str]) -> str:
    """Texto legible para el modelo de embeddings (sin stemming agresivo)."""
    chunks = []
    for p in parts:
        if _is_missing(p):
            continue
        s = str(p).strip()
        if s and s.lower() != "nan":
            chunks.append(s)
    raw = " \n ".join(chunks)
    raw = _url_re.sub(" ", raw)
    t = normalize_whitespace(raw.lower())
    return t


def fingerprint_row(empresa: str, web_host: str, embedding_text: str) -> str:
    e = normalize_whitespace(strip_accents(str(empresa or "").lower()))
    h = (web_host or "").lower()
    tail = lemmatize_and_stem_spanish(embedding_text)[:500]
    return f"{e}|{h}|{tail}"
