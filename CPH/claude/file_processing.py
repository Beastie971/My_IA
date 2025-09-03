#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestion des fichiers et nettoyage du texte pour OCR Juridique v7
"""

import os
import re
import unicodedata
from typing import Tuple

# =============================================================================
# GESTION DES FICHIERS
# =============================================================================

def read_text_file(file_path: str) -> Tuple[str, str]:
    """Lit un fichier texte avec détection automatique de l'encodage."""
    if not file_path or not os.path.exists(file_path):
        return "", "Fichier non trouvé"
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return content.strip(), f"Fichier lu avec encodage {encoding}"
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return "", f"Erreur lecture : {str(e)}"
    return "", "Impossible de décoder le fichier"

def get_file_type(file_path: str) -> str:
    """Détermine le type de fichier à partir de l'extension."""
    if not file_path:
        return "UNKNOWN"
    ext = file_path.lower()
    if ext.endswith('.pdf'):
        return "PDF"
    elif ext.endswith(('.txt', '.text')):
        return "TXT"
    else:
        return "UNKNOWN"

# =============================================================================
# NETTOYAGE ET TRAITEMENT DU TEXTE
# =============================================================================

def _normalize_unicode(text: str) -> str:
    """Normalise les caractères Unicode."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "'": "'", "'": "'", """: '"', """: '"',
        "–": "-", "—": "-", "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def smart_clean(text: str, pages_texts=None) -> str:
    """Nettoie intelligemment le texte OCR."""
    if not text:
        return text

    text = _normalize_unicode(text)

    if pages_texts:
        pages_lines = [t.splitlines() for t in pages_texts]
        text = "\n".join("\n".join(lines) for lines in pages_lines)

    # Réparer les césures
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Patterns de numéros de page
    page_patterns = [
        r"(?im)^\s*page\s*[:\-]?\s*\d+\s*(?:/|sur|de|of)\s*\d+\s*$",
        r"(?im)^\s*p(?:age)?\.?\s*[:\-]?\s*\d+\s*$",
        r"(?im)^\s*\d+\s*/\s*\d+\s*$",
        r"(?im)^\s*[-–—]+\s*\d+\s*[-–—]+\s*$",
        r"(?im)^\s*\d{1,3}\s*$",
        r"(?im)^\s*page\s+n[°º]\s*\d+\s*$",
        r"(?im)^\s*page\s*[-–—]+\s*\d+\s*[-–—]+\s*$",
    ]
    
    for pattern in page_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # Nettoyage général
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[–—\-]{2,}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-–—\s]*\s*$", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"(?<![.!?:;])\n(?!\n)(?=[a-zàâäçéèêëîïôöùûüÿñ\(\,\;\:\.\-])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    text = text.strip()
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r'^[\d\s\-–—\.]{1,10}$', stripped):
            cleaned_lines.append(line)
        elif not stripped:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()