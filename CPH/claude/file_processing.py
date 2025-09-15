#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de traitement de fichiers pour OCR Juridique
Version: 1.0
Date: 2025-09-15
"""

import os
import re
import unicodedata
from typing import Tuple

def get_file_type(file_path: str) -> str:
    """Détermine le type de fichier."""
    if not file_path:
        return "UNKNOWN"
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return "PDF"
    elif ext in ['.txt', '.text']:
        return "TXT"
    elif ext in ['.doc', '.docx']:
        return "DOC"
    else:
        return "UNKNOWN"

def read_text_file(file_path: str) -> Tuple[str, str]:
    """Lit un fichier texte avec gestion des encodages."""
    if not os.path.exists(file_path):
        return "", "Fichier introuvable"
    
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                return content, f"Lu avec encodage {encoding}"
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return "", f"Erreur lecture: {str(e)}"
    
    return "", "Impossible de décoder le fichier"

def smart_clean(text: str, pages_texts=None) -> str:
    """Nettoyage intelligent du texte OCR."""
    if not text:
        return ""
    
    # Normalisation Unicode
    text = _normalize_unicode(text)
    
    # Suppression des caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Correction des espaces multiples
    text = re.sub(r' +', ' ', text)
    
    # Correction des sauts de ligne multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Suppression des espaces en début/fin de ligne
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    return text.strip()

def _normalize_unicode(text: str) -> str:
    """Normalise les caractères Unicode."""
    if not text:
        return ""
    
    # Normalisation NFD puis NFC
    text = unicodedata.normalize('NFD', text)
    text = unicodedata.normalize('NFC', text)
    
    return text
