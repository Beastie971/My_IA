#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Système de cache OCR pour OCR Juridique v7
"""

import os
import hashlib
import pickle
from typing import Optional
from config import _script_dir

# =============================================================================
# SYSTÈME DE CACHE OCR
# =============================================================================

def get_pdf_hash(pdf_path: str) -> Optional[str]:
    """Calcule le hash MD5 d'un fichier PDF."""
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    try:
        with open(pdf_path, 'rb') as f:
            hash_obj = hashlib.md5()
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except:
        return None

def get_cache_path(pdf_hash: str, nettoyer: bool) -> str:
    """Retourne le chemin du fichier cache pour un PDF."""
    cache_dir = os.path.join(_script_dir(), "cache_ocr")
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_clean" if nettoyer else "_raw"
    return os.path.join(cache_dir, f"ocr_{pdf_hash}{suffix}.pkl")

def save_ocr_cache(pdf_hash: str, nettoyer: bool, ocr_data: dict) -> bool:
    """Sauvegarde les données OCR dans le cache."""
    try:
        cache_path = get_cache_path(pdf_hash, nettoyer)
        with open(cache_path, 'wb') as f:
            pickle.dump(ocr_data, f)
        print(f"OCR mis en cache : {cache_path}")
        return True
    except Exception as e:
        print(f"Erreur sauvegarde cache : {e}")
        return False

def load_ocr_cache(pdf_hash: str, nettoyer: bool) -> Optional[dict]:
    """Charge les données OCR depuis le cache."""
    try:
        cache_path = get_cache_path(pdf_hash, nettoyer)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"OCR chargé depuis le cache : {cache_path}")
            return data
    except Exception as e:
        print(f"Erreur chargement cache : {e}")
    return None

def clear_ocr_cache() -> int:
    """Vide le cache OCR et retourne le nombre de fichiers supprimés."""
    try:
        cache_dir = os.path.join(_script_dir(), "cache_ocr")
        if not os.path.exists(cache_dir):
            return 0
        
        count = 0
        for filename in os.listdir(cache_dir):
            if filename.startswith("ocr_") and filename.endswith(".pkl"):
                os.remove(os.path.join(cache_dir, filename))
                count += 1
        return count
    except Exception as e:
        print(f"Erreur nettoyage cache : {e}")
        return 0
