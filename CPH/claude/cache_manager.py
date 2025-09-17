#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestionnaire de cache pour OCR
Version: 1.1 - CORRIGÉ avec get_pdf_hash
Date: 2025-09-17
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any

CACHE_DIR = "cache"

def get_pdf_hash(file_path: str) -> Optional[str]:
    """Calcule le hash MD5 d'un fichier PDF."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    except Exception as e:
        print(f"Erreur calcul hash: {e}")
        return None

def load_ocr_cache(pdf_hash: str, cleaned: bool) -> Optional[Dict[str, Any]]:
    """Charge les données OCR depuis le cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}_{'clean' if cleaned else 'raw'}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Erreur chargement cache: {e}")
    
    return None

def save_ocr_cache(pdf_hash: str, cleaned: bool, data: Dict[str, Any]) -> bool:
    """Sauvegarde les données OCR dans le cache."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}_{'clean' if cleaned else 'raw'}.json")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Erreur sauvegarde cache: {e}")
        return False

def clear_ocr_cache():
    """Nettoie le cache OCR."""
    try:
        if os.path.exists(CACHE_DIR):
            for file in os.listdir(CACHE_DIR):
                if file.endswith('.json'):
                    os.remove(os.path.join(CACHE_DIR, file))
            print("Cache OCR nettoyé")
        return True
    except Exception as e:
        print(f"Erreur nettoyage cache: {e}")
        return False
