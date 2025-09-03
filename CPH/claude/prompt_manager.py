#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestion des prompts pour OCR Juridique v7
"""

import os
import json
from typing import Dict, Tuple
from config import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT, PROMPT_STORE_PATH

# =============================================================================
# GESTION DES PROMPTS
# =============================================================================

def load_prompt_store() -> Dict[str, str]:
    """Charge le magasin de prompts depuis le fichier JSON."""
    default_store = {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT}
    
    if not os.path.exists(PROMPT_STORE_PATH):
        try:
            os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
            with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(default_store, f, ensure_ascii=False, indent=2)
            return default_store
        except Exception as e:
            print(f"Attention: Impossible de créer le store initial: {e}")
            return default_store
    
    try:
        with open(PROMPT_STORE_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
        
        if not isinstance(store, dict) or not store:
            return default_store
            
        if DEFAULT_PROMPT_NAME not in store:
            store[DEFAULT_PROMPT_NAME] = DEFAULT_PROMPT_TEXT
            
        return store
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Erreur chargement store: {e}")
        return default_store

def save_prompt_store(store: Dict[str, str]) -> Tuple[bool, str]:
    """Sauvegarde le magasin de prompts."""
    try:
        if not isinstance(store, dict):
            return False, "Erreur: store n'est pas un dictionnaire"
        
        os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
        json_data = json.dumps(store, ensure_ascii=False, indent=2)
        
        with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
            f.write(json_data)
        
        return True, f"Enregistré dans : `{PROMPT_STORE_PATH}`"
        
    except Exception as e:
        return False, f"Échec d'enregistrement : {e}"