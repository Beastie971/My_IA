#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gestion des prompts pour OCR Juridique v7
"""

import os
import json
import glob
from typing import Dict, Tuple, List
from config import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT, PROMPT_STORE_PATH, PROMPT_STORE_DIR

# =============================================================================
# GESTION DES PROMPTS AVEC FICHIERS TXT
# =============================================================================

def scan_txt_prompts() -> Dict[str, str]:
    """Scanne le répertoire prompts/ pour les fichiers .txt et les charge."""
    txt_prompts = {}
    
    try:
        # Chercher tous les fichiers .txt dans le répertoire prompts/
        txt_files = glob.glob(os.path.join(PROMPT_STORE_DIR, "*.txt"))
        
        for txt_file in txt_files:
            try:
                # Utiliser le nom du fichier (sans extension) comme clé
                filename = os.path.basename(txt_file)
                prompt_name = os.path.splitext(filename)[0]
                
                # Lire le contenu du fichier en préservant exactement le formatage
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()  # Ne pas utiliser .strip() pour préserver le formatage exact
                
                if content:  # Seulement si le fichier n'est pas vide
                    txt_prompts[f"📄 {prompt_name}"] = content
                    print(f"✅ Prompt chargé depuis {filename}")
                
            except Exception as e:
                print(f"⚠️ Erreur lecture {txt_file}: {e}")
                continue
    
    except Exception as e:
        print(f"⚠️ Erreur scan répertoire prompts: {e}")
    
    return txt_prompts

def load_prompt_store() -> Dict[str, str]:
    """Charge le magasin de prompts depuis le fichier JSON + fichiers TXT."""
    default_store = {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT}
    
    # Charger d'abord les prompts JSON (anciens)
    json_prompts = {}
    if os.path.exists(PROMPT_STORE_PATH):
        try:
            with open(PROMPT_STORE_PATH, "r", encoding="utf-8") as f:
                json_prompts = json.load(f)
            
            if not isinstance(json_prompts, dict):
                json_prompts = {}
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Erreur chargement store JSON: {e}")
            json_prompts = {}
    
    # Charger les prompts depuis les fichiers .txt
    txt_prompts = scan_txt_prompts()
    
    # Combiner tous les prompts
    all_prompts = {}
    all_prompts.update(default_store)  # Prompt par défaut en premier
    all_prompts.update(json_prompts)   # Puis prompts JSON
    all_prompts.update(txt_prompts)    # Enfin prompts TXT (priorité)
    
    # S'assurer que le prompt par défaut existe
    if DEFAULT_PROMPT_NAME not in all_prompts:
        all_prompts[DEFAULT_PROMPT_NAME] = DEFAULT_PROMPT_TEXT
    
    print(f"📋 Prompts chargés: {len(all_prompts)} au total")
    return all_prompts

def save_prompt_store(store: Dict[str, str]) -> Tuple[bool, str]:
    """Sauvegarde uniquement les prompts JSON (pas les fichiers TXT)."""
    try:
        if not isinstance(store, dict):
            return False, "Erreur: store n'est pas un dictionnaire"
        
        # Filtrer les prompts TXT (ceux avec 📄 prefix)
        json_prompts = {k: v for k, v in store.items() if not k.startswith("📄 ")}
        
        os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
        json_data = json.dumps(json_prompts, ensure_ascii=False, indent=2)
        
        with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
            f.write(json_data)
        
        return True, f"Enregistré dans : `{PROMPT_STORE_PATH}` (prompts TXT préservés)"
        
    except Exception as e:
        return False, f"Échec d'enregistrement : {e}"

def list_available_prompts() -> List[str]:
    """Liste tous les prompts disponibles avec leur source."""
    prompts = load_prompt_store()
    
    print("\n=== PROMPTS DISPONIBLES ===")
    for name in sorted(prompts.keys()):
        source = "TXT" if name.startswith("📄 ") else "JSON" if name != DEFAULT_PROMPT_NAME else "Défaut"
        preview = prompts[name][:80] + "..." if len(prompts[name]) > 80 else prompts[name]
        print(f"[{source}] {name}")
        print(f"    {preview}")
        print()
    
    return list(prompts.keys())

def create_sample_txt_prompts():
    """Crée des exemples de prompts TXT si le répertoire est vide."""
    try:
        if not os.path.exists(PROMPT_STORE_DIR):
            os.makedirs(PROMPT_STORE_DIR, exist_ok=True)
        
        # Vérifier s'il y a déjà des fichiers .txt
        existing_txt = glob.glob(os.path.join(PROMPT_STORE_DIR, "*.txt"))
        if existing_txt:
            return  # Déjà des fichiers, ne pas écraser
        
        # Créer des exemples
        samples = {
            "analyse_courte.txt": """Tu es un juriste. Analyse ce document juridique de manière concise en identifiant :
1. Les faits principaux
2. Les arguments juridiques
3. Les références légales mentionnées
4. Une conclusion synthétique

Réponds en français juridique, de façon structurée mais brève.""",
            
            "expert_detaille.txt": """Tu es un juriste senior spécialisé en droit du travail. 

Effectue une analyse juridique approfondie du document en respectant ces consignes :

MÉTHODE :
- Analyse exclusivement le contenu du document fourni
- Ne jamais inventer ou supposer des faits non mentionnés
- Signaler explicitement si une information est "non précisée dans le document"

STRUCTURE ATTENDUE :
- Qualification juridique des faits et contexte procédural
- Moyens et arguments des parties (textuels uniquement)
- Références légales présentes dans le document
- Analyse critique des moyens soulevés
- Évaluation de la portée juridique

Rédige en français juridique, style paragraphes continus, sans listes à puces.""",
            
            "resume_executif.txt": """Tu es un conseiller juridique. Produis un résumé exécutif du document en 3 paragraphes maximum :

1. CONTEXTE : Situation et parties impliquées
2. ENJEUX : Points juridiques clés et arguments principaux  
3. IMPACT : Conséquences juridiques potentielles

Style : Accessible, concis, orienté décision."""
        }
        
        for filename, content in samples.items():
            filepath = os.path.join(PROMPT_STORE_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Exemple créé : {filename}")
        
        print(f"📁 Exemples de prompts créés dans {PROMPT_STORE_DIR}")
        
    except Exception as e:
        print(f"⚠️ Erreur création exemples : {e}")
