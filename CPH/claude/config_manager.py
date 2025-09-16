#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de gestion de la configuration
Version: 1.0
Date: 2025-09-11
Fonctionnalités: Configuration Ollama, modèles, sauvegarde/chargement
"""

import os
import json

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ollama_config.json")

def load_ollama_config():
    """Charge la configuration Ollama sauvegardée."""
    try:
        print(f"CONFIG LOAD: Tentative chargement depuis {CONFIG_FILE_PATH}")
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
                url = config.get('last_ollama_url', 'http://localhost:11434')
                print(f"CONFIG LOAD: URL chargée: {url}")
                return url
        else:
            print(f"CONFIG LOAD: Fichier {CONFIG_FILE_PATH} non trouvé, utilisation par défaut")
            return 'http://localhost:11434'
    except Exception as e:
        print(f"CONFIG LOAD ERROR: {e}")
        return 'http://localhost:11434'

def save_ollama_config(ollama_url):
    """Sauvegarde la configuration Ollama."""
    try:
        print(f"CONFIG SAVE: Sauvegarde de '{ollama_url}' dans {CONFIG_FILE_PATH}")
        
        config_dir = os.path.dirname(CONFIG_FILE_PATH)
        if not os.path.exists(config_dir):
            print(f"CONFIG SAVE: Création répertoire {config_dir}")
            os.makedirs(config_dir, exist_ok=True)
        
        config = {'last_ollama_url': ollama_url}
        
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                saved_url = saved_config.get('last_ollama_url')
                if saved_url == ollama_url:
                    print(f"CONFIG SAVE: Succès - '{ollama_url}' sauvegardée et vérifiée")
                    return True
                else:
                    print(f"CONFIG SAVE: Erreur - URL relue '{saved_url}' != URL demandée '{ollama_url}'")
                    return False
        else:
            print(f"CONFIG SAVE: Fichier {CONFIG_FILE_PATH} non créé")
            return False
            
    except PermissionError as e:
        print(f"CONFIG SAVE: Erreur permissions - {e}")
        print(f"CONSEIL: Vérifiez les permissions d'écriture dans {config_dir}")
        return False
    except Exception as e:
        print(f"CONFIG SAVE: Erreur générale - {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_models_list():
    """Initialise la liste des modèles au démarrage."""
    print("INIT_MODELS: Initialisation liste des modèles")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"INIT_MODELS: {len(models)} modèles trouvés: {models}")
            return models
        else:
            print(f"INIT_MODELS: Erreur API: {response.status_code}")
            return ["mistral:latest", "llama2:latest", "deepseek-coder:latest"]
            
    except Exception as e:
        print(f"INIT_MODELS: Exception: {e}")
        return ["mistral:latest", "llama2:latest", "deepseek-coder:latest"]

def save_url_on_change(url):
    """Sauvegarde automatiquement l'URL quand elle change."""
    if url and url.strip():
        success = save_ollama_config(url.strip())
        if success:
            return f"URL sauvegardée: {url}"
        else:
            return "Erreur sauvegarde URL"
    return "URL vide"