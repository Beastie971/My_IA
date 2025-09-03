#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR structuré + Analyse juridique (Ollama/RunPod) - VERSION COMPLÈTE v7
Support PDF (OCR avec cache) et TXT avec anonymisation
Author: Assistant Claude
Version: 7.0
"""

import os
import sys
import argparse
import traceback

# Import des modules
from config import check_dependencies, PROMPT_STORE_DIR
from ai_providers import get_ollama_models
from gradio_interface import build_ui

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """Point d'entrée principal du programme."""
    parser = argparse.ArgumentParser(description="OCR + Analyse juridique avec Ollama/RunPod (PDF/TXT + Anonymisation)")
    parser.add_argument('--list-models', action='store_true', help='Lister les modèles Ollama disponibles')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='URL du serveur Ollama')
    parser.add_argument('--host', default='127.0.0.1', help='Adresse d\'écoute (défaut: 127.0.0.1)')
    parser.add_argument('--port', type=int, help='Port d\'écoute (défaut: auto)')
    
    args = parser.parse_args()
    
    # Vérification des dépendances
    check_dependencies()
    
    if args.list_models:
        print("Modèles Ollama disponibles:")
        try:
            models = get_ollama_models(args.ollama_url)
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        except Exception as e:
            print(f"Erreur: {e}")
        return
    
    print("🚀 Démarrage de l'interface OCR Juridique (Ollama/RunPod + Anonymisation)...")
    print(f"📁 Répertoire des prompts : {PROMPT_STORE_DIR}")
    
    if not os.path.exists(PROMPT_STORE_DIR):
        os.makedirs(PROMPT_STORE_DIR, exist_ok=True)
    
    try:
        models = get_ollama_models(args.ollama_url)
        print(f"🤖 Modèles disponibles : {len(models)}")
        
        app = build_ui()
        
        launch_kwargs = {
            'server_name': args.host,
            'share': False,
            'inbrowser': True
        }
        
        if args.port:
            launch_kwargs['server_port'] = args.port
            
        app.launch(**launch_kwargs)
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
