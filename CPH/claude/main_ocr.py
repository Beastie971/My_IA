#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR structuré + Analyse juridique (Ollama/RunPod) - VERSION COMPLÈTE v7.1-FALKON-FIXED
Support PDF (OCR avec cache) et TXT avec anonymisation - Interface dual files avec Falkon
Author: Assistant Claude
Version: 7.1-FALKON-FIXED - Modifié pour utiliser le navigateur Falkon avec corrections lambda
Date: 2025-01-04
Modifications: Interface dual files, analyse comparative unique, lancement automatique Falkon
"""

import os
import sys
import argparse
import traceback
import subprocess
import time

# Import des modules
from config import check_dependencies, PROMPT_STORE_DIR
from ai_providers import get_ollama_models
from gradio_interface import build_ui

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def launch_falkon(url, delay=3):
    """Lance Falkon avec l'URL de l'application."""
    try:
        print(f"🦅 Lancement de Falkon dans {delay}s...")
        time.sleep(delay)  # Laisser le temps au serveur de démarrer
        
        # Vérifier si Falkon est disponible
        result = subprocess.run(['which', 'falkon'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Falkon n'est pas installé ou disponible dans le PATH")
            print("💡 Installez Falkon avec : sudo apt install falkon (Ubuntu/Debian)")
            print(f"📱 Vous pouvez ouvrir manuellement : {url}")
            return False
        
        # Lancer Falkon
        subprocess.run(['falkon', url], check=False)
        print(f"✅ Falkon lancé avec : {url}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du lancement de Falkon : {e}")
        print(f"📱 Ouvrez manuellement : {url}")
        return False

def main():
    """Point d'entrée principal du programme."""
    parser = argparse.ArgumentParser(description="OCR + Analyse juridique avec Ollama/RunPod (PDF/TXT + Anonymisation) - Interface dual files avec Falkon")
    parser.add_argument('--list-models', action='store_true', help='Lister les modèles Ollama disponibles')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='URL du serveur Ollama')
    parser.add_argument('--host', default='127.0.0.1', help='Adresse d\'écoute (défaut: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7860, help='Port d\'écoute (défaut: 7860)')
    parser.add_argument('--no-browser', action='store_true', help='Ne pas lancer Falkon automatiquement')
    parser.add_argument('--debug', action='store_true', help='Mode debug avec plus d\'informations')
    
    args = parser.parse_args()
    
    # Mode debug
    if args.debug:
        print("🔍 Mode debug activé")
        print(f"🐍 Python: {sys.version}")
        print(f"📁 Répertoire courant: {os.getcwd()}")
        print(f"🔧 Arguments: {args}")
    
    # Vérification des dépendances
    print("🔍 Vérification des dépendances...")
    try:
        check_dependencies()
        print("✅ Dépendances OK")
    except Exception as e:
        print(f"❌ Erreur dépendances: {e}")
        return 1
    
    if args.list_models:
        print("🤖 Modèles Ollama disponibles:")
        try:
            models = get_ollama_models(args.ollama_url)
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        except Exception as e:
            print(f"❌ Erreur récupération modèles: {e}")
        return 0
    
    print("🚀 Démarrage de l'interface OCR Juridique...")
    print(f"📁 Répertoire des prompts : {PROMPT_STORE_DIR}")
    print(f"🦅 Navigateur configuré : Falkon")
    
    # Créer le répertoire des prompts si nécessaire
    if not os.path.exists(PROMPT_STORE_DIR):
        try:
            os.makedirs(PROMPT_STORE_DIR, exist_ok=True)
            print(f"📁 Répertoire créé : {PROMPT_STORE_DIR}")
        except Exception as e:
            print(f"⚠️ Impossible de créer le répertoire des prompts : {e}")
    
    try:
        print("🤖 Chargement des modèles Ollama...")
        models = get_ollama_models(args.ollama_url)
        print(f"✅ {len(models)} modèle(s) disponible(s)")
        if args.debug:
            for model in models[:3]:  # Afficher les 3 premiers
                print(f"  - {model}")
        
        print("🏗️ Construction de l'interface...")
        app = build_ui()
        print("✅ Interface construite")
        
        # Configuration du lancement
        launch_kwargs = {
            'server_name': args.host,
            'share': False,
            'inbrowser': False,  # Désactiver l'ouverture automatique du navigateur par défaut
            'show_error': True,
            'quiet': not args.debug,
            'server_port': args.port
        }
        
        # Construire l'URL
        url = f"http://{args.host}:{args.port}"
        
        print(f"📡 Serveur en cours de démarrage sur : {url}")
        
        # Lancer Falkon en arrière-plan si demandé
        if not args.no_browser:
            import threading
            falkon_thread = threading.Thread(target=launch_falkon, args=(url, 4))
            falkon_thread.daemon = True
            falkon_thread.start()
        else:
            print(f"📱 Pour ouvrir dans Falkon : falkon {url}")
            print(f"📱 Ou dans votre navigateur : {url}")
        
        print("🎯 Lancement de l'application Gradio...")
        
        # Lancer l'application Gradio
        app.launch(**launch_kwargs)
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
        return 0
    except Exception as e:
        print(f"❌ Erreur lors du démarrage : {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
