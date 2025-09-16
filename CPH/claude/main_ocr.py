#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR structuré + Analyse juridique (Ollama/RunPod) - VERSION ANTI-TIMEOUT 524
Support PDF (OCR avec cache) et TXT avec anonymisation - Interface async + Falkon
Author: Assistant Claude
Version: 8.3-CLOUDFLARE-SAFE-CTRLC - Protection complète timeout 524 + Ctrl+C fix
Date: 2025-09-15
Modifications: Mode asynchrone, timeout sécurisé, estimation intelligente, correction Ctrl+C
"""

import os
import sys
import argparse
import traceback
import subprocess
import time
from pathlib import Path

# Import des modules avec gestion d'erreurs
try:
    from config import check_dependencies, PROMPT_STORE_DIR
    from ai_providers import get_ollama_models, calculate_smart_timeout
    from gradio_interface import build_ui
    from async_task_manager import task_manager
    
    print("✅ Modules principaux chargés avec succès")
    MODULES_OK = True
except ImportError as e:
    print(f"❌ Erreur import modules: {e}")
    MODULES_OK = False

# =============================================================================
# CONFIGURATION CLOUDFLARE SAFE
# =============================================================================

CLOUDFLARE_TIMEOUT = 85  # Limite Cloudflare
MAX_STARTUP_TIME = 30   # Temps max démarrage
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def check_system_requirements():
    """Vérifie les prérequis système pour fonctionnement optimal."""
    print("🔍 Vérification des prérequis système...")
    
    issues = []
    
    # Vérifier Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ requis (actuellement: {sys.version})")
    
    # Vérifier espace disque (pour cache OCR)
    try:
        statvfs = os.statvfs('.')
        free_space = statvfs.f_frsize * statvfs.f_available / (1024**3)  # GB
        if free_space < 1:
            issues.append(f"Espace disque faible: {free_space:.1f}GB disponible")
    except:
        pass
    
    # Vérifier les dépendances critiques
    critical_deps = ['gradio', 'requests', 'pdf2image', 'pytesseract']
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            issues.append(f"Dépendance manquante: {dep}")
    
    if issues:
        print("⚠️ Problèmes détectés:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ Prérequis système OK")
    return True

def cleanup_tasks():
    """Nettoie les tâches en cours."""
    if MODULES_OK:
        try:
            active_tasks = task_manager.list_active_tasks()
            if active_tasks:
                print(f"🛑 Arrêt de {len(active_tasks)} tâche(s) active(s)...")
                for task_id in active_tasks:
                    task_manager.cancel_task(task_id)
                    
            # Nettoyer le répertoire temporaire des tâches
            import shutil
            temp_dir = "async_tasks"
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"🗑️ Répertoire temporaire nettoyé: {temp_dir}")
                except Exception as e:
                    print(f"⚠️ Erreur nettoyage: {e}")
                    
        except Exception as e:
            print(f"⚠️ Erreur lors du nettoyage: {e}")

def launch_falkon_safe(url, delay=3):
    """Lance Falkon avec gestion d'erreurs améliorée."""
    try:
        print(f"🦅 Lancement de Falkon dans {delay}s...")
        time.sleep(delay)
        
        # Vérifier si Falkon est disponible
        result = subprocess.run(['which', 'falkon'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ Falkon n'est pas installé ou disponible dans le PATH")
            print("💡 Alternatives:")
            print("   - Ubuntu/Debian: sudo apt install falkon")
            print("   - Arch: sudo pacman -S falkon")
            print("   - Ou utilisez votre navigateur par défaut")
            print(f"📱 URL à ouvrir manuellement : {url}")
            return False
        
        # Lancer Falkon avec timeout
        proc = subprocess.Popen(['falkon', url], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
        
        # Vérifier que le processus s'est bien lancé
        time.sleep(1)
        if proc.poll() is None:
            print(f"✅ Falkon lancé avec succès : {url}")
            return True
        else:
            print("❌ Falkon n'a pas pu démarrer correctement")
            return False
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout lors de la vérification de Falkon")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du lancement de Falkon : {e}")
        print(f"📱 Ouvrez manuellement : {url}")
        return False

def estimate_startup_time():
    """Estime le temps de démarrage selon la configuration."""
    base_time = 5  # Temps de base
    
    if MODULES_OK:
        try:
            # Tester rapidement la connexion Ollama
            models = get_ollama_models()
            if models and len(models) > 3:
                base_time += 2  # Plus de modèles = startup un peu plus long
        except:
            base_time += 3  # Problème Ollama = plus long
    
    return min(base_time, MAX_STARTUP_TIME)

def create_startup_info(host, port):
    """Crée les informations de démarrage."""
    url = f"http://{host}:{port}"
    
    info = f"""
🚀 OCR Juridique - Anti-Timeout 524 v8.3

CONFIGURATION:
📡 Serveur: {url}
⏱️ Timeout Cloudflare: {CLOUDFLARE_TIMEOUT}s (sécurisé)
🛡️ Mode asynchrone: {'Activé' if MODULES_OK else 'Indisponible'}
🔧 Modules: {'Complets' if MODULES_OK else 'Mode dégradé'}

FONCTIONNALITÉS:
🚀 Analyse intelligente: Direct ↔ Asynchrone automatique
📊 Estimation temps réel: Calcul avant traitement  
⚡ Chunks optimisés: Taille adaptative selon document
🛡️ Protection 524: Timeout sécurisé + polling
    """
    
    if not MODULES_OK:
        info += """
⚠️  MODE DÉGRADÉ ACTIF
Certaines fonctionnalités avancées ne sont pas disponibles.
Vérifiez l'installation des dépendances.
"""
    
    return info

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """Point d'entrée principal du programme avec protection timeout."""
    
    parser = argparse.ArgumentParser(
        description="OCR + Analyse juridique avec protection Cloudflare timeout 524"
    )
    parser.add_argument('--list-models', action='store_true', 
                       help='Lister les modèles Ollama disponibles')
    parser.add_argument('--ollama-url', default='http://localhost:11434', 
                       help='URL du serveur Ollama')
    parser.add_argument('--host', default=DEFAULT_HOST, 
                       help=f'Adresse d\'écoute (défaut: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, 
                       help=f'Port d\'écoute (défaut: {DEFAULT_PORT})')
    parser.add_argument('--no-browser', action='store_true', 
                       help='Ne pas lancer Falkon automatiquement')
    parser.add_argument('--debug', action='store_true', 
                       help='Mode debug avec plus d\'informations')
    parser.add_argument('--check-system', action='store_true',
                       help='Vérifier les prérequis système et quitter')
    parser.add_argument('--timeout', type=int, default=CLOUDFLARE_TIMEOUT,
                       help=f'Timeout max en secondes (défaut: {CLOUDFLARE_TIMEOUT})')
    
    args = parser.parse_args()
    
    # Affichage des informations de démarrage
    print(create_startup_info(args.host, args.port))
    
    # Mode debug
    if args.debug:
        print("🔍 Mode debug activé")
        print(f"🐍 Python: {sys.version}")
        print(f"📁 Répertoire courant: {os.getcwd()}")
        print(f"🔧 Arguments: {args}")
        print(f"🛡️ Timeout configuré: {args.timeout}s")
    
    # Vérification système seule
    if args.check_system:
        success = check_system_requirements()
        return 0 if success else 1
    
    # Vérification des prérequis
    if not check_system_requirements():
        print("❌ Prérequis non satisfaits, arrêt")
        return 1
    
    # Vérification des dépendances
    if MODULES_OK:
        print("🔍 Vérification des dépendances...")
        try:
            check_dependencies()
            print("✅ Dépendances OK")
        except Exception as e:
            print(f"❌ Erreur dépendances: {e}")
            return 1
    else:
        print("⚠️ Mode dégradé - certains modules non disponibles")
    
    # Liste des modèles seulement
    if args.list_models:
        print("🤖 Modèles Ollama disponibles:")
        if MODULES_OK:
            try:
                models = get_ollama_models(args.ollama_url)
                for i, model in enumerate(models, 1):
                    # Estimation rapide du temps par modèle
                    est_time = calculate_smart_timeout(5000, model)  # Test 5000 chars
                    safe_icon = "✅" if est_time < CLOUDFLARE_TIMEOUT else "⚠️"
                    print(f"  {i:2d}. {model:30} | Estimation: {est_time}s {safe_icon}")
                print(f"\n✅ = Compatible Cloudflare (< {CLOUDFLARE_TIMEOUT}s)")
                print(f"⚠️ = Peut nécessiter mode asynchrone")
            except Exception as e:
                print(f"❌ Erreur récupération modèles: {e}")
        else:
            print("❌ Modules non disponibles pour lister les modèles")
        return 0
    
    # Démarrage principal
    print("🚀 Démarrage de l'interface OCR Juridique Anti-Timeout...")
    print(f"📁 Répertoire des prompts : {PROMPT_STORE_DIR}")
    print(f"🦅 Navigateur configuré : Falkon")
    print(f"⏱️ Protection timeout : {args.timeout}s")
    
    # Créer le répertoire des prompts si nécessaire
    if not os.path.exists(PROMPT_STORE_DIR):
        try:
            os.makedirs(PROMPT_STORE_DIR, exist_ok=True)
            print(f"📁 Répertoire créé : {PROMPT_STORE_DIR}")
        except Exception as e:
            print(f"⚠️ Impossible de créer le répertoire des prompts : {e}")
    
    try:
        # Test rapide des modèles Ollama
        if MODULES_OK:
            print("🤖 Test rapide Ollama...")
            models = get_ollama_models(args.ollama_url)
            print(f"✅ {len(models)} modèle(s) disponible(s)")
            
            if args.debug and models:
                # Afficher les 3 premiers avec estimation
                for model in models[:3]:
                    est_time = calculate_smart_timeout(3000, model)
                    print(f"  - {model} (estimation: {est_time}s)")
        else:
            print("⚠️ Ollama non testé (modules manquants)")
        
        print("🏗️ Construction de l'interface...")
        
        # Construction de l'interface avec gestion d'erreur
        try:
            app = build_ui()
            print("✅ Interface construite avec succès")
        except Exception as e:
            print(f"❌ Erreur construction interface: {e}")
            if args.debug:
                traceback.print_exc()
            return 1
        
        # Configuration du lancement
        url = f"http://{args.host}:{args.port}"
        
        # Estimation du temps de démarrage
        estimated_startup = estimate_startup_time()
        print(f"⏱️ Estimation démarrage: {estimated_startup}s")
        
        # Configuration Gradio compatible
        launch_kwargs = {
            'server_name': args.host,
            'server_port': args.port,
            'share': False,
            'inbrowser': False,  # Désactiver l'ouverture automatique par défaut
            'show_error': True,
            'quiet': not args.debug,
            'favicon_path': None
        }
        
        print(f"📡 Serveur en cours de démarrage sur : {url}")
        print(f"🛡️ Protection Cloudflare : timeout max {args.timeout}s")
        
        # Lancer Falkon en arrière-plan si demandé
        if not args.no_browser:
            import threading
            falkon_delay = max(estimated_startup + 1, 4)  # Au moins 4s
            falkon_thread = threading.Thread(
                target=launch_falkon_safe, 
                args=(url, falkon_delay),
                daemon=True
            )
            falkon_thread.start()
        else:
            print(f"📱 Pour ouvrir dans Falkon : falkon {url}")
            print(f"📱 Ou dans votre navigateur : {url}")
        
        print("🎯 Lancement de l'application Gradio...")
        print(f"🔄 Mode polling asynchrone : {'Activé' if MODULES_OK else 'Désactivé'}")
        
        # Message de sécurité Cloudflare
        print(f"""
🛡️ PROTECTION CLOUDFLARE ACTIVE:
   - Timeout automatique: {args.timeout}s
   - Mode asynchrone: Basculement automatique
   - Estimation préalable: Calcul temps de traitement
   - Chunks optimisés: Taille adaptative
   
🚀 UTILISATION:
   - Documents < 10k chars: Traitement direct
   - Documents > 10k chars: Mode asynchrone automatique
   - Suivez les estimations affichées en temps réel
   
⚠️  CTRL+C pour arrêter l'application
        """)
        
        # Démarrage avec gestion d'interruption SIMPLIFIÉE
        try:
            # Lancer l'application Gradio (Ctrl+C natif)
            app.launch(**launch_kwargs)
            
        except KeyboardInterrupt:
            print("\n🛑 Interruption Ctrl+C détectée")
            cleanup_tasks()
            return 0
        except Exception as e:
            print(f"❌ Erreur pendant l'exécution : {e}")
            if args.debug:
                traceback.print_exc()
            cleanup_tasks()
            return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Arrêt demandé par l'utilisateur")
        cleanup_tasks()
        return 0
    except Exception as e:
        print(f"❌ Erreur lors du démarrage : {e}")
        if args.debug:
            traceback.print_exc()
        cleanup_tasks()
        return 1
    
    finally:
        # Nettoyage final simple
        print("🧹 Nettoyage final...")
        cleanup_tasks()
        print("👋 Application arrêtée proprement")
    
    return 0

# =============================================================================
# FONCTION DE TEST RAPIDE
# =============================================================================

def quick_test():
    """Test rapide des fonctionnalités principales."""
    print("🧪 Test rapide des fonctionnalités...")
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Import des modules
    try:
        if MODULES_OK:
            print("✅ Test 1: Import modules OK")
            tests_passed += 1
        else:
            print("❌ Test 1: Import modules ECHEC")
    except:
        print("❌ Test 1: Import modules ECHEC")
    
    # Test 2: Gestionnaire de tâches
    try:
        if MODULES_OK:
            from async_task_manager import task_manager
            task_count = len(task_manager.list_active_tasks())
            print(f"✅ Test 2: Gestionnaire tâches OK ({task_count} actives)")
            tests_passed += 1
        else:
            print("❌ Test 2: Gestionnaire tâches non disponible")
    except Exception as e:
        print(f"❌ Test 2: Gestionnaire tâches ECHEC - {e}")
    
    # Test 3: Timeout calculation
    try:
        if MODULES_OK:
            timeout = calculate_smart_timeout(5000, "mistral:7b")
            safe = timeout < CLOUDFLARE_TIMEOUT
            print(f"✅ Test 3: Calcul timeout OK ({timeout}s, safe: {safe})")
            tests_passed += 1
        else:
            print("❌ Test 3: Calcul timeout non disponible")
    except Exception as e:
        print(f"❌ Test 3: Calcul timeout ECHEC - {e}")
    
    # Test 4: Interface builder
    try:
        app = build_ui()
        if app:
            print("✅ Test 4: Construction interface OK")
            tests_passed += 1
        else:
            print("❌ Test 4: Construction interface ECHEC")
    except Exception as e:
        print(f"❌ Test 4: Construction interface ECHEC - {e}")
    
    # Test 5: Répertoires et permissions
    try:
        test_dir = "test_temp_dir"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        os.rmdir(test_dir)
        print("✅ Test 5: Permissions fichiers OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5: Permissions fichiers ECHEC - {e}")
    
    # Résultat final
    success_rate = (tests_passed / total_tests) * 100
    print(f"\n📊 Résultat tests: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
    
    if tests_passed == total_tests:
        print("🎉 Tous les tests passent, application prête !")
        return True
    elif tests_passed >= 3:
        print("⚠️ Tests partiellement réussis, fonctionnement dégradé possible")
        return True
    else:
        print("❌ Trop de tests échoués, vérifiez l'installation")
        return False

# =============================================================================
# COMMANDES UTILITAIRES
# =============================================================================

def show_help_cloudflare():
    """Affiche l'aide spécifique pour les problèmes Cloudflare."""
    help_text = """
🛡️ AIDE PROTECTION CLOUDFLARE TIMEOUT 524

PROBLÈME:
L'erreur 524 "A timeout occurred" se produit quand Cloudflare attend plus 
de 100 secondes une réponse du serveur (RunPod, serveur distant, etc.).

SOLUTIONS IMPLÉMENTÉES:

1. 🚀 MODE ASYNCHRONE AUTOMATIQUE
   - Basculement auto pour documents > 10k caractères
   - Polling intelligent toutes les 3 secondes
   - Tâches en arrière-plan avec suivi

2. ⏱️ TIMEOUT SÉCURISÉ
   - Limite globale: 85 secondes (marge Cloudflare)
   - Timeout par chunk: 30 secondes max
   - Estimation préalable du temps de traitement

3. 📊 CHUNKS OPTIMISÉS
   - Taille adaptative: 1500-2000 caractères
   - Découpage intelligent (préserve structure juridique)
   - Fusion des petits chunks pour efficacité

4. 🔍 ESTIMATION INTELLIGENTE
   - Calcul temps selon modèle + taille document
   - Recommandation mode direct/asynchrone
   - Alertes visuelles en temps réel

UTILISATION:

✅ Documents courts (< 5k chars):
   → Mode direct automatique

🚀 Documents longs (> 10k chars):
   → Mode asynchrone automatique
   → Cliquez "Vérifier statut" pour suivre
   → "Récupérer résultat" quand terminé

⚡ Forcer asynchrone:
   → Bouton "Forcer Asynchrone" pour tests

CONSEILS:

📝 Utilisez des prompts concis (style télégraphique)
🔧 Modèles recommandés: Mistral 7B (rapide) ou LLaMA 3.1 8B
📊 Surveillez l'estimation temps en temps réel
🛡️ En cas d'erreur 524: relancez en mode asynchrone
"""
    print(help_text)

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    # Commandes spéciales
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            success = quick_test()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--help-cloudflare":
            show_help_cloudflare()
            sys.exit(0)
        elif sys.argv[1] == "--version":
            print("OCR Juridique Anti-Timeout 524 v8.3")
            print("Protection Cloudflare intégrée")
            print("Mode asynchrone intelligent")
            sys.exit(0)
    
    # Lancement normal
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🚨 Arrêt forcé par Ctrl+C")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        traceback.print_exc()
        sys.exit(1)
