#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fichier de compatibilité pour résoudre les erreurs d'import
À utiliser temporairement pendant la migration
"""

import gradio as gr

# Variables manquantes communes
analysis_mode = "classic"  # Variable par défaut
mode_analysis = "classic"  # Alias
processing_mode = "standard"
enable_chunks = False
enable_global_synthesis = False
chunk_size = 3000
chunk_overlap = 200
synthesis_prompt = ""

# Dictionnaires de configuration
ANALYSIS_MODES = {
    "classic": "Analyse classique",
    "chunks": "Analyse par chunks", 
    "synthesis": "Synthèse globale"
}

PROVIDERS = [
    "Ollama local",
    "Ollama distant", 
    "RunPod.io"
]

# Fonctions de fallback simples pour éviter les erreurs
def on_provider_change_fn(provider):
    """Fonction de fallback pour changement de provider."""
    print(f"FALLBACK: Provider changé vers {provider}")
    return (
        gr.update(visible=(provider == "Ollama distant"), value=""),
        gr.update(visible=(provider == "RunPod.io"), value=""),
        gr.update(visible=(provider == "RunPod.io"), value=""),
        gr.update(value=f"Provider: {provider} (mode fallback)")
    )

def analyze_with_modes_fn(*args):
    """Fonction de fallback pour analyse."""
    print(f"FALLBACK: Fonction d'analyse appelée avec {len(args)} arguments")
    # Retourne le bon nombre d'éléments attendus par Gradio
    return (
        "Fonction d'analyse non disponible (mode fallback)",  # resultat_final
        "",  # stats1
        "",  # text1
        "",  # preview1
        "",  # stats2
        "",  # text2
        "",  # preview2
        "",  # current_text1
        "",  # current_text2
        "",  # current_file_path1
        "",  # current_file_path2
        "Mode fallback activé",  # debug_info
        ""   # chunk_report
    )

def analyze_with_chunks_fn(*args):
    """Alias pour analyse."""
    return analyze_with_modes_fn(*args)

def on_test_connection_fn(provider, ollama_url_val, runpod_endpoint, runpod_token):
    """Fonction de fallback pour test connexion."""
    print(f"FALLBACK: Test de connexion appelé pour {provider}")
    models = ["llama2", "mistral", "codellama"]  # Modèles factices
    return (
        gr.update(choices=models, value=models[0]),
        gr.update(value=f"Connexion simulée pour {provider} (mode fallback)")
    )

def on_select_prompt_fn(name, store_dict):
    """Fonction de fallback pour sélection prompt."""
    print(f"FALLBACK: Sélection prompt {name}")
    return (
        gr.update(value=f"Prompt {name} (mode fallback)"),
        gr.update(value=f"FALLBACK: Prompt {name} sélectionné")
    )

def process_files_fn(file1, file2, nettoyer, anonymiser, force_processing, processing_mode):
    """Fonction de fallback pour traitement fichiers."""
    print("FALLBACK: Traitement fichiers appelé")
    return (
        "Traitement fichiers non disponible (mode fallback)",  # status
        "0 caractères",  # stats1
        "",  # preview1
        "",  # anon_report1
        "0 caractères",  # stats2
        "",  # preview2
        "",  # anon_report2
        "",  # current_text1
        "",  # current_text2
        "",  # current_file_path1
        "",  # current_file_path2
        "Mode fallback",  # debug_info
        ""   # chunk_report
    )

def clear_all_states_fn():
    """Fonction de fallback pour nettoyage."""
    print("FALLBACK: Nettoyage états appelé")
    return (
        "",  # unified_analysis_box
        "",  # text1_stats  
        "",  # preview1_box
        "",  # anonymization1_report
        "",  # text2_stats
        "",  # preview2_box
        "",  # anonymization2_report
        "",  # current_text1
        "",  # current_text2
        "",  # current_file_path1
        "",  # current_file_path2
        "Cache nettoyé (mode fallback)",  # debug_prompt_box
        ""   # chunk_report_box
    )

def save_url_callback_fn(url):
    """Fonction de fallback pour sauvegarde URL."""
    print(f"FALLBACK: Sauvegarde URL {url}")
    return gr.update(value=f"URL sauvegardée: {url} (mode fallback)")

# Fonctions utilitaires additionnelles
def load_prompts():
    """Charge les prompts par défaut."""
    return {
        "Analyse juridique": "Analysez ce document juridique en détail.",
        "Résumé": "Faites un résumé de ce document.",
        "Points clés": "Identifiez les points clés de ce document."
    }

def get_default_config():
    """Configuration par défaut."""
    return {
        "provider": "Ollama local",
        "model": "llama2",
        "temperature": 0.7,
        "max_tokens": 2000,
        "chunk_size": 3000,
        "chunk_overlap": 200
    }

# Variables globales pour compatibilité
DEFAULT_PROMPTS = load_prompts()
DEFAULT_CONFIG = get_default_config()

# Export de toutes les variables et fonctions
__all__ = [
    # Fonctions principales
    'on_provider_change_fn',
    'analyze_with_modes_fn',
    'analyze_with_chunks_fn',
    'on_test_connection_fn',
    'on_select_prompt_fn',
    'process_files_fn',
    'clear_all_states_fn',
    'save_url_callback_fn',
    # Variables
    'analysis_mode',
    'mode_analysis',
    'processing_mode',
    'enable_chunks',
    'enable_global_synthesis',
    'chunk_size',
    'chunk_overlap',
    'synthesis_prompt',
    # Dictionnaires
    'ANALYSIS_MODES',
    'PROVIDERS',
    'DEFAULT_PROMPTS',
    'DEFAULT_CONFIG',
    # Utilitaires
    'load_prompts',
    'get_default_config'
]

print("✅ Fichier de compatibilité callbacks_fix.py chargé (version étendue)")
print("⚠️  Toutes les fonctions sont en mode fallback")
print(f"📝 {len(__all__)} éléments exportés")
print("🔧 Variables manquantes ajoutées : analysis_mode, mode_analysis, etc.")
