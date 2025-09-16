#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface utilisateur Gradio pour OCR Juridique - VERSION CORRIGÉE
Version: 8.3-MODULES-EXISTANTS
Date: 2025-09-16
Utilise VOS modules existants au lieu de chercher des modules inexistants
"""

import os
import gradio as gr
from datetime import datetime
import json
import threading
import time
import requests

# ========================================
# IMPORTS DES MODULES EXISTANTS RÉELS
# ========================================

try:
    # VOS modules qui existent réellement
    from ai_providers import (
        generate_with_ollama, 
        generate_with_runpod, 
        get_ollama_models,
        test_connection,
        calculate_smart_timeout
    )
    from chunck_analysis import ChunkAnalyzer
    from prompt_manager import prompt_manager, get_all_prompts_for_dropdown, get_prompt_content
    from processing_pipeline import process_file_to_text, do_analysis_only
    from config import DEFAULT_PROMPT_TEXT, EXPERT_PROMPT_TEXT
    
    MODULES_AVAILABLE = True
    print("✅ Tous les modules existants chargés avec succès")
    
except ImportError as e:
    print(f"❌ Erreur import modules: {e}")
    MODULES_AVAILABLE = False

# ========================================
# CONFIGURATION RÉELLE BASÉE SUR VOS MODULES
# ========================================

# Domaines supportés (basés sur vos prompts existants)
SUPPORTED_DOMAINS = {
    "droit_travail": "Droit du travail",
    "contractuel": "Droit contractuel", 
    "procedure": "Procédure civile",
    "immobilier": "Droit immobilier"
}

# Types de synthèse disponibles
SYNTHESIS_TYPES = {
    "synthese_executive": "Synthèse exécutive",
    "analyse_detaillee": "Analyse détaillée",
    "rapport_structure": "Rapport structuré",
    "conclusions": "Conclusions juridiques"
}

# Configuration par défaut
DEFAULT_CONFIG = {
    "provider": "Ollama local",
    "chunk_size": 3000,
    "chunk_overlap": 200,
    "ollama_url": "http://localhost:11434"
}

app_state = {
    "current_provider": "Ollama local",
    "ollama_url": "http://localhost:11434",
    "models_list": ["mistral:7b-instruct", "llama3:latest", "mixtral:8x7b"],
    "chunk_analyzer": None,
    "ready": MODULES_AVAILABLE
}

# ========================================
# GESTION ARRÊT AUTOMATIQUE RUNPOD (CONSERVÉ)
# ========================================

class RunPodManager:
    """Gestionnaire pour l'arrêt automatique des pods RunPod."""
    
    def __init__(self):
        self.last_activity = time.time()
        self.auto_stop_enabled = False
        self.timeout_minutes = 15
        self.pod_id = None
        self.api_key = None
        self.monitor_thread = None
        
    def update_activity(self):
        """Met à jour le timestamp de dernière activité."""
        self.last_activity = time.time()
        
    def configure_auto_stop(self, endpoint, token, timeout_minutes=15):
        """Configure l'arrêt automatique."""
        try:
            if "runpod" in endpoint and token:
                self.pod_id = self._extract_pod_id(endpoint)
                self.api_key = token
                self.timeout_minutes = timeout_minutes
                self.auto_stop_enabled = True
                
                if not self.monitor_thread or not self.monitor_thread.is_alive():
                    self.monitor_thread = threading.Thread(target=self._monitor_activity, daemon=True)
                    self.monitor_thread.start()
                
                return f"Arrêt auto configuré: {timeout_minutes}min d'inactivité"
            else:
                return "Configuration arrêt auto échouée"
        except Exception as e:
            return f"Erreur config arrêt auto: {str(e)}"
    
    def _extract_pod_id(self, endpoint):
        """Extrait l'ID du pod depuis l'endpoint."""
        try:
            if "runpod" in endpoint:
                parts = endpoint.split("//")[1].split(".")[0].split("-")
                return parts[-1] if len(parts) > 1 else None
        except:
            pass
        return None
    
    def _monitor_activity(self):
        """Thread de monitoring de l'activité."""
        while self.auto_stop_enabled:
            try:
                time.sleep(60)
                
                if self.auto_stop_enabled and self.pod_id and self.api_key:
                    inactive_time = (time.time() - self.last_activity) / 60
                    
                    if inactive_time > self.timeout_minutes:
                        print(f"Inactivité détectée: {inactive_time:.1f}min")
                        self._stop_pod()
                        break
                        
            except Exception as e:
                print(f"Erreur monitoring RunPod: {e}")
                break
    
    def _stop_pod(self):
        """Arrête le pod RunPod."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"https://api.runpod.ai/v2/{self.pod_id}/terminate"
            response = requests.post(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                print(f"Pod {self.pod_id} arrêté automatiquement")
                self.auto_stop_enabled = False
            else:
                print(f"Échec arrêt pod: {response.status_code}")
                
        except Exception as e:
            print(f"Erreur arrêt pod: {e}")
    
    def disable_auto_stop(self):
        """Désactive l'arrêt automatique."""
        self.auto_stop_enabled = False
        return "Arrêt automatique désactivé"

# Instance globale du gestionnaire
runpod_manager = RunPodManager()

# ========================================
# FONCTIONS UTILITAIRES ADAPTÉES À VOS MODULES
# ========================================

def load_config():
    """Charge la configuration."""
    if MODULES_AVAILABLE:
        try:
            # Initialiser l'analyseur de chunks avec VOS paramètres
            app_state["chunk_analyzer"] = ChunkAnalyzer(
                chunk_size=DEFAULT_CONFIG["chunk_size"],
                overlap=DEFAULT_CONFIG["chunk_overlap"]
            )
            return True
        except Exception as e:
            print(f"Erreur chargement config: {e}")
    return False

def track_activity():
    """Marque une activité utilisateur (pour RunPod)."""
    runpod_manager.update_activity()

# ========================================
# FONCTIONS D'ANALYSE ADAPTÉES À VOS MODULES
# ========================================

def analyze_hybrid_mode(text1, text2, prompt, model_step1, model_step2, model_step3,
                        provider, temperature, top_p, max_tokens, 
                        chunk_size, chunk_overlap, domain, synthesis_type, enable_step3,
                        ollama_url, runpod_endpoint, runpod_token):
    """Analyse hybride en utilisant VOS modules existants."""
    
    track_activity()
    
    if not MODULES_AVAILABLE:
        return fallback_analysis(text1, text2, prompt)
    
    # Préparation du texte
    if text1 and text2:
        full_text = f"=== DOCUMENT 1 ===\n{text1}\n\n=== DOCUMENT 2 ===\n{text2}"
        doc_info = f"2 documents - {len(text1):,} + {len(text2):,} caractères"
    elif text1:
        full_text = text1
        doc_info = f"1 document - {len(text1):,} caractères"
    elif text2:
        full_text = text2
        doc_info = f"1 document - {len(text2):,} caractères"
    else:
        return ("ERREUR: Aucun texte fourni", "", "", "Aucun texte", "")
    
    try:
        start_time = time.time()
        
        # ÉTAPE 1: Découpage et extraction avec le modèle rapide
        print(f"📄 Étape 1: Découpage et extraction avec {model_step1}")
        
        analyzer = ChunkAnalyzer(chunk_size=chunk_size, overlap=chunk_overlap)
        chunks = analyzer.smart_chunk_text(full_text, preserve_structure=True)
        
        # Analyse de chaque chunk avec le modèle rapide (Étape 1)
        step1_results = []
        step1_start = time.time()
        
        for chunk in chunks:
            chunk_prompt = get_prompt_content("Prompt par chunk") or prompt
            
            if provider == "RunPod.io":
                result = generate_with_runpod(
                    model=model_step1,
                    system_prompt=chunk_prompt,
                    user_text=chunk['text'],
                    num_ctx=4096,
                    num_predict=1000,
                    temperature=temperature,
                    endpoint=runpod_endpoint,
                    token=runpod_token
                )
            else:
                result = generate_with_ollama(
                    model=model_step1,
                    system_prompt=chunk_prompt,
                    user_text=chunk['text'],
                    num_ctx=4096,
                    num_predict=1000,
                    temperature=temperature,
                    ollama_url=ollama_url
                )
            
            step1_results.append(result)
            time.sleep(0.5)  # Pause entre chunks
        
        step1_duration = time.time() - step1_start
        
        # ÉTAPE 2: Fusion avec le modèle à grand contexte
        print(f"🔀 Étape 2: Fusion avec {model_step2}")
        step2_start = time.time()
        
        # Combiner tous les résultats de l'étape 1
        combined_analyses = "\n\n".join([f"=== ANALYSE CHUNK {i+1} ===\n{result}" 
                                        for i, result in enumerate(step1_results)])
        
        fusion_prompt = get_prompt_content("Prompt de fusion") or f"""
{prompt}

TÂCHE SPÉCIALE: Fusionnez les analyses suivantes en un rapport cohérent et structuré.
Éliminez les redondances et créez une synthèse unifiée.

ANALYSES À FUSIONNER:
{combined_analyses}
"""
        
        if provider == "RunPod.io":
            step2_result = generate_with_runpod(
                model=model_step2,
                system_prompt=fusion_prompt,
                user_text="FUSION DES ANALYSES DEMANDÉE",
                num_ctx=8192,
                num_predict=max_tokens,
                temperature=temperature,
                endpoint=runpod_endpoint,
                token=runpod_token
            )
        else:
            step2_result = generate_with_ollama(
                model=model_step2,
                system_prompt=fusion_prompt,
                user_text="FUSION DES ANALYSES DEMANDÉE",
                num_ctx=8192,
                num_predict=max_tokens,
                temperature=temperature,
                ollama_url=ollama_url
            )
        
        step2_duration = time.time() - step2_start
        
        # ÉTAPE 3: Synthèse premium (optionnelle)
        step3_result = step2_result
        step3_duration = 0
        
        if enable_step3:
            print(f"✨ Étape 3: Synthèse premium avec {model_step3}")
            step3_start = time.time()
            
            synthesis_prompt = f"""
{prompt}

TÂCHE DE SYNTHÈSE PREMIUM: Créez une synthèse narrative de haute qualité à partir de l'analyse suivante.
Type de synthèse demandé: {synthesis_type}
Domaine spécialisé: {domain if domain != "Aucun" else "Général"}

Améliorez la qualité rédactionnelle, la structure et la clarté.

ANALYSE À AMÉLIORER:
{step2_result}
"""
            
            if provider == "RunPod.io":
                step3_result = generate_with_runpod(
                    model=model_step3,
                    system_prompt=synthesis_prompt,
                    user_text="SYNTHÈSE PREMIUM DEMANDÉE",
                    num_ctx=4096,
                    num_predict=max_tokens,
                    temperature=temperature + 0.1,  # Légèrement plus créatif
                    endpoint=runpod_endpoint,
                    token=runpod_token
                )
            else:
                step3_result = generate_with_ollama(
                    model=model_step3,
                    system_prompt=synthesis_prompt,
                    user_text="SYNTHÈSE PREMIUM DEMANDÉE",
                    num_ctx=4096,
                    num_predict=max_tokens,
                    temperature=temperature + 0.1,
                    ollama_url=ollama_url
                )
            
            step3_duration = time.time() - step3_start
        
        total_duration = time.time() - start_time
        
        # Formatage du résultat final
        current_time = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        
        formatted_result = f"""{'=' * 80}
                    ANALYSE JURIDIQUE HYBRIDE 3 ÉTAPES
{'=' * 80}

HORODATAGE: {current_time}
ARCHITECTURE: Extraction → Fusion → Synthèse
{doc_info}

MODÈLES UTILISÉS:
🚀 Étape 1 (Extraction): {model_step1} - {step1_duration:.1f}s
🔀 Étape 2 (Fusion): {model_step2} - {step2_duration:.1f}s
{'✨ Étape 3 (Synthèse): ' + model_step3 + f' - {step3_duration:.1f}s' if enable_step3 else '○ Étape 3: Désactivée'}

PERFORMANCE:
- Chunks traités: {len(chunks)}
- Durée totale: {total_duration:.1f}s
- Fournisseur: {provider}

{'-' * 80}
                        PROMPT UTILISÉ
{'-' * 80}

{prompt}

{'-' * 80}
                    RÉSULTAT FINAL
{'-' * 80}

{step3_result}

{'-' * 80}
                DÉTAIL DES ANALYSES PAR CHUNKS
{'-' * 80}

{combined_analyses}
"""
        
        # Statistiques
        stats1 = f"{len(text1):,} caractères" if text1 else "Aucun texte"
        stats2 = f"{len(text2):,} caractères" if text2 else "Aucun texte"
        
        # Informations de debug
        debug_info = f"""ANALYSE HYBRIDE EXÉCUTÉE
{'=' * 50}

MODE: Hybride 3 étapes
DOMAINE: {domain}
SYNTHÈSE: {synthesis_type if enable_step3 else 'Désactivée'}

MODÈLES UTILISÉS:
- Étape 1 (Extraction): {model_step1}
- Étape 2 (Fusion): {model_step2}
- Étape 3 (Synthèse): {model_step3 if enable_step3 else 'N/A'}

PERFORMANCE:
- Chunks traités: {len(chunks)}
- Durée totale: {total_duration:.1f}s
- Score efficacité: {len(chunks) / total_duration * 60:.1f} chunks/min

ARCHITECTURE:
✓ Extraction rapide → {model_step1}
✓ Fusion intelligente → {model_step2}
{'✓' if enable_step3 else '○'} Synthèse premium → {model_step3 if enable_step3 else 'Désactivée'}"""
        
        # Rapport détaillé
        analysis_report = f"""RAPPORT D'ANALYSE HYBRIDE

Étapes exécutées:
1. Extraction par chunks: {step1_duration:.1f}s
2. Fusion et harmonisation: {step2_duration:.1f}s
3. Synthèse narrative: {step3_duration:.1f}s ({'Exécutée' if enable_step3 else 'Non exécutée'})

Optimisation qualité/prix réussie
Chunks traités: {len(chunks)}
Efficacité: {len(chunks) / total_duration * 60:.1f} chunks/min"""
        
        return (formatted_result, stats1, stats2, debug_info, analysis_report)
        
    except Exception as e:
        error_msg = f"ERREUR ANALYSE HYBRIDE\n\n{str(e)}"
        return (error_msg, "", "", error_msg, "Erreur")

def analyze_classic_mode(text1, text2, prompt, model, provider, temperature, top_p, max_tokens,
                        ollama_url, runpod_endpoint, runpod_token):
    """Analyse classique en utilisant VOS modules existants."""
    
    track_activity()
    
    # Préparation du texte
    if text1 and text2:
        full_text = f"=== DOCUMENT 1 ===\n{text1}\n\n=== DOCUMENT 2 ===\n{text2}"
    else:
        full_text = text1 or text2
    
    if not full_text:
        return ("ERREUR: Aucun texte fourni", "", "", "Aucun texte", "")
    
    try:
        # Utiliser directement vos fonctions existantes
        if provider == "RunPod.io":
            result = generate_with_runpod(
                model=model,
                system_prompt=prompt,
                user_text=full_text,
                num_ctx=8192,
                num_predict=max_tokens,
                temperature=temperature,
                endpoint=runpod_endpoint,
                token=runpod_token
            )
        else:
            result = generate_with_ollama(
                model=model,
                system_prompt=prompt,
                user_text=full_text,
                num_ctx=8192,
                num_predict=max_tokens,
                temperature=temperature,
                ollama_url=ollama_url
            )
        
        # Formatage
        current_time = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        formatted_result = f"""{'=' * 80}
                    ANALYSE JURIDIQUE CLASSIQUE
{'=' * 80}

HORODATAGE: {current_time}
MODÈLE: {model}
FOURNISSEUR: {provider}
MODE: Analyse directe
TEXTE: {len(full_text):,} caractères

{'-' * 80}
                        PROMPT UTILISÉ
{'-' * 80}

{prompt}

{'-' * 80}
                    RÉSULTAT DE L'ANALYSE
{'-' * 80}

{result}
"""
        
        stats1 = f"{len(text1):,} caractères" if text1 else "Aucun texte"
        stats2 = f"{len(text2):,} caractères" if text2 else "Aucun texte"
        
        debug_info = f"""ANALYSE CLASSIQUE EXÉCUTÉE

Mode: Direct (sans chunks)
Modèle: {model}
Provider: {provider}
Texte: {len(full_text):,} caractères"""
        
        return (formatted_result, stats1, stats2, debug_info, "Mode classique")
        
    except Exception as e:
        error_msg = f"ERREUR ANALYSE CLASSIQUE\n\n{str(e)}"
        return (error_msg, "", "", error_msg, "Erreur")

def fallback_analysis(text1, text2, prompt):
    """Analyse de fallback si modules non disponibles."""
    return (
        "⚠️ MODULES D'ANALYSE NON DISPONIBLES\n\nVérifiez l'installation des dépendances.",
        f"{len(text1) if text1 else 0:,} caractères",
        f"{len(text2) if text2 else 0:,} caractères",
        "Mode fallback - modules manquants",
        "Erreur: modules non disponibles"
    )

# ========================================
# FONCTIONS CALLBACK ADAPTÉES
# ========================================

def on_provider_change(provider):
    """Gestion du changement de fournisseur."""
    app_state["current_provider"] = provider
    
    ollama_visible = provider == "Ollama distant"
    runpod_visible = provider == "RunPod.io"
    
    if provider == "Ollama local":
        status = "✅ Ollama local configuré"
        url_value = ""
        runpod_manager.disable_auto_stop()
    elif provider == "Ollama distant":
        url_value = app_state["ollama_url"]
        status = f"🌐 Ollama distant: {url_value}"
        runpod_manager.disable_auto_stop()
    else:
        url_value = ""
        status = "☁️ RunPod - Configurez endpoint et token"
    
    return (
        gr.update(visible=ollama_visible, value=url_value),
        gr.update(visible=runpod_visible, value=""),
        gr.update(visible=runpod_visible, value=""),
        gr.update(visible=runpod_visible),
        gr.update(visible=runpod_visible),
        gr.update(value=status)
    )

def test_connection_real(provider, ollama_url, runpod_endpoint, runpod_token):
    """Test de connexion réel en utilisant VOS fonctions."""
    track_activity()
    
    if not MODULES_AVAILABLE:
        return (
            gr.update(choices=["mistral:7b-instruct", "llama3:latest"], value="mistral:7b-instruct"),
            gr.update(value="⚠️ Mode dégradé - Modules non disponibles")
        )
    
    try:
        # Utiliser VOTRE fonction de test
        result = test_connection(provider, ollama_url, runpod_endpoint, runpod_token)
        
        if "réussie" in result or "Connexion" in result:
            # Récupérer les modèles selon le provider
            if provider in ["Ollama local", "Ollama distant"]:
                url = ollama_url if provider == "Ollama distant" else "http://localhost:11434"
                models = get_ollama_models(url)
            else:
                # Modèles RunPod par défaut
                models = [
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    "NousResearch/Nous-Hermes-2-Yi-34B"
                ]
            
            app_state["models_list"] = models
            return (
                gr.update(choices=models, value=models[0]),
                gr.update(value=f"✅ {result} - {len(models)} modèles")
            )
        else:
            return (
                gr.update(),
                gr.update(value=f"❌ {result}")
            )
    except Exception as e:
        return (
            gr.update(),
            gr.update(value=f"❌ Erreur: {str(e)}")
        )

def configure_runpod_autostop(runpod_endpoint, runpod_token, timeout_minutes):
    """Configure l'arrêt automatique RunPod."""
    if runpod_endpoint and runpod_token:
        message = runpod_manager.configure_auto_stop(runpod_endpoint, runpod_token, timeout_minutes)
        return gr.update(value=f"✅ {message}")
    else:
        return gr.update(value="❌ Endpoint et token requis")

def select_prompt(prompt_name):
    """Sélection d'un prompt en utilisant VOTRE gestionnaire."""
    if not MODULES_AVAILABLE:
        return gr.update(), gr.update(value="❌ Gestionnaire de prompts non disponible")
    
    try:
        content = get_prompt_content(prompt_name)
        if content:
            return (
                gr.update(value=content),
                gr.update(value=f"✅ Prompt '{prompt_name}' chargé")
            )
        else:
            return (
                gr.update(),
                gr.update(value=f"❌ Prompt '{prompt_name}' non trouvé")
            )
    except Exception as e:
        return (
            gr.update(),
            gr.update(value=f"❌ Erreur: {str(e)}")
        )

def process_file_real(file_path, clean_text=True, anonymize=False):
    """Traitement de fichier en utilisant VOS fonctions."""
    track_activity()
    
    if not file_path:
        return "Aucun fichier", "0 caractères", ""
    
    if not MODULES_AVAILABLE:
        return "⚠️ Modules non disponibles", "Simulation", "Texte simulé"
    
    try:
        # Utiliser VOTRE fonction de traitement
        message, stats, text, file_type, anon_report = process_file_to_text(
            file_path, clean_text, anonymize, force_ocr=False
        )
        
        if "❌" in message:
            return message, "Erreur", ""
        
        return message, stats, text
    except Exception as e:
        return f"❌ Erreur: {str(e)}", "Erreur", ""

def clear_all_fields():
    """Nettoie tous les champs."""
    track_activity()
    default_prompt = get_prompt_content("Analyse juridique hybride") or DEFAULT_PROMPT_TEXT
    return (
        "",  # text1
        "",  # text2
        "",  # result
        "",  # stats1
        "",  # stats2
        "",  # debug
        default_prompt,  # prompt
        "🧹 Champs nettoyés",  # status
        ""   # analysis_report
    )

# ========================================
# INTERFACE GRADIO ADAPTÉE À VOS MODULES
# ========================================

def create_hybrid_interface():
    """Crée l'interface Gradio en utilisant VOS modules existants."""
    
    load_config()
    
    with gr.Blocks(
        title="OCR Juridique - Mode Hybride",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(f"""
        # 📚 OCR Juridique - Mode Hybride v8.3
        
        **Analyse juridique optimisée** avec architecture 3 étapes utilisant VOS modules existants :
        
        📄 **Étape 1** : Extraction chunks → Modèle rapide (économique)  
        🔀 **Étape 2** : Fusion + harmonisation → Modèle contexte large  
        ✨ **Étape 3** : Synthèse narrative → Modèle qualité rédactionnelle
        
        **État** : {'✅ Modules disponibles' if MODULES_AVAILABLE else '⚠️ Modules manquants'}
        """)
        
        with gr.Tabs():
            
            # ======= CONFIGURATION =======
            with gr.Tab("🔧 Configuration"):
                with gr.Row():
                    with gr.Column():
                        provider = gr.Radio(
                            choices=["Ollama local", "Ollama distant", "RunPod.io"],
                            value="Ollama local",
                            label="Fournisseur IA"
                        )
                        
                        ollama_url = gr.Textbox(
                            label="URL Ollama distant",
                            value=app_state["ollama_url"],
                            visible=False
                        )
                        
                        runpod_endpoint = gr.Textbox(
                            label="Endpoint RunPod",
                            visible=False
                        )
                        
                        runpod_token = gr.Textbox(
                            label="Token RunPod",
                            type="password",
                            visible=False
                        )
                        
                        # Configuration arrêt automatique RunPod
                        with gr.Group(visible=False) as runpod_autostop_group:
                            gr.Markdown("### ⏱️ Arrêt automatique")
                            autostop_timeout = gr.Slider(
                                minimum=5, maximum=60, value=15, step=5,
                                label="Inactivité (minutes)"
                            )
                            configure_autostop_btn = gr.Button(
                                "⚙️ Configurer arrêt auto", 
                                variant="secondary"
                            )
                        
                        test_btn = gr.Button("🔍 Tester la connexion", variant="primary")
                        
                        status_msg = gr.Textbox(
                            label="Statut",
                            value="Prêt" if MODULES_AVAILABLE else "Modules manquants",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Modèles par étape")
                        
                        model_step1 = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="mistral:7b-instruct",
                            label="Étape 1 - Extraction (rapide)",
                            allow_custom_value=True
                        )
                        
                        model_step2 = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="llama3:latest",
                            label="Étape 2 - Fusion (contexte large)",
                            allow_custom_value=True
                        )
                        
                        model_step3 = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="llama3:latest",
                            label="Étape 3 - Synthèse (qualité)",
                            allow_custom_value=True
                        )
                        
                        with gr.Row():
                            temperature = gr.Slider(0, 2, value=0.2, step=0.1, label="Température")
                            top_p = gr.Slider(0, 1, value=0.9, step=0.1, label="Top-p")
                            max_tokens = gr.Slider(500, 8000, value=2000, step=500, label="Max tokens")
            
            # ======= DOCUMENTS =======
            with gr.Tab("📄 Documents"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Document 1")
                        file1 = gr.File(label="Fichier 1", file_types=[".pdf", ".txt", ".docx"])
                        with gr.Row():
                            clean1 = gr.Checkbox(label="Nettoyer", value=True)
                            anon1 = gr.Checkbox(label="Anonymiser", value=False)
                        text1 = gr.Textbox(label="Texte 1", lines=10)
                        stats1 = gr.Textbox(label="Statistiques 1", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### Document 2")
                        file2 = gr.File(label="Fichier 2", file_types=[".pdf", ".txt", ".docx"])
                        with gr.Row():
                            clean2 = gr.Checkbox(label="Nettoyer", value=True)
                            anon2 = gr.Checkbox(label="Anonymiser", value=False)
                        text2 = gr.Textbox(label="Texte 2", lines=10)
                        stats2 = gr.Textbox(label="Statistiques 2", interactive=False)
                
                clear_btn = gr.Button("🗑️ Nettoyer", variant="stop")
            
            # ======= ANALYSE =======
            with gr.Tab("🔍 Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Mode d'analyse")
                        analysis_mode = gr.Radio(
                            choices=["Hybride 3 étapes", "Classique direct"],
                            value="Hybride 3 étapes",
                            label="Mode"
                        )
                        
                        # Configuration hybride
                        with gr.Group() as hybrid_config:
                            chunk_size = gr.Slider(
                                1000, 5000, value=3000, step=500,
                                label="Taille chunks"
                            )
                            chunk_overlap = gr.Slider(
                                0, 500, value=200, step=50,
                                label="Chevauchement"
                            )
                            
                            domain = gr.Dropdown(
                                choices=["Aucun"] + list(SUPPORTED_DOMAINS.values()),
                                value="Aucun",
                                label="Domaine spécialisé"
                            )
                            
                            enable_step3 = gr.Checkbox(
                                label="Synthèse narrative premium (étape 3)",
                                value=True
                            )
                            
                            synthesis_type = gr.Dropdown(
                                choices=list(SYNTHESIS_TYPES.values()),
                                value="Synthèse exécutive",
                                label="Type de synthèse"
                            )
                        
                        # Prompts prédéfinis
                        gr.Markdown("### Prompts")
                        available_prompts = get_all_prompts_for_dropdown() if MODULES_AVAILABLE else ["Analyse juridique hybride"]
                        prompt_selector = gr.Dropdown(
                            choices=available_prompts,
                            label="Prompts prédéfinis"
                        )
                        select_prompt_btn = gr.Button("📝 Charger")
                        prompt_status = gr.Textbox(label="Statut", interactive=False)
                    
                    with gr.Column(scale=2):
                        default_prompt = get_prompt_content("Analyse juridique hybride") if MODULES_AVAILABLE else DEFAULT_PROMPT_TEXT
                        prompt_text = gr.Textbox(
                            label="Prompt d'analyse",
                            lines=10,
                            value=default_prompt,
                            placeholder="Décrivez l'analyse souhaitée..."
                        )
                        
                        analyze_btn = gr.Button(
                            "🚀 Lancer l'analyse", 
                            variant="primary", 
                            size="lg"
                        )
            
            # ======= RÉSULTATS =======
            with gr.Tab("📊 Résultats"):
                result_text = gr.Textbox(
                    label="Résultat de l'analyse",
                    lines=25,
                    show_copy_button=True
                )
                
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("📋 Rapport d'analyse", open=False):
                            analysis_report = gr.Textbox(
                                label="Détails", lines=8, interactive=False
                            )
                    with gr.Column():
                        with gr.Accordion("🔧 Debug", open=False):
                            debug_info = gr.Textbox(
                                label="Informations techniques", lines=8, interactive=False
                            )
        
        # ========================================
        # ÉVÉNEMENTS
        # ========================================
        
        # Configuration
        provider.change(
            on_provider_change,
            inputs=[provider],
            outputs=[ollama_url, runpod_endpoint, runpod_token, runpod_autostop_group, configure_autostop_btn, status_msg]
        )
        
        test_btn.click(
            test_connection_real,
            inputs=[provider, ollama_url, runpod_endpoint, runpod_token],
            outputs=[model_step1, status_msg]
        )
        
        # Configuration arrêt automatique RunPod
        configure_autostop_btn.click(
            configure_runpod_autostop,
            inputs=[runpod_endpoint, runpod_token, autostop_timeout],
            outputs=[status_msg]
        )
        
        # Gestion fichiers
        def handle_file1(file):
            if file:
                message, stats, text = process_file_real(file.name, clean1.value, anon1.value)
                return message, stats, text
            return "", "0 caractères", ""
        
        def handle_file2(file):
            if file:
                message, stats, text = process_file_real(file.name, clean2.value, anon2.value)
                return stats, text
            return "0 caractères", ""
        
        file1.change(handle_file1, inputs=[file1], outputs=[status_msg, stats1, text1])
        file2.change(handle_file2, inputs=[file2], outputs=[stats2, text2])
        
        # Prompts
        select_prompt_btn.click(
            select_prompt,
            inputs=[prompt_selector],
            outputs=[prompt_text, prompt_status]
        )
        
        # Affichage conditionnel config hybride
        def toggle_config(mode):
            return gr.update(visible=(mode == "Hybride 3 étapes"))
        
        analysis_mode.change(
            toggle_config,
            inputs=[analysis_mode],
            outputs=[hybrid_config]
        )
        
        # Analyse principale
        def route_analysis(mode, text1, text2, prompt, 
                          model_step1, model_step2, model_step3,
                          provider, temperature, top_p, max_tokens,
                          chunk_size, chunk_overlap, domain, synthesis_type, enable_step3,
                          ollama_url, runpod_endpoint, runpod_token):
            
            if mode == "Hybride 3 étapes":
                return analyze_hybrid_mode(
                    text1, text2, prompt, model_step1, model_step2, model_step3,
                    provider, temperature, top_p, max_tokens,
                    chunk_size, chunk_overlap, domain, synthesis_type, enable_step3,
                    ollama_url, runpod_endpoint, runpod_token
                )
            else:
                return analyze_classic_mode(
                    text1, text2, prompt, model_step1, provider, 
                    temperature, top_p, max_tokens,
                    ollama_url, runpod_endpoint, runpod_token
                )
        
        analyze_btn.click(
            route_analysis,
            inputs=[
                analysis_mode, text1, text2, prompt_text,
                model_step1, model_step2, model_step3,
                provider, temperature, top_p, max_tokens,
                chunk_size, chunk_overlap, domain, synthesis_type, enable_step3,
                ollama_url, runpod_endpoint, runpod_token
            ],
            outputs=[result_text, stats1, stats2, debug_info, analysis_report]
        )
        
        # Nettoyage
        clear_btn.click(
            clear_all_fields,
            outputs=[text1, text2, result_text, stats1, stats2, debug_info, 
                    prompt_text, status_msg, analysis_report]
        )
        
        # Mise à jour des modèles après test de connexion
        def update_all_models(provider, ollama_url, runpod_endpoint, runpod_token):
            dropdown_result, status_result = test_connection_real(provider, ollama_url, runpod_endpoint, runpod_token)
            return dropdown_result, dropdown_result, dropdown_result, status_result
        
        test_btn.click(
            update_all_models,
            inputs=[provider, ollama_url, runpod_endpoint, runpod_token],
            outputs=[model_step1, model_step2, model_step3, status_msg]
        )
    
    return demo

# ========================================
# FONCTION BUILD_UI ADAPTÉE
# ========================================

def build_ui():
    """Point d'entrée pour main_ocr.py - utilise VOS modules."""
    print("Construction interface hybride avec modules existants...")
    print(f"Modules disponibles: {MODULES_AVAILABLE}")
    
    if MODULES_AVAILABLE:
        print("✅ Utilisation de vos modules:")
        print("  - ai_providers.py")
        print("  - chunck_analysis.py") 
        print("  - prompt_manager.py")
        print("  - processing_pipeline.py")
        print("  - config.py")
    else:
        print("⚠️ Certains modules manquants - Mode dégradé")
    
    return create_hybrid_interface()

# ========================================
# LANCEMENT DIRECT
# ========================================

if __name__ == "__main__":
    print("Interface hybride - Test direct avec modules existants")
    demo = create_hybrid_interface()
    demo.launch(server_port=7860)
