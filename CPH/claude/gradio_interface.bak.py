#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface utilisateur Gradio pour OCR Juridique - Version Hybride COMPLÈTE
Version: 8.4-HYBRIDE-COMPLETE
Date: 2025-09-15
Fonctionnalités: Mode hybride 3 étapes + gestion prompts + corrections complètes
"""

import os
import gradio as gr
from datetime import datetime
import json
import threading
import time
import requests

# ========================================
# IMPORTS DES MODULES RÉELS
# ========================================

try:
    from hybrid_analyzer import (
        create_hybrid_analyzer, 
        SUPPORTED_DOMAINS, 
        SYNTHESIS_TYPES
    )
    from ai_wrapper import ai_call_wrapper, test_ai_connection, validate_ai_params
    from processing_pipeline import process_file_to_text
    from config_manager import load_ollama_config, save_url_on_change
    HYBRID_AVAILABLE = True
    print("Modules hybrides chargés avec succès")
except ImportError as e:
    print(f"Mode hybride non disponible: {e}")
    HYBRID_AVAILABLE = False
    # Fallback pour éviter les erreurs
    SUPPORTED_DOMAINS = {"Aucun": "Aucun domaine"}
    SYNTHESIS_TYPES = {"synthese_executive": "Synthèse exécutive"}

# Import du gestionnaire de prompts
try:
    from prompt_manager import (
        prompt_manager, 
        get_all_prompts_for_dropdown,
        get_prompt_content,
        save_prompt_ui,
        load_prompt_ui,
        delete_prompt_ui
    )
    PROMPT_MANAGER_AVAILABLE = True
    print("Gestionnaire de prompts chargé")
except ImportError as e:
    print(f"Gestionnaire de prompts non disponible: {e}")
    PROMPT_MANAGER_AVAILABLE = False

# ========================================
# GESTION ARRÊT AUTOMATIQUE RUNPOD
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
# CONFIGURATION GLOBALE
# ========================================

DEFAULT_CONFIG = {
    "provider": "Ollama local",
    "chunk_size": 3000,
    "chunk_overlap": 200,
    "ollama_url": "http://localhost:11434"
}

app_state = {
    "current_provider": "Ollama local",
    "ollama_url": "http://localhost:11434",
    "models_list": ["mistral:7b", "mixtral:8x7b", "llama3.1:8b"],
    "hybrid_analyzer": None,
    "ready": HYBRID_AVAILABLE
}

# Prompts juridiques de base (fallback si gestionnaire non disponible)
FALLBACK_PROMPTS = {
    "Analyse juridique hybride": """Analysez ce document juridique de manière approfondie :

1. IDENTIFICATION DU DOCUMENT
   - Nature et contexte
   - Parties impliquées
   - Enjeux principaux

2. ANALYSE JURIDIQUE
   - Fondements juridiques
   - Arguments développés
   - Points de droit

3. ÉVALUATION
   - Forces et faiblesses
   - Risques identifiés
   - Recommandations""",

    "Prompt par chunk": """Tu es juriste spécialisé en droit du travail. Voici un extrait d'un document de conclusions prud'homales. Analyse uniquement ce passage, sans inventer d'informations absentes.
1. Identifie les moyens de droit présents dans cet extrait et reformule-les en langage juridique français, sous forme narrative.
2. Si la section contient des arguments (discussion), résume-les de manière claire et structurée.
3. Ne conclus pas sur l'ensemble du dossier, limite-toi à ce passage.
4. N'inclus pas de noms propres ni de montants financiers.
Extrait :
[CHUNK]""",

    "Prompt de fusion": """Voici plusieurs analyses partielles issues d'un même document. Fusionne-les pour produire une synthèse unique, cohérente et sans répétitions, en respectant le style juridique demandé.
Consignes :
- Regroupe les moyens de droit en une présentation fluide et structurée.
- Intègre les arguments de la discussion en expliquant la logique des parties.
- Rédige en français juridique, sans puces ni numérotation, sous forme narrative.
- N'inclus pas de noms propres ni de montants financiers.
Analyses partielles :
[ANALYSES_CHUNKS]""",

    "Synthèse contractuelle": """Analysez ce contrat en détail :
- Obligations principales
- Clauses importantes
- Risques et garanties
- Recommandations d'amélioration""",

    "Analyse procédurale": """Examinez cette procédure :
- Chronologie des événements
- Moyens et arguments
- Respect des délais
- Prochaines étapes"""
}

# ========================================
# FONCTIONS UTILITAIRES
# ========================================

def load_config():
    """Charge la configuration."""
    if HYBRID_AVAILABLE:
        try:
            app_state["ollama_url"] = load_ollama_config()
            app_state["hybrid_analyzer"] = create_hybrid_analyzer(
                chunk_size=DEFAULT_CONFIG["chunk_size"],
                overlap=DEFAULT_CONFIG["chunk_overlap"]
            )
            return True
        except Exception as e:
            print(f"Erreur chargement config: {e}")
    return False

def save_config(url):
    """Sauvegarde la configuration."""
    if HYBRID_AVAILABLE:
        try:
            message = save_url_on_change(url)
            app_state["ollama_url"] = url
            return message
        except Exception as e:
            return f"Erreur sauvegarde: {str(e)}"
    return f"URL sauvegardée: {url}"

def track_activity():
    """Marque une activité utilisateur (pour RunPod)."""
    runpod_manager.update_activity()

def get_available_prompts():
    """Retourne les prompts disponibles."""
    if PROMPT_MANAGER_AVAILABLE:
        return get_all_prompts_for_dropdown()
    else:
        return list(FALLBACK_PROMPTS.keys())

def get_prompt_by_name(name):
    """Récupère un prompt par son nom."""
    if PROMPT_MANAGER_AVAILABLE:
        content = get_prompt_content(name)
        return content if content else ""
    else:
        return FALLBACK_PROMPTS.get(name, "")

# ========================================
# FONCTIONS D'ANALYSE
# ========================================

def analyze_hybrid_mode(text1, text2, prompt, model_step1, model_step2, model_step3,
                        provider, temperature, top_p, max_tokens, 
                        chunk_size, chunk_overlap, domain, synthesis_type, enable_step3,
                        ollama_url, runpod_endpoint, runpod_token):
    """Analyse avec le mode hybride 3 étapes."""
    
    track_activity()
    
    if not HYBRID_AVAILABLE:
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
        # Mise à jour de l'analyseur
        analyzer = create_hybrid_analyzer(chunk_size=chunk_size, overlap=chunk_overlap)
        
        # Configuration des modèles par étape
        analyzer.step1_config["preferred_models"] = [model_step1, "mistral:7b", "llama3.1:8b"]
        analyzer.step2_config["preferred_models"] = [model_step2, "mixtral:8x7b", "llama3.1:8b"]
        analyzer.step3_config["preferred_models"] = [model_step3, "llama3.1:8b", "mixtral:8x7b"]
        
        # Analyse hybride
        result = analyzer.analyze_hybrid(
            text=full_text,
            user_prompt=prompt.strip(),
            provider=provider,
            ollama_url=ollama_url,
            runpod_endpoint=runpod_endpoint,
            runpod_token=runpod_token,
            domain=domain if domain != "Aucun" else None,
            enable_step3=enable_step3,
            synthesis_type=synthesis_type
        )
        
        if result["success"]:
            formatted_result = analyzer.format_hybrid_result(result, prompt, doc_info)
            
            stats1 = f"{len(text1):,} caractères" if text1 else "Aucun texte"
            stats2 = f"{len(text2):,} caractères" if text2 else "Aucun texte"
            
            debug_info = f"""ANALYSE HYBRIDE EXÉCUTÉE
{'=' * 50}

MODE: Hybride 3 étapes
DOMAINE: {domain}
SYNTHÈSE: {synthesis_type if enable_step3 else 'Désactivée'}

MODÈLES UTILISÉS:
- Étape 1 (Extraction): {result['step1']['model_used']}
- Étape 2 (Fusion): {result['step2']['model_used']}
- Étape 3 (Synthèse): {result['step3']['model_used'] if result.get('step3') and result['step3']['success'] else 'N/A'}

PERFORMANCE:
- Chunks traités: {result['metadata']['chunks_count']}
- Durée totale: {result['metadata']['processing_time']:.1f}s
- Score efficacité: {result['stats']['efficiency_score']}

ARCHITECTURE:
✓ Extraction rapide → {model_step1}
✓ Fusion intelligente → {model_step2}
{'✓' if enable_step3 else '○'} Synthèse premium → {model_step3 if enable_step3 else 'Désactivée'}"""
            
            step3_duration = result['step3']['duration'] if result.get('step3') and result['step3']['success'] else 0
            step3_status = "Exécutée" if result.get('step3') and result['step3']['success'] else "Non exécutée"
            
            analysis_report = f"""RAPPORT D'ANALYSE HYBRIDE

Étapes exécutées:
1. Extraction par chunks: {result['step1']['duration']:.1f}s
2. Fusion et harmonisation: {result['step2']['duration']:.1f}s
3. Synthèse narrative: {step3_duration:.1f}s ({step3_status})

Optimisation qualité/prix réussie"""
            
            return (formatted_result, stats1, stats2, debug_info, analysis_report)
        else:
            error_msg = f"ERREUR ANALYSE HYBRIDE\n\n{result['error']}"
            return (error_msg, "", "", error_msg, "Erreur")
            
    except Exception as e:
        error_msg = f"ERREUR TECHNIQUE\n\n{str(e)}"
        return (error_msg, "", "", error_msg, "Erreur technique")

def analyze_classic_mode(text1, text2, prompt, model, provider, temperature, top_p, max_tokens,
                        ollama_url, runpod_endpoint, runpod_token):
    """Analyse classique directe."""
    
    track_activity()
    
    if text1 and text2:
        full_text = f"=== DOCUMENT 1 ===\n{text1}\n\n=== DOCUMENT 2 ===\n{text2}"
    else:
        full_text = text1 or text2
    
    if not full_text:
        return ("ERREUR: Aucun texte fourni", "", "", "Aucun texte", "")
    
    try:
        result = ai_call_wrapper(
            text=full_text,
            prompt=prompt.strip(),
            modele=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens_out=max_tokens,
            provider=provider,
            ollama_url_val=ollama_url,
            runpod_endpoint=runpod_endpoint,
            runpod_token=runpod_token
        )
        
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
# FONCTIONS CALLBACK
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
    """Test de connexion réel."""
    track_activity()
    
    if not HYBRID_AVAILABLE:
        return (
            gr.update(choices=["mistral:7b", "llama3.1:8b"], value="mistral:7b"),
            gr.update(value="⚠️ Mode dégradé - Test simulé")
        )
    
    try:
        result, models = test_ai_connection(provider, ollama_url, runpod_endpoint, runpod_token)
        
        if models:
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

def process_file_real(file_path, clean_text=True, anonymize=False):
    """Traitement de fichier réel."""
    track_activity()
    
    if not file_path:
        return "Aucun fichier", "0 caractères", ""
    
    if not HYBRID_AVAILABLE:
        return "⚠️ Modules non disponibles", "Simulation", "Texte simulé"
    
    try:
        # CORRECTION: utiliser force_ocr au lieu de force_processing
        message, stats, text, file_type, anon_report = process_file_to_text(
            file_path, clean_text, anonymize, force_ocr=True
        )
        
        if "⛔" in message:
            return message, "Erreur", ""
        
        return message, stats, text
    except Exception as e:
        return f"❌ Erreur: {str(e)}", "Erreur", ""

def clear_all_fields():
    """Nettoie tous les champs."""
    track_activity()
    default_prompt = get_prompt_by_name("Prompt par chunk")
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
# FONCTIONS GESTION PROMPTS
# ========================================

def load_selected_prompt(prompt_name):
    """Charge un prompt sélectionné."""
    if not prompt_name:
        return "", "❌ Sélectionnez un prompt"
    
    if PROMPT_MANAGER_AVAILABLE:
        return load_prompt_ui(prompt_name)
    else:
        content = FALLBACK_PROMPTS.get(prompt_name, "")
        status = f"✅ Prompt '{prompt_name}' chargé (fallback)"
        return content, status

def save_current_prompt(name, content):
    """Sauvegarde le prompt actuel."""
    if not name.strip():
        return "❌ Nom de prompt requis"
    
    if not content.strip():
        return "❌ Contenu de prompt requis"
    
    if PROMPT_MANAGER_AVAILABLE:
        return save_prompt_ui(name, content)
    else:
        return "❌ Gestionnaire de prompts non disponible"

def delete_selected_prompt(name):
    """Supprime un prompt sélectionné."""
    if not name.strip():
        return "❌ Sélectionnez un prompt à supprimer"
    
    if PROMPT_MANAGER_AVAILABLE:
        return delete_prompt_ui(name)
    else:
        return "❌ Gestionnaire de prompts non disponible"

def refresh_prompt_list():
    """Actualise la liste des prompts."""
    prompts = get_available_prompts()
    return gr.update(choices=prompts), f"✅ {len(prompts)} prompts disponibles"

# ========================================
# INTERFACE GRADIO PRINCIPALE
# ========================================

def create_hybrid_interface():
    """Crée l'interface Gradio avec mode hybride."""
    
    load_config()
    
    with gr.Blocks(
        title="OCR Juridique - Mode Hybride",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(f"""
        # 📚 OCR Juridique - Mode Hybride v8.4
        
        **Analyse juridique optimisée qualité/prix** avec architecture 3 étapes :
        
        📄 **Étape 1** : Extraction chunks → Mistral 7B (rapide, économique)  
        🔀 **Étape 2** : Fusion + harmonisation → Mixtral 8x7B (contexte large)  
        ✨ **Étape 3** : Synthèse narrative premium → LLaMA3.1 8B (qualité rédactionnelle)
        
        **État** : {'✅ Mode hybride disponible' if HYBRID_AVAILABLE else '⚠️ Mode dégradé'}
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
                        
                        with gr.Group(visible=False) as runpod_autostop_group:
                            gr.Markdown("### ⏱️ Arrêt automatique")
                            autostop_timeout = gr.Slider(
                                minimum=5, maximum=60, value=15, step=5,
                                label="Inactivité (minutes)",
                                info="Arrêt automatique du pod après inactivité"
                            )
                            configure_autostop_btn = gr.Button(
                                "⚙️ Configurer arrêt auto", 
                                variant="secondary"
                            )
                        
                        test_btn = gr.Button("🔍 Tester la connexion", variant="primary")
                        
                        status_msg = gr.Textbox(
                            label="Statut",
                            value="Prêt" if HYBRID_AVAILABLE else "Mode dégradé",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Modèles par étape")
                        
                        model_step1 = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="mistral:7b",
                            label="Étape 1 - Extraction (Mistral 7B recommandé)",
                            info="Modèle rapide pour extraction des chunks"
                        )
                        
                        model_step2 = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="mixtral:8x7b",
                            label="Étape 2 - Fusion (Mixtral 8x7B recommandé)",
                            info="Modèle à grand contexte pour fusion"
                        )
                        
                        model_step3 = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="llama3.1:8b",
                            label="Étape 3 - Synthèse (LLaMA3.1 8B recommandé)",
                            info="Modèle pour qualité rédactionnelle"
                        )
                        
                        with gr.Row():
                            temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="Température")
                            top_p = gr.Slider(0, 1, value=0.9, step=0.1, label="Top-p")
                            max_tokens = gr.Slider(500, 8000, value=3000, step=500, label="Max tokens")
            
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
                            label="Mode",
                            info="Hybride = optimisation qualité/prix"
                        )
                        
                        # Configuration hybride
                        with gr.Group() as hybrid_config:
                            chunk_size = gr.Slider(
                                1000, 5000, value=3000, step=500,
                                label="Taille chunks", info="Caractères par chunk"
                            )
                            chunk_overlap = gr.Slider(
                                0, 500, value=200, step=50,
                                label="Chevauchement", info="Continuité entre chunks"
                            )
                            
                            domain = gr.Dropdown(
                                choices=["Aucun"] + list(SUPPORTED_DOMAINS.values()) if HYBRID_AVAILABLE else ["Aucun"],
                                value="Aucun",
                                label="Domaine spécialisé"
                            )
                            
                            enable_step3 = gr.Checkbox(
                                label="Synthèse narrative premium (étape 3)",
                                value=True,
                                info="Qualité rédactionnelle optimale"
                            )
                            
                            synthesis_type = gr.Dropdown(
                                choices=list(SYNTHESIS_TYPES.values()) if HYBRID_AVAILABLE else ["Synthèse exécutive"],
                                value="Synthèse exécutive",
                                label="Type de synthèse"
                            )
                        
                        # Gestion des prompts
                        gr.Markdown("### 📝 Gestion des prompts")
                        with gr.Accordion("📂 Prompts disponibles", open=True):
                            prompt_selector = gr.Dropdown(
                                choices=get_available_prompts(),
                                label="Sélectionner un prompt",
                                value=""
                            )
                            
                            with gr.Row():
                                load_prompt_btn = gr.Button("📂 Charger", size="sm")
                                refresh_prompts_btn = gr.Button("🔄 Actualiser", size="sm")
                            
                            prompt_load_status = gr.Textbox(
                                label="Statut chargement", 
                                interactive=False,
                                lines=1
                            )

                        with gr.Accordion("✏️ Créer/Modifier prompts", open=False):
                            prompt_name_input = gr.Textbox(
                                label="Nom du prompt",
                                placeholder="Nom pour sauvegarder (ex: Mon prompt chunk personnalisé)"
                            )
                            
                            with gr.Row():
                                save_prompt_btn = gr.Button("💾 Sauvegarder", variant="primary", size="sm")
                                delete_prompt_btn = gr.Button("🗑️ Supprimer", variant="stop", size="sm")
                            
                            prompt_save_status = gr.Textbox(
                                label="Statut sauvegarde",
                                interactive=False,
                                lines=1
                            )
                    
                    with gr.Column(scale=2):
                        prompt_text = gr.Textbox(
                            label="Prompt d'analyse",
                            lines=12,
                            value=get_prompt_by_name("Prompt par chunk"),
                            placeholder="Tapez votre prompt ici ou sélectionnez-en un ci-dessus..."
                        )
                        
                        analyze_btn = gr.Button(
                            "🚀 Lancer l'analyse", 
                            variant="primary", 
                            size="lg"
                        )
            
            # ======= PROMPTS =======
            with gr.Tab("📝 Prompts"):
                gr.Markdown("""
                ## Gestion des prompts personnalisés
                
                **Prompts inclus par défaut :**
                - **Prompt par chunk** : Pour l'analyse par fragments (étape 1)
                - **Prompt de fusion** : Pour fusionner les analyses (étape 2)
                - **Analyse juridique hybride** : Analyse complète
                - **Synthèse contractuelle** : Spécialisé contrats
                - **Analyse procédurale** : Spécialisé procédures
                
                Vous pouvez créer vos propres prompts ou modifier les existants.
                """)
                
                if PROMPT_MANAGER_AVAILABLE:
                    # Interface complète de gestion
                    with gr.Group():
                        gr.Markdown("### 📝 Gestion complète des prompts")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                prompt_mgmt_selector = gr.Dropdown(
                                    choices=get_available_prompts(),
                                    label="Prompts disponibles",
                                    value=""
                                )
                                
                                with gr.Row():
                                    mgmt_load_btn = gr.Button("📂 Charger", size="sm")
                                    mgmt_refresh_btn = gr.Button("🔄 Actualiser", size="sm")
                                    mgmt_delete_btn = gr.Button("🗑️ Supprimer", size="sm", variant="stop")
                                
                                prompt_mgmt_status = gr.Textbox(
                                    label="Statut",
                                    interactive=False,
                                    lines=3
                                )
                            
                            with gr.Column(scale=2):
                                prompt_mgmt_name = gr.Textbox(
                                    label="Nom du prompt",
                                    placeholder="Nom pour sauvegarder..."
                                )
                                
                                prompt_mgmt_content = gr.Textbox(
                                    label="Contenu du prompt",
                                    lines=15,
                                    placeholder="Tapez votre prompt ici..."
                                )
                                
                                mgmt_save_btn = gr.Button("💾 Sauvegarder", variant="primary")
                else:
                    gr.Warning("Gestionnaire de prompts non disponible. Utilisez les prompts par défaut dans l'onglet Analyse.")
            
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
        
        # Gestion fichiers avec gestion des options
        def handle_file1_with_options(file, clean, anon):
            if file:
                message, stats, text = process_file_real(file.name, clean, anon)
                return message, stats, text
            return "", "0 caractères", ""
        
        def handle_file2_with_options(file, clean, anon):
            if file:
                message, stats, text = process_file_real(file.name, clean, anon)
                return stats, text
            return "0 caractères", ""
        
        file1.change(
            handle_file1_with_options, 
            inputs=[file1, clean1, anon1], 
            outputs=[status_msg, stats1, text1]
        )
        
        file2.change(
            handle_file2_with_options, 
            inputs=[file2, clean2, anon2], 
            outputs=[stats2, text2]
        )
        
        # Gestion des prompts dans l'onglet Analyse
        load_prompt_btn.click(
            load_selected_prompt,
            inputs=[prompt_selector],
            outputs=[prompt_text, prompt_load_status]
        )
        
        save_prompt_btn.click(
            save_current_prompt,
            inputs=[prompt_name_input, prompt_text],
            outputs=[prompt_save_status]
        )
        
        delete_prompt_btn.click(
            delete_selected_prompt,
            inputs=[prompt_selector],
            outputs=[prompt_save_status]
        )
        
        refresh_prompts_btn.click(
            refresh_prompt_list,
            outputs=[prompt_selector, prompt_load_status]
        )
        
        # Gestion des prompts dans l'onglet Prompts (si disponible)
        if PROMPT_MANAGER_AVAILABLE:
            mgmt_load_btn.click(
                load_prompt_ui,
                inputs=[prompt_mgmt_selector],
                outputs=[prompt_mgmt_content, prompt_mgmt_status]
            )
            
            mgmt_save_btn.click(
                save_prompt_ui,
                inputs=[prompt_mgmt_name, prompt_mgmt_content],
                outputs=[prompt_mgmt_status]
            )
            
            mgmt_delete_btn.click(
                delete_prompt_ui,
                inputs=[prompt_mgmt_selector],
                outputs=[prompt_mgmt_status]
            )
            
            def mgmt_refresh_and_update():
                prompts = get_available_prompts()
                return gr.update(choices=prompts), f"✅ {len(prompts)} prompts disponibles"
            
            mgmt_refresh_btn.click(
                mgmt_refresh_and_update,
                outputs=[prompt_mgmt_selector, prompt_mgmt_status]
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
        
        # Sauvegarde URL
        ollama_url.change(save_config, inputs=[ollama_url], outputs=[status_msg])
        
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
# FONCTION BUILD_UI
# ========================================

def build_ui():
    """Point d'entrée pour main_ocr.py"""
    print("Construction interface hybride...")
    print(f"Mode hybride: {'Disponible' if HYBRID_AVAILABLE else 'Non disponible'}")
    print(f"Gestion prompts: {'Disponible' if PROMPT_MANAGER_AVAILABLE else 'Non disponible'}")
    print(f"Gestion RunPod: Arrêt automatique configuré")
    
    # Initialiser les prompts par défaut
    if PROMPT_MANAGER_AVAILABLE:
        print(f"Prompts disponibles: {len(get_available_prompts())}")
    
    return create_hybrid_interface()

# ========================================
# LANCEMENT DIRECT
# ========================================

if __name__ == "__main__":
    print("Interface hybride - Test direct")
    demo = create_hybrid_interface()
    demo.launch(server_port=7860)
