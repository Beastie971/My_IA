#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface Gradio OCR Juridique - VERSION COMPLÈTE AVEC MODULES HYBRIDES
Version: 8.9-COMPLETE-HYBRID
Date: 2025-09-17
OBJECTIF: Interface complète avec analyse hybride + analyse simple intégrées
"""

import os
import gradio as gr
import json
import time
import requests
from datetime import datetime
import threading

# ========================================
# IMPORTS DES MODULES EXISTANTS
# ========================================

# Modules de base (déjà fonctionnels)
from processing_pipeline import process_file_to_text
from ai_providers import generate_with_ollama, generate_with_runpod, get_ollama_models
from config import DEFAULT_PROMPT_TEXT, EXPERT_PROMPT_TEXT

# Modules avancés (à intégrer)
try:
    from chunck_analysis import ChunkAnalyzer
    from ai_wrapper import ai_call_wrapper, test_ai_connection, validate_ai_params
    CHUNK_ANALYSIS_AVAILABLE = True
    print("✅ Module d'analyse par chunks disponible")
except ImportError as e:
    print(f"⚠️ Module d'analyse par chunks non disponible: {e}")
    CHUNK_ANALYSIS_AVAILABLE = False

# Tentative d'import du module hybride
try:
    # Si vous avez un module hybrid_analyzer
    from hybrid_analyzer import HybridAnalyzer, create_hybrid_analyzer
    HYBRID_AVAILABLE = True
    print("✅ Module d'analyse hybride disponible")
except ImportError:
    print("⚠️ Module d'analyse hybride non disponible - mode simple uniquement")
    HYBRID_AVAILABLE = False

# Module de gestion des prompts
try:
    from prompt_manager import prompt_manager, get_all_prompts_for_dropdown, get_prompt_content
    PROMPT_MANAGER_AVAILABLE = True
    print("✅ Gestionnaire de prompts disponible")
except ImportError:
    print("⚠️ Gestionnaire de prompts non disponible")
    PROMPT_MANAGER_AVAILABLE = False

# ========================================
# CONFIGURATION GLOBALE
# ========================================

app_state = {
    "ollama_url": "http://localhost:11434",
    "models_list": ["mistral:7b", "mixtral:8x7b", "llama3.1:8b"],
    "chunk_analyzer": None,
    "hybrid_analyzer": None
}

# Prompts juridiques optimisés professionnels
PROMPTS_FALLBACK = {
    "Analyse juridique approfondie": """Tu es avocat spécialisé en droit du travail avec 15 ans d'expérience au Barreau de Paris.

MISSION : Analyse juridique approfondie selon la méthodologie du Conseil d'État :

1. QUALIFICATION JURIDIQUE DES FAITS
- Identifie la nature exacte des faits (licenciement, rupture conventionnelle, etc.)
- Détermine les dispositions légales applicables (Code du travail, Convention collective)
- Précise le régime juridique et les conditions d'application

2. ANALYSE DES MOYENS DE DROIT
- Examine chaque argument juridique invoqué par les parties
- Vérifie la conformité aux procédures légales obligatoires
- Identifie les vices de procédure éventuels et leurs conséquences
- Analyse l'articulation des moyens principaux et subsidiaires

3. ÉVALUATION DES PRÉTENTIONS
- Examine le bien-fondé juridique de chaque demande
- Calcule les indemnités selon les barèmes légaux en vigueur
- Évalue les chances de succès devant la juridiction compétente
- Identifie les risques et opportunités procédurales

STYLE : Français juridique précis, rédaction narrative structurée SANS listes à puces.
MÉTHODE : Raisonnement juridique rigoureux, citations exactes, pas de reformulation des faits.
FOCUS : Qualification juridique, moyens de droit, évaluation des prétentions.""",

    "Extraction par chunks": """Tu es juriste senior spécialisé en procédure civile et droit du travail.

MISSION : Extraction juridique précise de ce fragment de document.

MÉTHODE D'ANALYSE :
- Identifie UNIQUEMENT les éléments juridiques présents dans ce passage
- Relève les moyens de droit invoqués avec leur qualification exacte
- Note les références légales citées (articles, jurisprudences, conventions)
- Extrait les demandes formulées avec leurs fondements

CONSIGNES STRICTES :
- Reste dans les limites de ce fragment, sans extrapolation
- Utilise le vocabulaire juridique français approprié
- N'invente aucune information absente du texte
- Structure l'extraction par catégories juridiques

STYLE : Concision juridique, terminologie précise, objectivité absolue.""",

    "Fusion juridique intelligente": """Tu es magistrat expérimenté spécialisé en droit social.

MISSION : Fusion cohérente d'analyses juridiques partielles en synthèse unifiée.

MÉTHODE DE SYNTHÈSE :
- Consolide les informations juridiques sans répétitions
- Respecte la chronologie procédurale et la logique juridique française
- Harmonise les qualifications juridiques et les références légales
- Structure selon l'ordre logique : qualification → moyens → prétentions

OBJECTIFS :
- Créer une analyse globale cohérente et structurée
- Éliminer les redondances tout en préservant l'exhaustivité
- Maintenir la rigueur juridique et la précision terminologique
- Proposer une évaluation équilibrée des positions des parties

STYLE : Rédaction juridique narrative, français soutenu, argumentation structurée.""",

    "Synthèse premium": """Tu es avocat aux Conseils et juriste d'entreprise senior.

MISSION : Synthèse juridique premium de qualité Cour de cassation.

EXIGENCES DE QUALITÉ MAXIMALE :
- Analyse juridique de niveau expertise (15+ ans d'expérience)
- Rédaction juridique impeccable selon les standards du Barreau
- Structure argumentative rigoureuse et logique implacable
- Qualification juridique précise et références exactes

ARCHITECTURE DE LA SYNTHÈSE :
- Introduction : Qualification des faits et enjeux juridiques
- Développement : Analyse des moyens articulés selon leur force
- Évaluation : Chances de succès, risques, recommandations stratégiques
- Conclusion : Position juridique synthétique et orientations

STYLE : Excellence rédactionnelle, terminologie juridique irréprochable, 
argumentation de niveau Conseil d'État, clarté pédagogique."""
}

# Types d'analyse disponibles
ANALYSIS_MODES = ["Simple directe", "Hybride par chunks"]

# ========================================
# FONCTIONS D'ANALYSE INTÉGRÉES
# ========================================

def analyze_simple_direct(text, prompt, model, provider, temperature, max_tokens, 
                         ollama_url, runpod_endpoint, runpod_token):
    """Analyse simple directe optimisée pour le juridique."""
    try:
        # Configuration optimisée pour analyses juridiques
        juridical_config = {
            "temperature": min(temperature, 0.4),  # Plus déterministe pour juridique
            "top_p": 0.8,                         # Réduction créativité
            "repeat_penalty": 1.1,                # Éviter répétitions
            "num_ctx": 16384,                     # Contexte élargi pour juridique
            "num_predict": max_tokens,
            "top_k": 40                           # Vocabulaire focalisé
        }
        
        if provider == "RunPod.io":
            if not runpod_endpoint or not runpod_token:
                return "⛔ Endpoint et token RunPod requis"
            
            result = generate_with_runpod(
                model=model, 
                system_prompt=prompt, 
                user_text=text,
                num_ctx=juridical_config["num_ctx"],
                num_predict=juridical_config["num_predict"],
                temperature=juridical_config["temperature"],
                endpoint=runpod_endpoint,
                token=runpod_token
            )
        else:
            url = ollama_url if provider == "Ollama distant" else "http://localhost:11434"
            result = generate_with_ollama(
                model=model,
                system_prompt=prompt,
                user_text=text,
                num_ctx=juridical_config["num_ctx"],
                num_predict=juridical_config["num_predict"],
                temperature=juridical_config["temperature"],
                ollama_url=url
            )
        
        return result
    except Exception as e:
        return f"⛔ Erreur analyse simple: {str(e)}"

def analyze_hybrid_chunks(text, prompt, model, provider, temperature, max_tokens,
                         chunk_size, chunk_overlap, models_config,
                         ollama_url, runpod_endpoint, runpod_token):
    """Analyse hybride par chunks optimisée pour le juridique - utilise ChunkAnalyzer."""
    
    # Configuration chunks adaptée au juridique
    juridical_chunk_size = max(chunk_size, 4000)    # Chunks plus larges pour contexte juridique
    juridical_overlap = max(chunk_overlap, 400)     # Chevauchement important pour continuité
    
    # Utiliser prioritairement ChunkAnalyzer qui fonctionne
    if CHUNK_ANALYSIS_AVAILABLE:
        return analyze_by_chunks_enhanced(text, prompt, model, provider, temperature, max_tokens,
                                        juridical_chunk_size, juridical_overlap, ollama_url, runpod_endpoint, runpod_token)
    
    # Si module hybride disponible, essayer (mode expérimental)
    elif HYBRID_AVAILABLE:
        try:
            return analyze_hybrid_mode_experimental(text, prompt, models_config, provider, temperature, max_tokens,
                                                  juridical_chunk_size, juridical_overlap, ollama_url, runpod_endpoint, runpod_token)
        except Exception as e:
            # Fallback vers chunks si hybride échoue
            print(f"Hybride échoué, fallback vers chunks: {e}")
            if CHUNK_ANALYSIS_AVAILABLE:
                return analyze_by_chunks_enhanced(text, prompt, model, provider, temperature, max_tokens,
                                                juridical_chunk_size, juridical_overlap, ollama_url, runpod_endpoint, runpod_token)
    
    # Fallback : analyse simple avec message informatif
    result = analyze_simple_direct(text, prompt, model, provider, temperature, max_tokens,
                                 ollama_url, runpod_endpoint, runpod_token)
    
    return f"""⚠️ ANALYSE SIMPLIFIÉE (Modules chunks non disponibles)

{result}

💡 RECOMMANDATION : Pour l'analyse juridique optimale par chunks, installez :
- chunck_analysis.py (découpage intelligent par sections juridiques)
- hybrid_analyzer.py (analyse hybride 3 étapes : extraction → fusion → synthèse premium)"""

def analyze_hybrid_mode_experimental(text, prompt, models_config, provider, temperature, max_tokens,
                                   chunk_size, chunk_overlap, ollama_url, runpod_endpoint, runpod_token):
    """Mode hybride avec votre module HybridAnalyzer réel."""
    try:
        # Initialiser selon votre API réelle
        if not app_state["hybrid_analyzer"]:
            # Créer un ai_provider_manager compatible
            class AIProviderManager:
                def __init__(self):
                    self.provider = provider
                    self.ollama_url = ollama_url
                    self.runpod_endpoint = runpod_endpoint
                    self.runpod_token = runpod_token
                
                def call_model(self, model, system_prompt, user_text, **kwargs):
                    if provider == "RunPod.io":
                        return generate_with_runpod(
                            model=model,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            num_ctx=16384,
                            num_predict=max_tokens,
                            temperature=0.3,
                            endpoint=runpod_endpoint,
                            token=runpod_token
                        )
                    else:
                        url = ollama_url if provider == "Ollama distant" else "http://localhost:11434"
                        return generate_with_ollama(
                            model=model,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            num_ctx=16384,
                            num_predict=max_tokens,
                            temperature=0.3,
                            ollama_url=url
                        )
            
            ai_provider = AIProviderManager()
            app_state["hybrid_analyzer"] = create_hybrid_analyzer(ai_provider)
        
        analyzer = app_state["hybrid_analyzer"]
        
        # Utiliser analyze_document avec les bons paramètres
        result = analyzer.analyze_document(
            text=text,
            analysis_type="juridique",
            system_prompt=prompt,
            provider=provider,
            model=models_config.get("step2", "mixtral:8x7b"),
            temperature=0.3,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Formatage du résultat selon le type de retour
        if isinstance(result, dict):
            analysis_content = result.get('content', result.get('analysis', str(result)))
            processing_time = result.get('processing_time', 0)
            confidence = result.get('confidence', 'N/A')
            
            formatted = f"""⚡ ANALYSE HYBRIDE JURIDIQUE

🎯 Configuration :
- Type d'analyse : juridique spécialisée
- Provider : {provider}
- Modèle principal : {models_config.get("step2", "mixtral:8x7b")}
- Chunks : {chunk_size} caractères

📊 Métadonnées :
- Durée : {processing_time:.1f}s
- Confiance : {confidence}

📝 ANALYSE JURIDIQUE :

{analysis_content}

💡 Analyse produite avec HybridAnalyzer optimisé juridique."""
        else:
            formatted = f"""⚡ ANALYSE HYBRIDE JURIDIQUE

📝 RÉSULTAT :

{result}

💡 Analyse produite avec votre module HybridAnalyzer."""
        
        return formatted
        
    except Exception as e:
        return f"⛔ Erreur module hybride: {str(e)}"

def analyze_by_chunks_enhanced(text, prompt, model, provider, temperature, max_tokens,
                              chunk_size, chunk_overlap, ollama_url, runpod_endpoint, runpod_token):
    """Analyse par chunks optimisée avec prompts juridiques spécialisés."""
    try:
        # Initialiser l'analyseur avec configuration juridique
        if not app_state["chunk_analyzer"]:
            app_state["chunk_analyzer"] = ChunkAnalyzer(chunk_size=chunk_size, overlap=chunk_overlap)
        
        analyzer = app_state["chunk_analyzer"]
        
        # Découper avec préservation structure juridique
        chunks = analyzer.smart_chunk_text(text, preserve_structure=True)
        print(f"📄 Découpage juridique en {len(chunks)} chunks")
        
        # Configuration optimisée pour juridique
        juridical_config = {
            "temperature": 0.3,     # Déterminisme pour extraction juridique
            "num_ctx": 12288,       # Contexte adapté aux chunks
            "num_predict": 2000,    # Analyses de chunks détaillées
            "top_p": 0.8,
            "repeat_penalty": 1.1
        }
        
        # Prompt spécialisé extraction juridique
        extraction_prompt = PROMPTS_FALLBACK["Extraction par chunks"]
        
        # Fonction d'IA optimisée juridique
        def ai_function_juridical(text, prompt, **kwargs):
            if provider == "RunPod.io":
                return generate_with_runpod(
                    model=model,
                    system_prompt=prompt,
                    user_text=text,
                    num_ctx=juridical_config["num_ctx"],
                    num_predict=juridical_config["num_predict"],
                    temperature=juridical_config["temperature"],
                    endpoint=runpod_endpoint,
                    token=runpod_token
                )
            else:
                url = ollama_url if provider == "Ollama distant" else "http://localhost:11434"
                return generate_with_ollama(
                    model=model,
                    system_prompt=prompt,
                    user_text=text,
                    num_ctx=juridical_config["num_ctx"],
                    num_predict=juridical_config["num_predict"],
                    temperature=juridical_config["temperature"],
                    ollama_url=url
                )
        
        # Analyser les chunks avec prompt spécialisé
        analyzed_chunks = analyzer.analyze_chunks_with_ai(chunks, extraction_prompt, ai_function_juridical)
        
        # Synthèse avec prompt juridique spécialisé
        fusion_prompt = PROMPTS_FALLBACK["Fusion juridique intelligente"]
        
        synthesis = analyzer.synthesize_analyses(analyzed_chunks, fusion_prompt, ai_function_juridical)
        
        return f"""📊 ANALYSE JURIDIQUE PAR CHUNKS OPTIMISÉE

🔧 Configuration utilisée :
- Chunks juridiques : {len(chunks)} segments de {chunk_size:,} caractères
- Chevauchement : {chunk_overlap} caractères pour continuité procédurale
- Modèle d'extraction : {model} (température 0.3 pour précision)
- Prompt spécialisé : Extraction juridique ciblée + Fusion intelligente

📝 SYNTHÈSE JURIDIQUE :

{synthesis}"""
        
    except Exception as e:
        return f"⛔ Erreur analyse juridique par chunks: {str(e)}"

def analyze_hybrid_mode_enhanced(text, prompt, models_config, provider, temperature, max_tokens,
                                chunk_size, chunk_overlap, ollama_url, runpod_endpoint, runpod_token):
    """Analyse hybride 3 étapes optimisée pour qualité juridique maximale."""
    if not HYBRID_AVAILABLE:
        return "⛔ Module d'analyse hybride non disponible"
    
    try:
        # Initialiser l'analyseur hybride selon votre API
        if not app_state["hybrid_analyzer"]:
            # Créer un provider manager basique pour votre API
            class BasicAIProviderManager:
                def __init__(self):
                    self.provider = provider
                    self.ollama_url = ollama_url
                    self.runpod_endpoint = runpod_endpoint
                    self.runpod_token = runpod_token
                
                def call_ai(self, model, system_prompt, user_text, **kwargs):
                    if provider == "RunPod.io":
                        return generate_with_runpod(
                            model=model,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            num_ctx=16384,
                            num_predict=max_tokens,
                            temperature=0.3,
                            endpoint=runpod_endpoint,
                            token=runpod_token
                        )
                    else:
                        url = ollama_url if provider == "Ollama distant" else "http://localhost:11434"
                        return generate_with_ollama(
                            model=model,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            num_ctx=16384,
                            num_predict=max_tokens,
                            temperature=0.3,
                            ollama_url=url
                        )
            
            ai_provider = BasicAIProviderManager()
            app_state["hybrid_analyzer"] = create_hybrid_analyzer(ai_provider)
        
        analyzer = app_state["hybrid_analyzer"]
        
        # Configuration modèles optimale pour juridique
        step1_model = models_config.get("step1", "mistral:7b")
        step2_model = models_config.get("step2", "mixtral:8x7b")  
        step3_model = models_config.get("step3", "mixtral:8x7b")
        
        # Analyse hybride avec votre API
        result = analyzer.analyze_hybrid(
            text=text,
            user_prompt=prompt,
            provider=provider,
            ollama_url=ollama_url,
            runpod_endpoint=runpod_endpoint,
            runpod_token=runpod_token,
            step1_model=step1_model,
            step2_model=step2_model,
            step3_model=step3_model,
            temperature=0.3,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if result.get("success"):
            formatted = f"""⚡ ANALYSE JURIDIQUE HYBRIDE PREMIUM

🎯 Architecture 3 étapes optimisée :
└── Phase 1 : Extraction juridique ciblée ({step1_model})
└── Phase 2 : Fusion contextuelle intelligente ({step2_model}) 
└── Phase 3 : Synthèse premium niveau expertise ({step3_model})

📊 Performance :
- Chunks traités : {result.get('metadata', {}).get('chunks_count', 'N/A')}
- Durée totale : {result.get('metadata', {}).get('processing_time', 0):.1f}s
- Qualité juridique : Premium (configuration optimisée)

📝 SYNTHÈSE JURIDIQUE FINALE :

{result.get('result', 'Résultat non disponible')}

💡 Analyse produite avec méthodologie d'expertise juridique française."""
            
            return formatted
        else:
            return f"⛔ Erreur analyse hybride: {result.get('error', 'Erreur inconnue')}"
            
    except Exception as e:
        return f"⛔ Erreur analyse hybride: {str(e)}"

def analyze_hybrid_mode(text, prompt, models_config, provider, temperature, max_tokens,
                       chunk_size, chunk_overlap, ollama_url, runpod_endpoint, runpod_token):
    """Analyse hybride 3 étapes (si module disponible)."""
    if not HYBRID_AVAILABLE:
        return "⛔ Module d'analyse hybride non disponible - utilisez l'analyse par chunks"
    
    try:
        # Initialiser l'analyseur hybride
        if not app_state["hybrid_analyzer"]:
            app_state["hybrid_analyzer"] = create_hybrid_analyzer(
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
        
        analyzer = app_state["hybrid_analyzer"]
        
        # Configuration des modèles par étape
        step1_model = models_config.get("step1", model)
        step2_model = models_config.get("step2", model)  
        step3_model = models_config.get("step3", model)
        
        # Analyse hybride
        result = analyzer.analyze_hybrid(
            text=text,
            user_prompt=prompt,
            provider=provider,
            ollama_url=ollama_url,
            runpod_endpoint=runpod_endpoint,
            runpod_token=runpod_token,
            step1_model=step1_model,
            step2_model=step2_model,
            step3_model=step3_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if result["success"]:
            return analyzer.format_hybrid_result(result, prompt, f"{len(text):,} caractères")
        else:
            return f"⛔ Erreur analyse hybride: {result['error']}"
            
    except Exception as e:
        return f"⛔ Erreur analyse hybride: {str(e)}"

def analyze_text_complete(text, prompt, analysis_mode, model, provider, temperature, max_tokens,
                         chunk_size, chunk_overlap, models_config,
                         ollama_url, runpod_endpoint, runpod_token):
    """Fonction d'analyse complète qui route vers le bon mode."""
    if not text.strip():
        return "⛔ Aucun texte à analyser", "", "", "Aucun texte"
    
    start_time = time.time()
    
    try:
        print(f"🚀 Analyse {analysis_mode} - Modèle: {model} - Provider: {provider}")
        
        # Router vers le bon mode d'analyse
        if analysis_mode == "Simple directe":
            result = analyze_simple_direct(
                text, prompt, model, provider, temperature, max_tokens,
                ollama_url, runpod_endpoint, runpod_token
            )
        
        elif analysis_mode == "Hybride par chunks":
            result = analyze_hybrid_chunks(
                text, prompt, model, provider, temperature, max_tokens,
                chunk_size, chunk_overlap, models_config,
                ollama_url, runpod_endpoint, runpod_token
            )
        
        else:
            result = "⛔ Mode d'analyse non reconnu"
        
        # Formatage du résultat
        processing_time = time.time() - start_time
        timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        
        formatted_result = f"""{'=' * 80}
ANALYSE JURIDIQUE COMPLÈTE - {timestamp}
{'=' * 80}

MODE: {analysis_mode}
MODÈLE: {model}
FOURNISSEUR: {provider}
TEXTE: {len(text):,} caractères
DURÉE: {processing_time:.1f} secondes

{'-' * 80}
PROMPT UTILISÉ
{'-' * 80}

{prompt}

{'-' * 80}
RÉSULTAT DE L'ANALYSE
{'-' * 80}

{result}
"""
        
        stats = f"{len(text):,} caractères"
        debug = f"Analyse {analysis_mode} - {provider} - {model} - {processing_time:.1f}s"
        
        return formatted_result, stats, debug, f"Analyse {analysis_mode} terminée"
        
    except Exception as e:
        error_msg = f"⛔ ERREUR ANALYSE: {str(e)}"
        return error_msg, "", error_msg, "Erreur"

# ========================================
# FONCTIONS DE CONFIGURATION
# ========================================

def test_connection_complete(provider, ollama_url, runpod_endpoint, runpod_token):
    """Test de connexion utilisant les modules existants."""
    try:
        if hasattr(test_ai_connection, '__call__'):  # Si ai_wrapper disponible
            success, models = test_ai_connection(provider, ollama_url, runpod_endpoint, runpod_token)
            return success, models
        else:
            # Fallback sur les fonctions simples
            if provider == "RunPod.io":
                if not runpod_endpoint or not runpod_token:
                    return "⛔ Endpoint et token requis", []
                headers = {"Authorization": f"Bearer {runpod_token}", "Content-Type": "application/json"}
                response = requests.get(f"{runpod_endpoint}/v1/models", headers=headers, timeout=10)
                if response.status_code in [200, 404]:
                    models = ["meta-llama/Llama-3.1-70B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
                    return "✅ Connexion RunPod réussie", models
                else:
                    return f"⛔ Erreur RunPod {response.status_code}", []
            else:
                url = ollama_url if provider == "Ollama distant" else "http://localhost:11434"
                models = get_ollama_models(url)
                return f"✅ Connexion Ollama - {len(models)} modèles", models
                
    except Exception as e:
        return f"⛔ Erreur test connexion: {str(e)}", []

def get_available_prompts():
    """Récupère tous les prompts disponibles."""
    if PROMPT_MANAGER_AVAILABLE:
        try:
            return get_all_prompts_for_dropdown()
        except:
            pass
    return list(PROMPTS_FALLBACK.keys())

def load_prompt_content(prompt_name):
    """Charge le contenu d'un prompt."""
    if PROMPT_MANAGER_AVAILABLE:
        try:
            content = get_prompt_content(prompt_name)
            if content:
                return content
        except:
            pass
    return PROMPTS_FALLBACK.get(prompt_name, "")

# ========================================
# INTERFACE GRADIO COMPLÈTE
# ========================================

def create_complete_interface():
    """Interface Gradio complète avec tous les modes d'analyse."""
    
    with gr.Blocks(title="OCR Juridique - Version Complète", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(f"""
        # 📚 OCR Juridique - Version Complète v8.9
        
        **Modes d'analyse disponibles :**
        - 🚀 **Simple directe** : Analyse en une fois (rapide)
        - ⚡ **Hybride par chunks** : Découpage intelligent → Extraction → Fusion → Synthèse {('✅' if HYBRID_AVAILABLE or CHUNK_ANALYSIS_AVAILABLE else '⛔')}
        
        **Modules chargés :**
        - Traitement fichiers : ✅
        - Analyse par chunks/hybride : {('✅' if CHUNK_ANALYSIS_AVAILABLE or HYBRID_AVAILABLE else '⛔')}
        - Gestionnaire prompts : {('✅' if PROMPT_MANAGER_AVAILABLE else '⛔')}
        """)
        
        with gr.Tabs():
            
            # ===== CONFIGURATION =====
            with gr.Tab("⚙️ Configuration"):
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
                        
                        test_btn = gr.Button("🔍 Tester connexion", variant="primary")
                        status = gr.Textbox(label="Statut", interactive=False, value="Prêt")
                    
                    with gr.Column():
                        gr.Markdown("### Modèles")
                        
                        model_main = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="mistral:7b",
                            label="Modèle principal"
                        )
                        
                        # Configuration modèles optimisée juridique
                        with gr.Group(visible=(HYBRID_AVAILABLE or CHUNK_ANALYSIS_AVAILABLE)) as hybrid_models_group:
                            gr.Markdown("#### 🏛️ Configuration expertise juridique par phase")
                            
                            model_step1 = gr.Dropdown(
                                choices=app_state["models_list"],
                                value="mistral:7b",
                                label="⚡ Phase 1 - Extraction juridique",
                                info="Modèle rapide pour identifier moyens de droit (Mistral 7B optimal)"
                            )
                            model_step2 = gr.Dropdown(
                                choices=app_state["models_list"], 
                                value="mixtral:8x7b" if "mixtral:8x7b" in app_state["models_list"] else app_state["models_list"][0],
                                label="🧠 Phase 2 - Fusion procédurale",
                                info="Raisonnement juridique complexe (Mixtral 8x7B recommandé)"
                            )
                            model_step3 = gr.Dropdown(
                                choices=app_state["models_list"],
                                value="mixtral:8x7b" if "mixtral:8x7b" in app_state["models_list"] else app_state["models_list"][0],
                                label="⚖️ Phase 3 - Synthèse d'expertise",
                                info="Qualité rédactionnelle niveau Barreau (Mixtral 8x7B optimal)"
                            )
                            
                            # Paramètres juridiques optimisés
                            gr.Markdown("##### ⚙️ Paramètres juridiques optimisés")
                            
                            with gr.Row():
                                chunk_size = gr.Slider(
                                    3000, 8000, value=5000, step=500,
                                    label="Taille chunks juridiques",
                                    info="Chunks plus larges pour contexte procédural"
                                )
                                chunk_overlap = gr.Slider(
                                    200, 800, value=500, step=100,
                                    label="Chevauchement procédural", 
                                    info="Continuité entre sections juridiques"
                                )
                        
                        with gr.Row():
                            temperature = gr.Slider(
                                0, 1, value=0.3, step=0.1, 
                                label="Température juridique",
                                info="0.2-0.4 optimal pour analyses juridiques (déterminisme)"
                            )
                            max_tokens = gr.Slider(
                                1000, 6000, value=4000, step=500, 
                                label="Longueur analyse",
                                info="Analyses juridiques détaillées (3000-5000 recommandé)"
                            )
            
            # ===== DOCUMENTS =====
            with gr.Tab("📄 Documents"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Document")
                        file_input = gr.File(label="Fichier", file_types=[".pdf", ".txt", ".docx"])
                        
                        with gr.Row():
                            clean_text = gr.Checkbox(label="Nettoyer", value=True)
                            anonymize = gr.Checkbox(label="Anonymiser", value=False)
                        
                        text_input = gr.Textbox(label="Texte", lines=15)
                        stats = gr.Textbox(label="Statistiques", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### Configuration d'analyse")
                        
                        analysis_mode = gr.Radio(
                            choices=[mode for mode in ANALYSIS_MODES if 
                                   mode == "Simple directe" or 
                                   (mode == "Hybride par chunks" and (CHUNK_ANALYSIS_AVAILABLE or HYBRID_AVAILABLE))],
                            value="Simple directe",
                            label="Mode d'analyse"
                        )
                        
                        # Configuration chunks adaptée juridique
                        with gr.Group() as chunk_config:
                            gr.Markdown("##### 📄 Découpage procédural intelligent")
                            
                            with gr.Row():
                                chunk_size = gr.Slider(
                                    3000, 8000, value=5000, step=500,
                                    label="Taille segments juridiques",
                                    info="Segments plus larges pour contexte procédural complet"
                                )
                                chunk_overlap = gr.Slider(
                                    200, 800, value=500, step=100,
                                    label="Continuité procédurale",
                                    info="Chevauchement pour préserver liens juridiques"
                                )
                        
                        # Sélection de prompts juridiques
                        prompt_selector = gr.Dropdown(
                            choices=get_available_prompts(),
                            value=get_available_prompts()[0] if get_available_prompts() else "",
                            label="📋 Expertise juridique prédéfinie",
                            info="Prompts optimisés par spécialité juridique"
                        )
                        
                        load_prompt_btn = gr.Button("📝 Charger prompt")
            
            # ===== ANALYSE =====
            with gr.Tab("🔍 Analyse"):
                prompt_text = gr.Textbox(
                    label="Prompt d'analyse",
                    lines=12,
                    value=load_prompt_content(get_available_prompts()[0]) if get_available_prompts() else "",
                    placeholder="Décrivez l'analyse juridique souhaitée..."
                )
                
                analyze_btn = gr.Button("🚀 Lancer l'analyse", variant="primary", size="lg")
            
            # ===== RÉSULTATS =====
            with gr.Tab("📊 Résultats"):
                result_text = gr.Textbox(
                    label="Résultat de l'analyse",
                    lines=30,
                    show_copy_button=True
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Nettoyer", variant="stop")
                    
                    with gr.Column():
                        debug_info = gr.Textbox(
                            label="Informations de debug",
                            lines=5,
                            interactive=False
                        )
        
        # ========================================
        # ÉVÉNEMENTS
        # ========================================
        
        # Configuration provider
        def on_provider_change(prov):
            return (
                gr.update(visible=(prov == "Ollama distant")),
                gr.update(visible=(prov == "RunPod.io")),
                gr.update(visible=(prov == "RunPod.io")),
                f"Provider: {prov}"
            )
        
        provider.change(
            on_provider_change,
            inputs=[provider],
            outputs=[ollama_url, runpod_endpoint, runpod_token, status]
        )
        
        # Test connexion et mise à jour des modèles avec sélection intelligente
        def test_and_update_models(prov, url, endpoint, token):
            message, models = test_connection_complete(prov, url, endpoint, token)
            
            if models:
                app_state["models_list"] = models
                
                # Sélection intelligente des modèles par défaut
                def smart_select_model(preferred_models, fallback_index=0):
                    for preferred in preferred_models:
                        if preferred in models:
                            return preferred
                    return models[fallback_index] if len(models) > fallback_index else models[0]
                
                # Modèles optimaux par phase
                step1_model = smart_select_model(["mistral:7b", "mistral:latest", "llama3.1:8b"], 0)
                step2_model = smart_select_model(["mixtral:8x7b", "llama3.1:8b", "mistral:latest"], 
                                               min(1, len(models)-1))
                step3_model = smart_select_model(["llama3.1:8b", "mixtral:8x7b", "mistral:latest"], 
                                               min(2, len(models)-1))
                
                return (
                    gr.update(choices=models, value=models[0]),  # model_main
                    gr.update(choices=models, value=step1_model),  # model_step1
                    gr.update(choices=models, value=step2_model),  # model_step2  
                    gr.update(choices=models, value=step3_model),  # model_step3
                    f"✅ {message} | Modèles optimisés: {step1_model} → {step2_model} → {step3_model}"
                )
            else:
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    f"⛔ {message}"
                )
        
        test_btn.click(
            test_and_update_models,
            inputs=[provider, ollama_url, runpod_endpoint, runpod_token],
            outputs=[model_main, model_step1, model_step2, model_step3, status]
        )
        
        # Gestion fichiers
        def handle_file_upload(file, clean, anon):
            if file:
                try:
                    message, stats_result, text, file_type, anon_report = process_file_to_text(
                        file.name, clean, anon, force_ocr=False
                    )
                    return message, stats_result, text
                except Exception as e:
                    return f"⛔ Erreur: {str(e)}", "Erreur", ""
            return "", "0 caractères", ""
        
        file_input.change(
            handle_file_upload,
            inputs=[file_input, clean_text, anonymize],
            outputs=[status, stats, text_input]
        )
        
        # Chargement de prompts
        def load_selected_prompt(prompt_name):
            content = load_prompt_content(prompt_name)
            return content, f"✅ Prompt '{prompt_name}' chargé"
        
        load_prompt_btn.click(
            load_selected_prompt,
            inputs=[prompt_selector],
            outputs=[prompt_text, status]
        )
        
        # Affichage conditionnel de la config hybride selon le mode
        def toggle_hybrid_config(mode):
            return gr.update(visible=(mode == "Hybride par chunks"))
        
        analysis_mode.change(
            toggle_hybrid_config,
            inputs=[analysis_mode],
            outputs=[hybrid_models_group]
        )
        
        # Analyse principale
        def run_complete_analysis(text, prompt, mode, model_main, provider, temp, max_tok,
                                chunk_sz, chunk_ovlp, model1, model2, model3,
                                ollama_url, runpod_endpoint, runpod_token):
            
            models_config = {
                "step1": model1 if HYBRID_AVAILABLE else model_main,
                "step2": model2 if HYBRID_AVAILABLE else model_main,
                "step3": model3 if HYBRID_AVAILABLE else model_main
            }
            
            return analyze_text_complete(
                text, prompt, mode, model_main, provider, temp, max_tok,
                chunk_sz, chunk_ovlp, models_config,
                ollama_url, runpod_endpoint, runpod_token
            )
        
        analyze_btn.click(
            run_complete_analysis,
            inputs=[
                text_input, prompt_text, analysis_mode, model_main, provider, 
                temperature, max_tokens, chunk_size, chunk_overlap
            ] + [model_step1, model_step2, model_step3] + [
                ollama_url, runpod_endpoint, runpod_token
            ],
            outputs=[result_text, stats, debug_info, status]
        )
        
        # Nettoyage
        def clear_all_fields():
            return "", "", "", "🧹 Interface nettoyée", ""
        
        clear_btn.click(
            clear_all_fields,
            outputs=[text_input, result_text, stats, status, debug_info]
        )
    
    return demo

# ========================================
# FONCTIONS D'EXPORT
# ========================================

def build_ui():
    """Point d'entrée pour main_ocr.py"""
    print("🚀 Construction interface complète avec analyse hybride par chunks...")
    print(f"⚡ Analyse hybride/chunks: {'✅ Disponible' if (CHUNK_ANALYSIS_AVAILABLE or HYBRID_AVAILABLE) else '⛔ Non disponible'}")
    print(f"📝 Gestionnaire prompts: {'✅ Disponible' if PROMPT_MANAGER_AVAILABLE else '⛔ Non disponible'}")
    return create_complete_interface()

# ========================================
# LANCEMENT DIRECT
# ========================================

if __name__ == "__main__":
    print("🧪 Test interface complète")
    demo = create_complete_interface()
    
    ports_to_try = [7860, 7861, 7862, 7863, 7864]
    
    for port in ports_to_try:
        try:
            print(f"🌐 Lancement sur port {port}")
            demo.launch(
                server_port=port,
                server_name="127.0.0.1",
                share=False,
                show_error=True
            )
            break
        except Exception as e:
            print(f"⛔ Port {port} occupé: {e}")
            continue
