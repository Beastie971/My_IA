#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface utilisateur Gradio pour OCR Juridique v7 - VERSION COMPLÈTE CORRIGÉE
"""

import os
import traceback
import gradio as gr

from config import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT, PROMPT_STORE_PATH, calculate_text_stats
from file_processing import get_file_type, read_text_file, smart_clean
from anonymization import anonymize_text
from cache_manager import get_pdf_hash, load_ocr_cache, clear_ocr_cache
from ai_providers import get_ollama_models, refresh_models, test_connection
from prompt_manager import load_prompt_store
from processing_pipeline import process_file_to_text, do_analysis_only

# =============================================================================
# INTERFACE UTILISATEUR GRADIO COMPLÈTE
# =============================================================================

def build_ui():
    """Construit l'interface utilisateur Gradio complète."""
    models_list = get_ollama_models()
    store = load_prompt_store()
    prompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
    
    script_name = os.path.basename(__file__) if '__file__' in globals() else "ocr_legal_tool.py"

    with gr.Blocks(title=f"{script_name} - OCR Juridique + Ollama/RunPod") as demo:
        gr.Markdown("## OCR structuré + Analyse juridique (Ollama/RunPod) - Avec support TXT et Anonymisation")
        gr.Markdown(f"**Fichier des prompts** : `{PROMPT_STORE_PATH}`")

        # Section Upload et Configuration de base
        with gr.Row():
            input_file = gr.File(label="Uploader un fichier (PDF ou TXT)", file_types=[".pdf", ".txt", ".text"])
            with gr.Column():
                nettoyer = gr.Checkbox(label="Nettoyage avancé", value=True)
                anonymiser = gr.Checkbox(label="Anonymisation automatique", value=False)

        with gr.Row():
            force_processing = gr.Checkbox(label="Forcer nouveau traitement (ignorer cache PDF)", value=False)
            clear_cache_btn = gr.Button("Vider le cache OCR", variant="secondary", size="sm")
            cache_info = gr.Markdown("")

        # Section Fournisseur de modèles
        gr.Markdown("### Configuration du fournisseur de modèles")
        
        with gr.Row():
            provider_choice = gr.Radio(
                label="Fournisseur", 
                choices=["Ollama local", "Ollama distant", "RunPod.io"], 
                value="Ollama local"
            )
        
        # Champs de configuration - toujours présents mais visibilité conditionnelle
        with gr.Row():
            with gr.Column():
                ollama_url = gr.Textbox(
                    label="URL Ollama distant", 
                    value="http://localhost:11434",
                    placeholder="ex: http://192.168.1.100:11434 ou myserver.com:11434",
                    interactive=True,
                    visible=False
                )
            with gr.Column():
                runpod_endpoint = gr.Textbox(
                    label="Endpoint RunPod", 
                    placeholder="https://api.runpod.ai/v2/xxx/openai/v1",
                    interactive=True,
                    visible=False
                )
                runpod_token = gr.Textbox(
                    label="Token RunPod", 
                    placeholder="Token d'authentification",
                    type="password",
                    interactive=True,
                    visible=False
                )
        
        with gr.Row():
            test_connection_btn = gr.Button("Tester la connexion", variant="secondary", size="sm", visible=False)
            connection_status = gr.Markdown("✅ Utilisation d'Ollama local sur http://localhost:11434", visible=True)

        # Section Modèle et paramètres
        with gr.Row():
            if "mistral:7b-instruct" in models_list:
                default_model = "mistral:7b-instruct"
            elif "deepseek-coder:latest" in models_list:
                default_model = "deepseek-coder:latest"
            elif "mistral:latest" in models_list:
                default_model = "mistral:latest"
            else:
                default_model = models_list[0] if models_list else "mistral:latest"
                
            modele = gr.Dropdown(label="Modèle", choices=models_list, value=default_model)
            refresh_models_btn = gr.Button("Actualiser modèles", variant="secondary", size="sm")
            
        with gr.Row():
            profil = gr.Radio(label="Profil", choices=["Rapide", "Confort", "Maxi"], value="Confort")
            max_tokens_out = gr.Slider(label="Longueur (tokens)", minimum=256, maximum=2048, step=128, value=1280)
        
        with gr.Row():
            mode_analysis = gr.Radio(label="Mode", choices=["Standard", "Expert"], value="Standard")
            comparer = gr.Checkbox(label="Comparer avec l'autre mode", value=False)

        # Section Prompt
        gr.Markdown("### Prompt – gestion persistante")

        with gr.Row():
            prompt_selector = gr.Dropdown(label="Choisir un prompt", choices=prompt_names, value=DEFAULT_PROMPT_NAME)

        prompt_box = gr.Textbox(
            label="Contenu du prompt (modifiable)",
            value=store.get(DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT),
            lines=12,
            interactive=True
        )

        # Boutons principaux
        with gr.Row():
            process_btn = gr.Button("1. Traiter fichier (PDF/TXT)", variant="secondary")
            analyze_btn = gr.Button("2. Analyser", variant="primary", size="lg")
            full_btn = gr.Button("Traitement + Analyse", variant="primary")

        # Onglets de résultats
        with gr.Tabs():
            with gr.Tab("Analyse (mode choisi)"):
                analysis_box = gr.Textbox(label="Analyse juridique", lines=36, show_copy_button=True)
            with gr.Tab("Analyse (autre mode)"):
                analysis_alt_box = gr.Textbox(label="Analyse comparative", lines=24, show_copy_button=True)
            with gr.Tab("Contrôle qualité"):
                compare_box = gr.Textbox(label="Rapport CQ & comparatif", lines=18, show_copy_button=True)
            with gr.Tab("Texte source"):
                text_stats = gr.Textbox(label="Statistiques", lines=2, interactive=False)
                preview_box = gr.Textbox(label="Texte extrait/lu", interactive=False, show_copy_button=True, lines=25)
            with gr.Tab("Anonymisation"):
                anonymization_report_box = gr.Textbox(
                    label="Rapport d'anonymisation", 
                    interactive=False, 
                    show_copy_button=True, 
                    lines=25, 
                    placeholder="Le rapport d'anonymisation apparaîtra ici si l'anonymisation est activée..."
                )

        # États
        current_text = gr.State(value="")
        current_file_path = gr.State(value="")
        analysis_metadata = gr.State(value={})

        # =============================================================================
        # FONCTIONS CALLBACK - TOUTES CORRIGÉES AVEC gr.update()
        # =============================================================================

        def clear_cache():
            """Vide le cache OCR."""
            count = clear_ocr_cache()
            if count > 0:
                return gr.update(value=f"Cache vidé : {count} fichier(s) supprimé(s)")
            else:
                return gr.update(value="Cache déjà vide")

        def on_provider_change(provider):
            """Gère le changement de fournisseur."""
            ollama_visible = provider == "Ollama distant"
            runpod_visible = provider == "RunPod.io"
            test_btn_visible = provider != "Ollama local"
            
            # Réinitialiser les champs selon le fournisseur
            ollama_url_value = "http://localhost:11434" if ollama_visible else ""
            runpod_endpoint_value = "" if runpod_visible else ""
            runpod_token_value = "" if runpod_visible else ""
            
            status_message = ""
            if provider == "Ollama local":
                status_message = "✅ Utilisation d'Ollama local sur http://localhost:11434"
            elif provider == "Ollama distant":
                status_message = "⚙️ Configurez l'URL de votre serveur Ollama distant dans le champ ci-dessous"
            elif provider == "RunPod.io":
                status_message = "⚙️ Configurez votre endpoint et token RunPod dans les champs ci-dessous"
            
            return (
                gr.update(visible=ollama_visible, value=ollama_url_value, interactive=True),
                gr.update(visible=runpod_visible, value=runpod_endpoint_value, interactive=True),
                gr.update(visible=runpod_visible, value=runpod_token_value, interactive=True),
                gr.update(visible=test_btn_visible),
                gr.update(visible=True, value=status_message)
            )

        def on_test_connection(provider, ollama_url_val, runpod_endpoint, runpod_token):
            """Test la connexion au fournisseur sélectionné."""
            result = test_connection(provider, ollama_url_val, runpod_endpoint, runpod_token)
            return gr.update(value=result)

        def auto_refresh_on_url_change(provider, ollama_url_val):
            """Actualise automatiquement les modèles quand l'URL Ollama change."""
            if provider == "Ollama distant" and ollama_url_val.strip():
                return refresh_models(provider, ollama_url_val)
            return gr.update(), ""

        def launch_processing_only(file_path, nettoyer, anonymiser, force_processing):
            """Lance uniquement le traitement du fichier."""
            try:
                message, stats, preview, file_type, anon_report = process_file_to_text(file_path, nettoyer, anonymiser, force_processing)
                return (
                    message,     # analysis_box (message de statut)
                    "",          # analysis_alt_box  
                    "",          # compare_box
                    stats,       # text_stats
                    preview,     # preview_box
                    anon_report, # anonymization_report_box
                    preview,     # current_text (state)
                    file_path if file_path else ""  # current_file_path (state)
                )
            except Exception as e:
                traceback.print_exc()
                return f"❌ Erreur traitement : {str(e)}", "", "", "Erreur", "", "", "", ""

        def launch_analysis_only(text_content, file_path, modele, profil, max_tokens_out, 
                                prompt_text, mode_analysis, comparer, nettoyer, anonymiser, 
                                provider, ollama_url_val, runpod_endpoint, runpod_token):
            """Lance uniquement l'analyse."""
            try:
                if not text_content and not file_path:
                    return "❌ Aucun texte ni fichier disponible.", "", "", "", "", "", "", {}
                
                source_type = "UNKNOWN"
                anon_report = ""
                if not text_content and file_path:
                    message, stats, preview, file_type, anon_report = process_file_to_text(file_path, nettoyer, anonymiser, False)
                    if "❌" in message:
                        return message, "", "", stats, preview, anon_report, preview, {}
                    text_content = preview
                    source_type = file_type
                
                use_runpod = provider == "RunPod.io"
                ollama_url = ollama_url_val if provider == "Ollama distant" else "http://localhost:11434"
                
                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    text_content, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer, 
                    source_type, anonymiser, use_runpod, runpod_endpoint, runpod_token, ollama_url
                )
                
                metadata['nettoyer'] = nettoyer
                metadata['anonymiser'] = anonymiser
                
                stats = calculate_text_stats(text_content)
                
                return (
                    analyse,      # analysis_box
                    analyse_alt,  # analysis_alt_box  
                    qc_compare,   # compare_box
                    stats,        # text_stats
                    text_content, # preview_box
                    anon_report,  # anonymization_report_box
                    text_content, # current_text (state)
                    metadata      # analysis_metadata (state)
                )
                
            except Exception as e:
                traceback.print_exc()
                return f"❌ Erreur analyse : {str(e)}", "", "", "Erreur", "", "", "", {}

        def launch_full_pipeline(file_path, nettoyer, anonymiser, force_processing, modele, profil, max_tokens_out, 
                               prompt_text, mode_analysis, comparer, provider, ollama_url_val, runpod_endpoint, runpod_token):
            """Lance le pipeline complet."""
            try:
                if not file_path:
                    return "❌ Aucun fichier fourni.", "", "", "", "", "", "", {}
                
                message, stats, text_content, file_type, anon_report = process_file_to_text(file_path, nettoyer, anonymiser, force_processing)
                if "❌" in message:
                    return message, "", "", stats, text_content, anon_report, text_content, {}
                
                use_runpod = provider == "RunPod.io"
                ollama_url = ollama_url_val if provider == "Ollama distant" else "http://localhost:11434"
                
                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    text_content, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer, 
                    file_type, anonymiser, use_runpod, runpod_endpoint, runpod_token, ollama_url
                )
                
                metadata['nettoyer'] = nettoyer
                metadata['anonymiser'] = anonymiser
                
                return (
                    analyse,      # analysis_box
                    analyse_alt,  # analysis_alt_box
                    qc_compare,   # compare_box
                    stats,        # text_stats
                    text_content, # preview_box
                    anon_report,  # anonymization_report_box
                    text_content, # current_text (state)
                    metadata      # analysis_metadata (state)
                )
                
            except Exception as e:
                traceback.print_exc()
                return f"❌ Erreur pipeline : {str(e)}", "", "", "Erreur", "", "", "", {}

        def on_file_upload(file_path, nettoyer, anonymiser):
            """Gère l'upload de fichier avec préchargement du cache."""
            if not file_path:
                return "", "", "", "", ""
            
            try:
                file_type = get_file_type(file_path)
                
                if file_type == "PDF" and not anonymiser:
                    pdf_hash = get_pdf_hash(file_path)
                    if pdf_hash:
                        ocr_data = load_ocr_cache(pdf_hash, nettoyer)
                        if ocr_data:
                            preview = ocr_data['preview']
                            stats = ocr_data['stats']
                            return stats, preview, "", preview, file_path
                elif file_type == "TXT":
                    content, read_message = read_text_file(file_path)
                    if content:
                        anon_report = ""
                        if nettoyer:
                            content = smart_clean(content)
                        if anonymiser:
                            content, anon_report = anonymize_text(content)
                        stats = calculate_text_stats(content)
                        return stats, content, anon_report, content, file_path
                
                return "", "", "", "", file_path
            except:
                return "", "", "", "", file_path

        def on_select(name, store):
            """Gère la sélection d'un prompt."""
            try:
                if name not in store:
                    name = DEFAULT_PROMPT_NAME
                text = store.get(name, DEFAULT_PROMPT_TEXT)
                return gr.update(value=text)
            except Exception as e:
                return gr.update(value=DEFAULT_PROMPT_TEXT)

        # =============================================================================
        # CONNEXIONS DES ÉVÉNEMENTS
        # =============================================================================

        # Gestion du changement de fournisseur
        provider_choice.change(
            fn=on_provider_change,
            inputs=[provider_choice],
            outputs=[ollama_url, runpod_endpoint, runpod_token, test_connection_btn, connection_status]
        )

        # Test de connexion
        test_connection_btn.click(
            fn=on_test_connection,
            inputs=[provider_choice, ollama_url, runpod_endpoint, runpod_token],
            outputs=[connection_status]
        )

        # Auto-actualisation quand l'URL Ollama change
        ollama_url.change(
            fn=auto_refresh_on_url_change,
            inputs=[provider_choice, ollama_url],
            outputs=[modele, cache_info]
        )

        # Upload de fichier
        input_file.change(
            fn=on_file_upload,
            inputs=[input_file, nettoyer, anonymiser],
            outputs=[text_stats, preview_box, anonymization_report_box, current_text, current_file_path]
        )
        
        # Sélection de prompt
        prompt_selector.change(
            fn=on_select,
            inputs=[prompt_selector, gr.State(value=store)],
            outputs=[prompt_box]
        )

        # Actualisation des modèles
        refresh_models_btn.click(
            fn=refresh_models,
            inputs=[provider_choice, ollama_url],
            outputs=[modele, cache_info]
        )

        # Traitement seul
        process_btn.click(
            fn=launch_processing_only,
            inputs=[input_file, nettoyer, anonymiser, force_processing],
            outputs=[analysis_box, analysis_alt_box, compare_box, text_stats, preview_box, anonymization_report_box, current_text, current_file_path]
        )

        # Analyse seule
        analyze_btn.click(
            fn=launch_analysis_only,
            inputs=[current_text, current_file_path, modele, profil, max_tokens_out, 
                   prompt_box, mode_analysis, comparer, nettoyer, anonymiser,
                   provider_choice, ollama_url, runpod_endpoint, runpod_token],
            outputs=[analysis_box, analysis_alt_box, compare_box, text_stats, preview_box, anonymization_report_box, current_text, analysis_metadata]
        )

        # Pipeline complet
        full_btn.click(
            fn=launch_full_pipeline,
            inputs=[input_file, nettoyer, anonymiser, force_processing, modele, profil, max_tokens_out, 
                   prompt_box, mode_analysis, comparer, provider_choice, ollama_url, runpod_endpoint, runpod_token],
            outputs=[analysis_box, analysis_alt_box, compare_box, text_stats, preview_box, anonymization_report_box, current_text, analysis_metadata]
        )

        # Nettoyage du cache
        clear_cache_btn.click(
            fn=clear_cache,
            outputs=[cache_info]
        )

        # Documentation intégrée
        gr.Markdown("""
        ### Guide d'utilisation

        **Flux de travail recommandé :**
        1. **Configurer le fournisseur** - Choisir entre Ollama local, distant ou RunPod
        2. **Uploader un fichier** (PDF ou TXT) - Le cache se charge automatiquement si disponible
        3. **Actualiser les modèles** - Charger la liste des modèles disponibles
        4. **Traiter fichier** - Traite le PDF (OCR) ou lit le TXT directement
        5. **Analyser** - Lance l'analyse juridique sur le texte disponible

        **Fournisseurs de modèles :**
        - **Ollama local** : `http://localhost:11434` (nécessite `ollama serve`)
        - **Ollama distant** : URL personnalisée (ex: `http://192.168.1.100:11434`)
        - **RunPod.io** : Modèles cloud via API OpenAI-compatible

        **Configuration selon fournisseur :**
        - **Ollama local** : Aucune configuration requise
        - **Ollama distant** : URL du serveur Ollama
        - **RunPod** : Endpoint et token d'authentification

        **Types de fichiers supportés :**
        - **PDF** : Traitement OCR avec cache intelligent
        - **TXT** : Lecture directe avec nettoyage optionnel

        **Options de traitement :**
        - **Nettoyage avancé** : Supprime les artefacts OCR, numéros de page, etc.
        - **Anonymisation automatique** : Remplace noms, prénoms, sociétés, adresses par des références uniques

        **Anonymisation :**
        - Détecte et remplace automatiquement les données personnelles
        - Génère des références uniques cohérentes ([Personne-1], [Société-2], etc.)
        - Produit un rapport détaillé des remplacements effectués
        - Compatible avec les fichiers PDF et TXT

        **Modèles recommandés :**
        - **Ollama local/distant** : mistral:7b-instruct, llama3:latest, deepseek-coder
        - **RunPod** : meta-llama/Llama-3.1-70B-Instruct, mistralai/Mistral-7B-Instruct

        **Profils d'inférence :**
        - **Rapide** : Documents < 15 pages (8k contexte)
        - **Confort** : Documents 15-30 pages (16k contexte)  
        - **Maxi** : Documents 30+ pages (32k contexte)

        **Répertoires** : Cache `./cache_ocr/` | Prompts `./prompts/`
        
        **Note** : Le cache OCR n'est pas utilisé quand l'anonymisation est activée pour éviter les conflits.
        """)

    return demo
