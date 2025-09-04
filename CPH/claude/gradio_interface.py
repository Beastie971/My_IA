#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface utilisateur Gradio pour OCR Juridique v7.2-FINAL-CLEAN
Version: 7.2-FINAL-CLEAN - Interface dual files propre, analyse unique, toutes lambda corrigées  
Date: 2025-01-04
Modifications: Analyse comparative unique de deux fichiers, zéro lambda function
"""

import os
import traceback
import threading
import gradio as gr

from config import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT, PROMPT_STORE_PATH, calculate_text_stats
from file_processing import get_file_type, read_text_file, smart_clean
from anonymization import anonymize_text
from cache_manager import get_pdf_hash, load_ocr_cache, clear_ocr_cache
from ai_providers import get_ollama_models, refresh_models, test_connection
from prompt_manager import load_prompt_store
from processing_pipeline import process_file_to_text, do_analysis_only

# =============================================================================
# INTERFACE UTILISATEUR GRADIO PROPRE - ANALYSE COMPARATIVE UNIQUE
# =============================================================================

def build_ui():
    """Construit l'interface utilisateur Gradio pour traiter deux fichiers en parallèle."""
    models_list = get_ollama_models()
    store = load_prompt_store()
    prompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
    
    script_name = os.path.basename(__file__) if '__file__' in globals() else "ocr_legal_tool.py"

    # =============================================================================
    # FONCTIONS CALLBACK DÉFINIES AVANT L'INTERFACE
    # =============================================================================

    def clear_cache_fn():
        """Vide le cache OCR."""
        count = clear_ocr_cache()
        if count > 0:
            return gr.update(value=f"Cache vidé : {count} fichier(s) supprimé(s)")
        else:
            return gr.update(value="Cache déjà vide")

    def on_provider_change_fn(provider):
        """Gère le changement de fournisseur."""
        ollama_visible = provider == "Ollama distant"
        runpod_visible = provider == "RunPod.io"
        test_btn_visible = provider != "Ollama local"
        
        ollama_url_value = "http://localhost:11434" if ollama_visible else ""
        runpod_endpoint_value = "" if runpod_visible else ""
        runpod_token_value = "" if runpod_visible else ""
        
        status_message = ""
        if provider == "Ollama local":
            status_message = "✅ Utilisation d'Ollama local sur http://localhost:11434"
        elif provider == "Ollama distant":
            status_message = "⚙️ Configurez l'URL de votre serveur Ollama distant"
        elif provider == "RunPod.io":
            status_message = "⚙️ Configurez votre endpoint et token RunPod"
        
        return (
            gr.update(visible=ollama_visible, value=ollama_url_value),
            gr.update(visible=runpod_visible, value=runpod_endpoint_value),
            gr.update(visible=runpod_visible, value=runpod_token_value),
            gr.update(visible=test_btn_visible),
            gr.update(value=status_message)
        )

    def on_test_connection_fn(provider, ollama_url_val, runpod_endpoint, runpod_token):
        """Test la connexion."""
        result = test_connection(provider, ollama_url_val, runpod_endpoint, runpod_token)
        return gr.update(value=result)

    def on_file_upload_fn(file_path, file_num, nettoyer, anonymiser):
        """Gère l'upload d'un fichier."""
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
            
            return "", "", "", "", ""
        except:
            return "", "", "", "", ""

    def on_file1_upload_fn(file_path, nettoyer, anonymiser):
        """Gère l'upload du fichier 1."""
        return on_file_upload_fn(file_path, 1, nettoyer, anonymiser)
    
    def on_file2_upload_fn(file_path, nettoyer, anonymiser):
        """Gère l'upload du fichier 2."""
        return on_file_upload_fn(file_path, 2, nettoyer, anonymiser)

    def on_select_prompt_fn(name, store_dict):
        """Gère la sélection d'un prompt."""
        try:
            if name not in store_dict:
                name = DEFAULT_PROMPT_NAME
            text = store_dict.get(name, DEFAULT_PROMPT_TEXT)
            return gr.update(value=text)
        except:
            return gr.update(value=DEFAULT_PROMPT_TEXT)

    def refresh_models_fn(provider, ollama_url_val):
        """Actualise les modèles."""
        return refresh_models(provider, ollama_url_val)

    def auto_refresh_on_url_change_fn(provider, ollama_url_val):
        """Actualise automatiquement les modèles quand l'URL Ollama change."""
        if provider == "Ollama distant" and ollama_url_val.strip():
            return refresh_models(provider, ollama_url_val)
        return gr.update(), ""

    def process_both_files_fn(file1, file2, nettoyer, anonymiser, force_processing, processing_mode):
        """Traite les deux fichiers."""
        if not file1 and not file2:
            return ("❌ Aucun fichier fourni", "", "", "", "", "", "", "", "", "", "")
        
        try:
            results = {}
            
            def process_single_file(file_path, file_key):
                if file_path:
                    message, stats, preview, file_type, anon_report = process_file_to_text(
                        file_path, nettoyer, anonymiser, force_processing
                    )
                    results[file_key] = (message, stats, preview, file_type, anon_report)
                else:
                    results[file_key] = ("Aucun fichier", "", "", "UNKNOWN", "")
            
            if processing_mode == "Parallèle":
                threads = []
                if file1:
                    t1 = threading.Thread(target=process_single_file, args=(file1, 'file1'))
                    threads.append(t1)
                    t1.start()
                if file2:
                    t2 = threading.Thread(target=process_single_file, args=(file2, 'file2'))
                    threads.append(t2)
                    t2.start()
                
                for t in threads:
                    t.join()
            else:
                if file1:
                    process_single_file(file1, 'file1')
                if file2:
                    process_single_file(file2, 'file2')
            
            r1 = results.get('file1', ("", "", "", "UNKNOWN", ""))
            r2 = results.get('file2', ("", "", "", "UNKNOWN", ""))
            
            status_msg = []
            if file1:
                status_msg.append(f"Fichier 1: {r1[0]}")
            if file2:
                status_msg.append(f"Fichier 2: {r2[0]}")
            combined_status = "\n".join(status_msg) if status_msg else "❌ Aucun fichier traité"
            
            return (
                combined_status,  # unified_analysis_box
                r1[1],  # text1_stats
                r1[2],  # preview1_box
                r1[4],  # anonymization1_report
                r2[1],  # text2_stats
                r2[2],  # preview2_box
                r2[4],  # anonymization2_report
                r1[2],  # current_text1
                r2[2],  # current_text2
                file1 if file1 else "",  # current_file_path1
                file2 if file2 else ""   # current_file_path2
            )
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"❌ Erreur traitement : {str(e)}"
            return (error_msg, "Erreur", "", "", "Erreur", "", "", "", "", "", "")

    def analyze_both_files_fn(text1, text2, file_path1, file_path2, modele, profil, max_tokens_out,
                              prompt_text, mode_analysis, nettoyer, anonymiser, processing_mode,
                              provider, ollama_url_val, runpod_endpoint, runpod_token):
        """Analyse unifiée des deux fichiers."""
        
        if not text1 and file_path1:
            message, stats, text1, file_type, anon_report = process_file_to_text(
                file_path1, nettoyer, anonymiser, False
            )
            if "❌" in message:
                text1 = ""
        
        if not text2 and file_path2:
            message, stats, text2, file_type, anon_report = process_file_to_text(
                file_path2, nettoyer, anonymiser, False
            )
            if "❌" in message:
                text2 = ""
        
        if not text1 and not text2:
            return ("❌ Aucun texte disponible pour l'analyse", "", "", "", "", "", "", "", "", "", "")
        
        try:
            use_runpod = provider == "RunPod.io"
            ollama_url = ollama_url_val if provider == "Ollama distant" else "http://localhost:11434"
            
            comparative_prompt = f"""{prompt_text}

INSTRUCTION SPÉCIALE : Tu analyses deux documents juridiques simultanément.

{"DOCUMENT 1 :" if text1 else ""}
{text1 if text1 else ""}

{"DOCUMENT 2 :" if text2 else ""}
{text2 if text2 else ""}

Rédige une analyse juridique UNIQUE qui :
1. Identifie les moyens juridiques communs et divergents
2. Compare les stratégies argumentaires des deux documents
3. Met en évidence les différences dans les demandes et montants
4. Synthétise les enjeux juridiques principaux

Traite les deux documents comme un ensemble cohérent à analyser de manière comparative."""

            if text1 and text2:
                combined_text = f"=== DOCUMENT 1 ===\n{text1}\n\n=== DOCUMENT 2 ===\n{text2}"
                info_prefix = "Analyse comparative de deux documents juridiques"
            elif text1:
                combined_text = text1
                info_prefix = "Analyse du premier document uniquement"
            else:
                combined_text = text2  
                info_prefix = "Analyse du deuxième document uniquement"
            
            print(f"📊 {info_prefix}")
            print(f"Longueur totale du texte combiné : {len(combined_text):,} caractères")
            
            analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                combined_text, modele, profil, max_tokens_out, comparative_prompt, mode_analysis, False,
                "DUAL_DOCUMENTS", anonymiser, use_runpod, runpod_endpoint, runpod_token, ollama_url
            )
            
            document_info = f"""=== ANALYSE COMPARATIVE ===
{info_prefix}
Fichier 1 : {"✅ Traité" if text1 else "❌ Non fourni"}
Fichier 2 : {"✅ Traité" if text2 else "❌ Non fourni"}
Longueur totale analysée : {len(combined_text):,} caractères
========================

"""
            
            unified_analysis = document_info + analyse
            
            stats1 = calculate_text_stats(text1) if text1 else ""
            stats2 = calculate_text_stats(text2) if text2 else ""
            
            return (
                unified_analysis,  # unified_analysis_box
                stats1,  # text1_stats
                text1 or "",  # preview1_box
                "",  # anonymization1_report
                stats2,  # text2_stats
                text2 or "",  # preview2_box
                "",  # anonymization2_report
                text1 or "",  # current_text1
                text2 or "",  # current_text2
                file_path1 or "",  # current_file_path1
                file_path2 or ""   # current_file_path2
            )
            
        except Exception as e:
            traceback.print_exc()
            error_msg = f"❌ Erreur analyse : {str(e)}"
            return (error_msg, "Erreur", "", "", "Erreur", "", "", "", "", "", "")

    def full_pipeline_dual_fn(file1, file2, nettoyer, anonymiser, force_processing, modele, profil,
                              max_tokens_out, prompt_text, mode_analysis, processing_mode,
                              provider, ollama_url_val, runpod_endpoint, runpod_token):
        """Pipeline complet."""
        
        process_results = process_both_files_fn(file1, file2, nettoyer, anonymiser, force_processing, processing_mode)
        
        text1 = process_results[7]  # current_text1
        text2 = process_results[8]  # current_text2
        
        if not text1 and not text2:
            return process_results
        
        analyze_results = analyze_both_files_fn(
            text1, text2, file1, file2, modele, profil, max_tokens_out,
            prompt_text, mode_analysis, nettoyer, anonymiser, processing_mode,
            provider, ollama_url_val, runpod_endpoint, runpod_token
        )
        
        return (
            analyze_results[0],  # unified_analysis_box
            process_results[1],  # text1_stats
            process_results[2],  # preview1_box
            process_results[3],  # anonymization1_report
            process_results[4],  # text2_stats
            process_results[5],  # preview2_box
            process_results[6],  # anonymization2_report
            analyze_results[7],  # current_text1
            analyze_results[8],  # current_text2
            analyze_results[9],  # current_file_path1
            analyze_results[10]  # current_file_path2
        )

    # =============================================================================
    # CONSTRUCTION DE L'INTERFACE GRADIO
    # =============================================================================

    with gr.Blocks(title=f"{script_name} - OCR Juridique DUAL CLEAN") as demo:
        gr.Markdown("## OCR structuré + Analyse juridique (Ollama/RunPod) - **ANALYSE COMPARATIVE UNIQUE**")
        gr.Markdown(f"**Fichier des prompts** : `{PROMPT_STORE_PATH}`")

        # Section Upload des deux fichiers
        gr.Markdown("### Upload des fichiers à analyser ensemble")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**📄 FICHIER 1**")
                input_file1 = gr.File(
                    label="Premier fichier (PDF ou TXT)", 
                    file_types=[".pdf", ".txt", ".text"]
                )
            with gr.Column(scale=1):
                gr.Markdown("**📄 FICHIER 2**")
                input_file2 = gr.File(
                    label="Deuxième fichier (PDF ou TXT)", 
                    file_types=[".pdf", ".txt", ".text"]
                )

        # Configuration commune
        with gr.Row():
            nettoyer = gr.Checkbox(label="Nettoyage avancé", value=True)
            anonymiser = gr.Checkbox(label="Anonymisation automatique", value=False)
            force_processing = gr.Checkbox(label="Forcer nouveau traitement (ignorer cache PDF)", value=False)

        with gr.Row():
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
        
        # Champs de configuration
        with gr.Row():
            with gr.Column():
                ollama_url = gr.Textbox(
                    label="URL Ollama distant", 
                    value="http://localhost:11434",
                    placeholder="ex: http://192.168.1.100:11434",
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
            max_tokens_out = gr.Slider(label="Longueur (tokens)", minimum=256, maximum=4096, step=128, value=4096)
        
        with gr.Row():
            mode_analysis = gr.Radio(label="Mode", choices=["Standard", "Expert"], value="Standard")
            processing_mode = gr.Radio(
                label="Mode de traitement", 
                choices=["Parallèle", "Séquentiel"], 
                value="Parallèle",
                info="Parallèle = plus rapide, Séquentiel = plus stable"
            )

        # Section Prompt COMMUN
        gr.Markdown("### Prompt COMMUN - Analyse unique comparative")

        with gr.Row():
            prompt_selector = gr.Dropdown(label="Choisir un prompt", choices=prompt_names, value=DEFAULT_PROMPT_NAME)

        prompt_box = gr.Textbox(
            label="Contenu du prompt (produit une analyse unique des deux fichiers)",
            value=store.get(DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT),
            lines=12,
            interactive=True
        )

        # Boutons principaux
        with gr.Row():
            process_files_btn = gr.Button("1. Traiter les deux fichiers", variant="secondary", size="lg")
            analyze_files_btn = gr.Button("2. Analyser (analyse comparative unique)", variant="primary", size="lg")
            full_pipeline_btn = gr.Button("Pipeline complet (Traitement + Analyse comparative)", variant="primary")

        # RÉSULTAT UNIQUE - INTERFACE SIMPLIFIÉE
        gr.Markdown("---")
        gr.Markdown("### 🎯 **ANALYSE JURIDIQUE COMPARATIVE UNIQUE**")
        
        unified_analysis_box = gr.Textbox(
            label="📋 Analyse comparative complète des deux documents", 
            lines=40, 
            show_copy_button=True,
            placeholder="L'analyse comparative unique des deux fichiers apparaîtra ici après traitement...",
            container=True,
            show_label=True
        )

        # Onglets secondaires pour informations techniques
        gr.Markdown("---")
        gr.Markdown("### 📁 Informations techniques (optionnel)")
        
        with gr.Tabs():
            with gr.Tab("📄 Textes sources"):
                gr.Markdown("*Textes extraits/nettoyés pour vérification*")
                with gr.Row():
                    with gr.Column(scale=1):
                        text1_stats = gr.Textbox(label="📊 Stats fichier 1", lines=1, interactive=False)
                        preview1_box = gr.Textbox(
                            label="📄 Texte fichier 1", 
                            interactive=False, 
                            show_copy_button=True, 
                            lines=15
                        )
                    with gr.Column(scale=1):
                        text2_stats = gr.Textbox(label="📊 Stats fichier 2", lines=1, interactive=False)
                        preview2_box = gr.Textbox(
                            label="📄 Texte fichier 2", 
                            interactive=False, 
                            show_copy_button=True, 
                            lines=15
                        )
            
            with gr.Tab("🔒 Anonymisation"):
                gr.Markdown("*Rapports d'anonymisation si activée*")
                with gr.Row():
                    with gr.Column(scale=1):
                        anonymization1_report = gr.Textbox(
                            label="🔒 Anonymisation fichier 1", 
                            interactive=False, 
                            show_copy_button=True, 
                            lines=10
                        )
                    with gr.Column(scale=1):
                        anonymization2_report = gr.Textbox(
                            label="🔒 Anonymisation fichier 2", 
                            interactive=False, 
                            show_copy_button=True, 
                            lines=10
                        )

        # États pour stocker les textes
        current_text1 = gr.State(value="")
        current_text2 = gr.State(value="")
        current_file_path1 = gr.State(value="")
        current_file_path2 = gr.State(value="")
        processing_metadata = gr.State(value={})

        # =============================================================================
        # CONNEXIONS DES ÉVÉNEMENTS - TOUTES PROPRES
        # =============================================================================

        # Changement de fournisseur
        provider_choice.change(
            fn=on_provider_change_fn,
            inputs=[provider_choice],
            outputs=[ollama_url, runpod_endpoint, runpod_token, test_connection_btn, connection_status]
        )

        # Test de connexion
        test_connection_btn.click(
            fn=on_test_connection_fn,
            inputs=[provider_choice, ollama_url, runpod_endpoint, runpod_token],
            outputs=[connection_status]
        )

        # Auto-actualisation URL Ollama
        ollama_url.change(
            fn=auto_refresh_on_url_change_fn,
            inputs=[provider_choice, ollama_url],
            outputs=[modele, cache_info]
        )

        # Upload fichiers
        input_file1.change(
            fn=on_file1_upload_fn,
            inputs=[input_file1, nettoyer, anonymiser],
            outputs=[text1_stats, preview1_box, anonymization1_report, current_text1, current_file_path1]
        )
        
        input_file2.change(
            fn=on_file2_upload_fn,
            inputs=[input_file2, nettoyer, anonymiser],
            outputs=[text2_stats, preview2_box, anonymization2_report, current_text2, current_file_path2]
        )
        
        # Sélection de prompt
        prompt_selector.change(
            fn=on_select_prompt_fn,
            inputs=[prompt_selector, gr.State(value=store)],
            outputs=[prompt_box]
        )

        # Actualisation modèles
        refresh_models_btn.click(
            fn=refresh_models_fn,
            inputs=[provider_choice, ollama_url],
            outputs=[modele, cache_info]
        )

        # Boutons principaux
        process_files_btn.click(
            fn=process_both_files_fn,
            inputs=[input_file1, input_file2, nettoyer, anonymiser, force_processing, processing_mode],
            outputs=[unified_analysis_box, text1_stats, preview1_box, 
                    anonymization1_report, text2_stats, preview2_box, anonymization2_report,
                    current_text1, current_text2, current_file_path1, current_file_path2]
        )

        analyze_files_btn.click(
            fn=analyze_both_files_fn,
            inputs=[current_text1, current_text2, current_file_path1, current_file_path2,
                   modele, profil, max_tokens_out, prompt_box, mode_analysis,
                   nettoyer, anonymiser, processing_mode, provider_choice, ollama_url, 
                   runpod_endpoint, runpod_token],
            outputs=[unified_analysis_box, text1_stats, preview1_box,
                    anonymization1_report, text2_stats, preview2_box, anonymization2_report,
                    current_text1, current_text2, current_file_path1, current_file_path2]
        )

        full_pipeline_btn.click(
            fn=full_pipeline_dual_fn,
            inputs=[input_file1, input_file2, nettoyer, anonymiser, force_processing,
                   modele, profil, max_tokens_out, prompt_box, mode_analysis, processing_mode,
                   provider_choice, ollama_url, runpod_endpoint, runpod_token],
            outputs=[unified_analysis_box, text1_stats, preview1_box,
                    anonymization1_report, text2_stats, preview2_box, anonymization2_report,
                    current_text1, current_text2, current_file_path1, current_file_path2]
        )

        # Cache
        clear_cache_btn.click(
            fn=clear_cache_fn,
            outputs=[cache_info]
        )

        # Documentation finale
        gr.Markdown("""
        ### 🎯 **Interface Dual Files - Version Finale Propre**

        **✅ Version corrigée définitive - Zéro lambda function !**

        **Fonctionnalités :**
        - **Analyse unique** des deux fichiers ensemble
        - **Interface simplifiée** avec une seule zone d'analyse
        - **Traitement parallèle ou séquentiel** 
        - **Compatible avec tous les navigateurs** (y compris Falkon)

        **Utilisation :**
        1. **Uploadez** vos deux fichiers (PDF/TXT)
        2. **Configurez** vos paramètres (modèle, prompt, etc.)
        3. **Cliquez** sur "Pipeline complet" 
        4. **Consultez** l'analyse comparative unique

        **Note technique :** Cette version utilise exclusivement des fonctions nommées 
        définies avant l'interface. Plus aucun problème de lambda functions !
        """)

    return demo
