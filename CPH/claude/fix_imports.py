#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correctif rapide pour résoudre les problèmes d'imports
Version: 1.0-QUICK-FIX
Date: 2025-09-15
"""

import os
import sys

def fix_gradio_interface():
    """Corrige le fichier gradio_interface.py pour résoudre les imports."""
    
    gradio_file = "gradio_interface.py"
    
    if not os.path.exists(gradio_file):
        print(f"❌ Fichier {gradio_file} non trouvé")
        return False
    
    try:
    from config import DEFAULT_PROMPT_TEXT, calculate_text_stats
    MODULES_AVAILABLE['config'] = True
    print("✅ Module config disponible")
except ImportError as e:
    MODULES_AVAILABLE['config'] = False
    print(f"⚠️ Module config non disponible: {e}")

try:
    from async_task_manager import task_manager, start_analysis_async, check_analysis_status, get_analysis_result
    MODULES_AVAILABLE['async'] = True
    print("✅ Module async_task_manager disponible")
except ImportError as e:
    MODULES_AVAILABLE['async'] = False
    print(f"⚠️ Module async_task_manager non disponible: {e}")

# ========================================
# CONFIGURATION DE BASE
# ========================================

CLOUDFLARE_TIMEOUT = 85
MAX_DIRECT_PROCESSING = 70

app_state = {
    "current_provider": "Ollama local",
    "ollama_url": "http://localhost:11434",
    "models_list": ["mistral:7b-instruct", "mixtral:8x7b", "llama3.1:8b"],
    "last_task_id": None
}

# Prompts par défaut
DEFAULT_PROMPTS = {
    "Analyse rapide": """Tu es juriste spécialisé en droit du travail. Analyse rapidement ce document :

1. MOYENS DE DROIT (2-3 lignes max)
   - Fondements juridiques invoqués

2. DEMANDES PRINCIPALES (style télégraphique)  
   - Prétentions chiffrées ou non

3. RECOMMANDATION (1 phrase)

Style: Concis, télégraphique.""",

    "Extraction moyens": """Extrait uniquement les moyens de droit de ce document :
• Articles de loi invoqués
• Conventions collectives citées
• Principes généraux du droit
Format: Liste simple."""
}

# ========================================
# FONCTIONS UTILITAIRES SIMPLIFIÉES
# ========================================

def estimate_processing_time(text_length):
    """Estimation simple du temps de traitement."""
    estimated = text_length / 1000  # 1000 chars/seconde
    
    if estimated > MAX_DIRECT_PROCESSING:
        return estimated, "🚀 Mode ASYNCHRONE recommandé", "⚠️"
    elif estimated > 40:
        return estimated, "⚡ Mode direct (surveiller)", "👀"
    else:
        return estimated, "✅ Mode direct optimal", "✅"

def track_activity():
    """Marque une activité utilisateur."""
    pass  # Placeholder

# ========================================
# FONCTIONS D'ANALYSE SIMPLIFIÉES
# ========================================

def analyze_with_protection(text1, text2, prompt, model, provider, 
                           temperature, top_p, max_tokens, profil,
                           ollama_url, runpod_endpoint, runpod_token, force_async=False):
    """Analyse avec protection timeout basique."""
    
    # Préparation du texte
    if text1 and text2:
        full_text = f"=== DOCUMENT 1 ===\\n{text1}\\n\\n=== DOCUMENT 2 ===\\n{text2}"
        doc_info = f"2 documents - {len(text1):,} + {len(text2):,} caractères"
    elif text1:
        full_text = text1
        doc_info = f"1 document - {len(text1):,} caractères"
    elif text2:
        full_text = text2
        doc_info = f"1 document - {len(text2):,} caractères"
    else:
        return ("ERREUR: Aucun texte fourni", "", "", "Aucun texte", "", False, "")
    
    # Estimation
    estimated_time, strategy, icon = estimate_processing_time(len(full_text))
    
    if not MODULES_AVAILABLE['processing']:
        return (f"⚠️ MODULES NON DISPONIBLES\\n\\nDocument: {doc_info}\\nEstimation: {estimated_time:.1f}s\\nStratégie: {strategy}", 
                "", "", "Modules manquants", "", False, "")
    
    # Vérifier si async nécessaire
    needs_async = estimated_time > MAX_DIRECT_PROCESSING or force_async
    
    if needs_async and MODULES_AVAILABLE['async']:
        # MODE ASYNCHRONE
        def analysis_task():
            try:
                result = do_analysis_only(
                    user_text=full_text,
                    modele=model,
                    profil=profil,
                    max_tokens_out=max_tokens,
                    system_prompt=prompt,
                    mode_analysis="standard",
                    comparer=False,
                    source_type="DOCUMENT",
                    anonymiser=False,
                    use_runpod=(provider == "RunPod.io"),
                    runpod_endpoint=runpod_endpoint,
                    runpod_token=runpod_token,
                    ollama_url=ollama_url
                )
                
                analysis, _, _, metadata = result
                
                formatted_result = f"""{'=' * 80}
            ANALYSE JURIDIQUE ASYNCHRONE - ANTI-TIMEOUT 524
{'=' * 80}

HORODATAGE: {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}
MODÈLE: {model}
FOURNISSEUR: {provider}
MODE: Asynchrone (protection Cloudflare)
DOCUMENT: {doc_info}

{'-' * 80}
                    RÉSULTAT DE L'ANALYSE
{'-' * 80}

{analysis}"""
                
                stats1 = f"{len(text1):,} caractères" if text1 else "Aucun texte"
                stats2 = f"{len(text2):,} caractères" if text2 else "Aucun texte"
                
                return (formatted_result, stats1, stats2, "Mode asynchrone", "Terminé")
                
            except Exception as e:
                return (f"ERREUR ANALYSE ASYNCHRONE: {str(e)}", "", "", str(e), "")
        
        # Lancer en asynchrone
        task_id = start_analysis_async(analysis_task, task_name=f"Analyse ({len(full_text):,} chars)")
        app_state["last_task_id"] = task_id
        
        async_message = f"""🚀 ANALYSE LANCÉE EN MODE ASYNCHRONE

📊 Document: {doc_info}
⏱️ Estimation: {estimated_time:.1f}s (> {MAX_DIRECT_PROCESSING}s)
🆔 Tâche ID: {task_id}

🔄 Instructions:
1. Cliquez sur "🔍 Vérifier statut" pour suivre
2. Cliquez sur "📥 Récupérer résultat" quand terminé"""
        
        return (async_message, "", "", f"Mode asynchrone - Tâche {task_id}", "", True, task_id)
    
    else:
        # MODE DIRECT
        try:
            result = do_analysis_only(
                user_text=full_text,
                modele=model,
                profil=profil,
                max_tokens_out=max_tokens,
                system_prompt=prompt,
                mode_analysis="standard",
                comparer=False,
                source_type="DOCUMENT",
                anonymiser=False,
                use_runpod=(provider == "RunPod.io"),
                runpod_endpoint=runpod_endpoint,
                runpod_token=runpod_token,
                ollama_url=ollama_url
            )
            
            analysis, _, _, metadata = result
            
            formatted_result = f"""{'=' * 80}
            ANALYSE JURIDIQUE DIRECTE - CLOUDFLARE SAFE
{'=' * 80}

HORODATAGE: {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}
MODÈLE: {model}
FOURNISSEUR: {provider}
MODE: Direct (< {MAX_DIRECT_PROCESSING}s)
DOCUMENT: {doc_info}

{'-' * 80}
                    RÉSULTAT DE L'ANALYSE
{'-' * 80}

{analysis}"""
            
            stats1 = f"{len(text1):,} caractères" if text1 else "Aucun texte"
            stats2 = f"{len(text2):,} caractères" if text2 else "Aucun texte"
            
            return (formatted_result, stats1, stats2, "Mode direct", "Terminé", False, "")
            
        except Exception as e:
            return (f"ERREUR ANALYSE: {str(e)}", "", "", str(e), "", False, "")

# ========================================
# INTERFACE GRADIO SIMPLIFIÉE
# ========================================

def build_ui():
    """Interface simplifiée anti-timeout 524."""
    
    print(f"🔧 Construction interface simplifiée...")
    print(f"   - Processing: {'✅' if MODULES_AVAILABLE['processing'] else '❌'}")
    print(f"   - AI Providers: {'✅' if MODULES_AVAILABLE['ai_providers'] else '❌'}")
    print(f"   - Config: {'✅' if MODULES_AVAILABLE['config'] else '❌'}")
    print(f"   - Async: {'✅' if MODULES_AVAILABLE['async'] else '❌'}")
    
    with gr.Blocks(title="OCR Juridique - Anti-Timeout 524") as demo:
        
        gr.Markdown(f"""
        # 🛡️ OCR Juridique - Anti-Timeout 524 (Compatible)
        
        **Protection Cloudflare intégrée** - Version simplifiée compatible
        
        ⏱️ **Timeout sécurisé** : {CLOUDFLARE_TIMEOUT}s max  
        📊 **Estimation** : Temps de traitement calculé  
        🚀 **Mode async** : {'Disponible' if MODULES_AVAILABLE['async'] else 'Non disponible'}
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
                        
                        test_btn = gr.Button("🔍 Tester connexion", variant="primary")
                        status_msg = gr.Textbox(
                            label="Statut",
                            value="Prêt (version compatible)",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Paramètres")
                        
                        model = gr.Dropdown(
                            choices=app_state["models_list"],
                            value="mistral:7b-instruct",
                            label="Modèle IA"
                        )
                        
                        profil = gr.Radio(
                            choices=["Rapide", "Confort"],
                            value="Rapide",
                            label="Profil (Rapide recommandé)"
                        )
                        
                        with gr.Row():
                            temperature = gr.Slider(0, 1, value=0.2, step=0.1, label="Température")
                            max_tokens = gr.Slider(500, 2000, value=1000, step=250, label="Max tokens")
                        
                        top_p = gr.Slider(0, 1, value=0.9, step=0.1, label="Top-p", visible=False)
            
            # ======= DOCUMENTS =======
            with gr.Tab("📄 Documents"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Document 1")
                        file1 = gr.File(label="Fichier 1", file_types=[".pdf", ".txt"])
                        with gr.Row():
                            clean1 = gr.Checkbox(label="Nettoyer", value=True)
                            anon1 = gr.Checkbox(label="Anonymiser", value=False)
                        text1 = gr.Textbox(label="Texte 1", lines=8)
                        stats1 = gr.Textbox(label="Statistiques 1", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### Document 2 (optionnel)")
                        file2 = gr.File(label="Fichier 2", file_types=[".pdf", ".txt"])
                        with gr.Row():
                            clean2 = gr.Checkbox(label="Nettoyer", value=True)
                            anon2 = gr.Checkbox(label="Anonymiser", value=False)
                        text2 = gr.Textbox(label="Texte 2", lines=8)
                        stats2 = gr.Textbox(label="Statistiques 2", interactive=False)
                
                # Estimation
                estimation_display = gr.Textbox(
                    label="📊 Estimation de traitement",
                    value="✅ Aucun texte → Mode direct optimal",
                    interactive=False
                )
                
                clear_btn = gr.Button("🗑️ Nettoyer", variant="stop")
            
            # ======= ANALYSE =======
            with gr.Tab("🔍 Analyse"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Prompts")
                        
                        prompt_selector = gr.Dropdown(
                            choices=list(DEFAULT_PROMPTS.keys()),
                            label="Prompts prédéfinis"
                        )
                        select_prompt_btn = gr.Button("📝 Charger")
                        
                        force_async = gr.Checkbox(
                            label="🚀 Forcer mode asynchrone",
                            value=False,
                            visible=MODULES_AVAILABLE['async']
                        )
                        
                        prompt_status = gr.Textbox(label="Statut", interactive=False)
                    
                    with gr.Column(scale=2):
                        prompt_text = gr.Textbox(
                            label="Prompt d'analyse",
                            lines=10,
                            value=DEFAULT_PROMPTS["Analyse rapide"]
                        )
                        
                        analyze_btn = gr.Button(
                            "🚀 Analyse Intelligente", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        # Contrôles async si disponible
                        if MODULES_AVAILABLE['async']:
                            with gr.Group():
                                gr.Markdown("### 🔄 Gestion asynchrone")
                                current_task_id = gr.Textbox(
                                    label="ID tâche en cours",
                                    interactive=False
                                )
                                
                                with gr.Row():
                                    check_status_btn = gr.Button("🔍 Vérifier", size="sm")
                                    get_result_btn = gr.Button("📥 Récupérer", size="sm")
                        else:
                            current_task_id = gr.Textbox(value="", visible=False)
                            check_status_btn = gr.Button("Non disponible", size="sm", interactive=False)
                            get_result_btn = gr.Button("Non disponible", size="sm", interactive=False)
            
            # ======= RÉSULTATS =======
            with gr.Tab("📊 Résultats"):
                result_text = gr.Textbox(
                    label="Résultat de l'analyse",
                    lines=25,
                    show_copy_button=True
                )
                
                with gr.Row():
                    with gr.Column():
                        debug_info = gr.Textbox(
                            label="Informations", lines=4, interactive=False
                        )
                    with gr.Column():
                        analysis_report = gr.Textbox(
                            label="Rapport", lines=4, interactive=False
                        )
        
        # ========================================
        # ÉVÉNEMENTS
        # ========================================
        
        # Configuration provider
        def on_provider_change(provider):
            ollama_visible = provider == "Ollama distant"
            runpod_visible = provider == "RunPod.io"
            
            if provider == "Ollama local":
                status = "✅ Ollama local (timeout sécurisé)"
                url_value = ""
            elif provider == "Ollama distant":
                url_value = app_state["ollama_url"]
                status = f"🌐 Ollama distant: {url_value}"
            else:
                url_value = ""
                status = "☁️ RunPod (anti-timeout)"
            
            return (
                gr.update(visible=ollama_visible, value=url_value),
                gr.update(visible=runpod_visible, value=""),
                gr.update(visible=runpod_visible, value=""),
                gr.update(value=status)
            )
        
        provider.change(
            on_provider_change,
            inputs=[provider],
            outputs=[ollama_url, runpod_endpoint, runpod_token, status_msg]
        )
        
        # Test connexion
        def test_connection_simple(provider, ollama_url, runpod_endpoint, runpod_token):
            if MODULES_AVAILABLE['ai_providers']:
                try:
                    result = test_connection(provider, ollama_url, runpod_endpoint, runpod_token)
                    return f"✅ {result}"
                except Exception as e:
                    return f"❌ Erreur: {str(e)}"
            else:
                return "⚠️ Module test non disponible"
        
        test_btn.click(
            test_connection_simple,
            inputs=[provider, ollama_url, runpod_endpoint, runpod_token],
            outputs=[status_msg]
        )
        
        # Gestion fichiers
        def handle_file1(file):
            if file and MODULES_AVAILABLE['processing']:
                try:
                    message, stats, text, file_type, anon_report = process_file_to_text(
                        file.name, clean1.value, anon1.value, force_ocr=True
                    )
                    return message, stats, text
                except Exception as e:
                    return f"❌ Erreur: {str(e)}", "Erreur", ""
            return "", "0 caractères", ""
        
        def handle_file2(file):
            if file and MODULES_AVAILABLE['processing']:
                try:
                    message, stats, text, file_type, anon_report = process_file_to_text(
                        file.name, clean2.value, anon2.value, force_ocr=True
                    )
                    return stats, text
                except Exception as e:
                    return f"Erreur: {str(e)}", ""
            return "0 caractères", ""
        
        file1.change(handle_file1, inputs=[file1], outputs=[status_msg, stats1, text1])
        file2.change(handle_file2, inputs=[file2], outputs=[stats2, text2])
        
        # Estimation en temps réel
        def update_estimation(text1, text2):
            total_length = len(text1 or "") + len(text2 or "")
            if total_length == 0:
                return "✅ Aucun texte → Mode direct optimal"
            
            estimated_time, strategy, icon = estimate_processing_time(total_length)
            doc_info = f"{total_length:,} caractères"
            return f"{icon} {doc_info} → {strategy} ({estimated_time:.1f}s)"
        
        for component in [text1, text2]:
            component.change(
                update_estimation,
                inputs=[text1, text2],
                outputs=[estimation_display]
            )
        
        # Prompts
        def select_prompt(prompt_name):
            if prompt_name in DEFAULT_PROMPTS:
                return (
                    gr.update(value=DEFAULT_PROMPTS[prompt_name]),
                    gr.update(value=f"✅ Prompt '{prompt_name}' chargé")
                )
            return (gr.update(), gr.update(value="❌ Prompt non trouvé"))
        
        select_prompt_btn.click(
            select_prompt,
            inputs=[prompt_selector],
            outputs=[prompt_text, prompt_status]
        )
        
        # Analyse principale
        def route_analysis(*args):
            result = analyze_with_protection(*args)
            if len(result) >= 7:
                formatted_result, stats1_r, stats2_r, debug, report, is_async, task_id = result
                return formatted_result, stats1_r, stats2_r, debug, report, task_id if is_async else ""
            else:
                return result[:5] + ("",)
        
        analyze_btn.click(
            route_analysis,
            inputs=[
                text1, text2, prompt_text, model, provider,
                temperature, top_p, max_tokens, profil,
                ollama_url, runpod_endpoint, runpod_token, force_async
            ],
            outputs=[result_text, stats1, stats2, debug_info, analysis_report, current_task_id]
        )
        
        # Gestion async si disponible
        if MODULES_AVAILABLE['async']:
            def check_status(task_id):
                if not task_id:
                    return "❌ Aucune tâche", "", "", "", ""
                try:
                    status = check_analysis_status(task_id)
                    if "error" in status:
                        return f"❌ {status['error']}", "", "", "", ""
                    
                    if status["status"] == "completed":
                        return f"✅ Terminé - Cliquez 'Récupérer'", "", "", "", ""
                    else:
                        progress = int(status["progress"] * 100)
                        return f"🔄 En cours... {progress}%", "", "", "", ""
                except Exception as e:
                    return f"❌ Erreur: {str(e)}", "", "", "", ""
            
            def get_result(task_id):
                if not task_id:
                    return "❌ Aucune tâche", "", "", "", ""
                try:
                    result_data = get_analysis_result(task_id)
                    if "error" in result_data:
                        return f"❌ {result_data['error']}", "", "", "", ""
                    
                    if result_data.get("success"):
                        analysis_result = result_data["result"]
                        if isinstance(analysis_result, tuple) and len(analysis_result) >= 5:
                            formatted_result, stats1, stats2, debug_info, analysis_report = analysis_result
                            return formatted_result, stats1, stats2, debug_info, analysis_report
                        else:
                            return str(analysis_result), "", "", "Résultat récupéré", ""
                    else:
                        return "❌ Résultat non disponible", "", "", "", ""
                except Exception as e:
                    return f"❌ Erreur: {str(e)}", "", "", "", ""
            
            check_status_btn.click(
                check_status,
                inputs=[current_task_id],
                outputs=[result_text, stats1, stats2, debug_info, analysis_report]
            )
            
            get_result_btn.click(
                get_result,
                inputs=[current_task_id],
                outputs=[result_text, stats1, stats2, debug_info, analysis_report]
            )
        
        # Nettoyage
        def clear_all():
            if MODULES_AVAILABLE['async'] and app_state.get("last_task_id"):
                try:
                    task_manager.cancel_task(app_state["last_task_id"])
                except:
                    pass
                app_state["last_task_id"] = None
            
            return (
                "", "", "", "", "", "", 
                DEFAULT_PROMPTS["Analyse rapide"],
                "🧹 Nettoyé", "", ""
            )
        
        clear_btn.click(
            clear_all,
            outputs=[text1, text2, result_text, stats1, stats2, debug_info,
                    prompt_text, status_msg, analysis_report, current_task_id]
        )
        
        # Sauvegarde URL
        def save_url(url):
            app_state["ollama_url"] = url
            return f"URL sauvée: {url}"
        
        ollama_url.change(save_url, inputs=[ollama_url], outputs=[status_msg])
    
    return demo

if __name__ == "__main__":
    print("🧪 Test interface compatible")
    demo = build_ui()
    demo.launch(server_port=7860)
'''
        
        # Écrire le nouveau contenu
        with open(gradio_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Fichier {gradio_file} corrigé avec succès")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction: {e}")
        return False

def main():
    """Point d'entrée principal du correctif."""
    print("🔧 Correctif rapide des imports - OCR Juridique v8.3")
    print("=" * 60)
    
    # Vérifier le répertoire courant
    if not os.path.exists("main_ocr.py"):
        print("❌ Erreur: Vous devez être dans le répertoire du projet OCR Juridique")
        print("   (le fichier main_ocr.py doit être présent)")
        return False
    
    print("📁 Répertoire de projet détecté")
    
    # Corriger gradio_interface.py
    print("\n🔧 Correction de gradio_interface.py...")
    if fix_gradio_interface():
        print("✅ Correction appliquée avec succès")
    else:
        print("❌ Échec de la correction")
        return False
    
    print("\n🧪 Test rapide...")
    try:
        # Test d'import
        sys.path.insert(0, os.getcwd())
        from gradio_interface import build_ui
        print("✅ Import build_ui réussi")
        
        # Test de construction
        app = build_ui()
        if app:
            print("✅ Construction interface réussie")
        else:
            print("⚠️ Interface construite mais pourrait avoir des problèmes")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 CORRECTIF APPLIQUÉ AVEC SUCCÈS !")
    print("=" * 60)
    print()
    print("Vous pouvez maintenant lancer l'application :")
    print("  python main_ocr.py")
    print()
    print("En cas de problème :")
    print("  1. Restaurez la sauvegarde : mv gradio_interface.py.backup gradio_interface.py")
    print("  2. Vérifiez les dépendances : pip install -r requirements.txt")
    print("  3. Créez le fichier async_task_manager.py si manquant")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

        # Lire le fichier actuel
        with open(gradio_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Sauvegarder l'original
        backup_file = f"{gradio_file}.backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Sauvegarde créée: {backup_file}")
        
        # Remplacer par la version compatible
        new_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface utilisateur Gradio pour OCR Juridique - VERSION QUICK-FIX COMPATIBLE
Version: 8.3-QUICK-FIX
Date: 2025-09-15
Fonctionnalités: Compatible avec structure existante + protection timeout basique
"""

import os
import gradio as gr
from datetime import datetime
import time

# ========================================
# IMPORTS SÉCURISÉS
# ========================================

MODULES_AVAILABLE = {}

# Test des imports un par un
try:
    from processing_pipeline import process_file_to_text, do_analysis_only
    MODULES_AVAILABLE['processing'] = True
    print("✅ Module processing_pipeline disponible")
except ImportError as e:
    MODULES_AVAILABLE['processing'] = False
    print(f"⚠️ Module processing_pipeline non disponible: {e}")

try:
    from ai_providers import get_ollama_models, test_connection
    MODULES_AVAILABLE['ai_providers'] = True
    print("✅ Module ai_providers disponible")
except ImportError as e:
    MODULES_AVAILABLE['ai_providers'] = False
    print(f"⚠️ Module ai_providers non disponible: {e}")

try