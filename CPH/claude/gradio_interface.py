#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface utilisateur Gradio pour OCR Juridique v7.7-CHUNKS-EMBEDDED
Version: 7.7-CHUNKS-EMBEDDED - Avec analyse par chunks intégrée directement
Date: 2025-09-11
Nouveautés v7.7: Analyse par chunks + synthèse juridique structurée (sans module externe)
"""

import os
import json
import re
import time
import traceback
import threading
import gradio as gr
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from config import PROMPT_STORE_PATH, calculate_text_stats
from file_processing import get_file_type, read_text_file, smart_clean
from anonymization import anonymize_text
from cache_manager import get_pdf_hash, load_ocr_cache, clear_ocr_cache
from ai_providers import get_ollama_models, refresh_models, test_connection
from prompt_manager import load_prompt_store
from processing_pipeline import process_file_to_text, do_analysis_only

# =============================================================================
# CLASSE CHUNK ANALYZER INTÉGRÉE
# =============================================================================

class ChunkAnalyzer:
    """Analyseur de documents longs par chunks avec synthèse juridique."""
    
    def __init__(self, chunk_size: int = 8000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.juridical_markers = [
            "article", "alinéa", "paragraphe", "considérant", "attendu",
            "par ces motifs", "dispositif", "en conséquence", "statuant",
            "vu", "considérant que", "attendu que", "il s'ensuit",
            "partant", "dès lors", "en effet", "cependant", "toutefois",
            "néanmoins", "or", "mais", "donc", "ainsi", "par ailleurs"
        ]
    
    def smart_chunk_text(self, text: str, preserve_structure: bool = True) -> List[Dict]:
        """Découpe intelligemment le texte en chunks."""
        print(f"CHUNK_ANALYZER: Découpage de {len(text)} caractères")
        
        if len(text) <= self.chunk_size:
            return [{
                'id': 1,
                'text': text,
                'start_pos': 0,
                'end_pos': len(text),
                'size': len(text),
                'overlap_prev': 0,
                'overlap_next': 0,
                'contains_juridical': self._detect_juridical_content(text)
            }]
        
        chunks = []
        current_pos = 0
        chunk_id = 1
        
        while current_pos < len(text):
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            if end_pos < len(text) and preserve_structure:
                end_pos = self._find_smart_break(text, current_pos, end_pos)
            
            chunk_text = text[current_pos:end_pos]
            
            # Chevauchements
            overlap_prev = 0
            if chunk_id > 1 and current_pos > 0:
                overlap_start = max(0, current_pos - self.overlap)
                overlap_text = text[overlap_start:current_pos]
                chunk_text = overlap_text + chunk_text
                overlap_prev = len(overlap_text)
            
            overlap_next = 0
            if end_pos < len(text):
                overlap_end = min(len(text), end_pos + self.overlap)
                next_overlap = text[end_pos:overlap_end]
                chunk_text += next_overlap
                overlap_next = len(next_overlap)
            
            chunk_info = {
                'id': chunk_id,
                'text': chunk_text,
                'start_pos': current_pos,
                'end_pos': end_pos,
                'size': len(chunk_text),
                'overlap_prev': overlap_prev,
                'overlap_next': overlap_next,
                'contains_juridical': self._detect_juridical_content(chunk_text)
            }
            
            chunks.append(chunk_info)
            current_pos = end_pos
            chunk_id += 1
        
        print(f"CHUNK_ANALYZER: {len(chunks)} chunks créés")
        return chunks
    
    def _find_smart_break(self, text: str, start: int, proposed_end: int) -> int:
        """Trouve un point de coupe intelligent."""
        search_start = max(start + self.chunk_size - 500, start)
        search_text = text[search_start:proposed_end + 200]
        
        break_patterns = [
            r'\n\n(?=[A-Z])',
            r'\n(?=Article|Considérant|Attendu|Par ces motifs)',
            r'\.[\s\n]+(?=[A-Z])',
            r'\n(?=[0-9]+[\.\)])',
            r'[\.!?][\s\n]+',
            r'\n',
        ]
        
        for pattern in break_patterns:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                best_match = matches[-1]
                return search_start + best_match.end()
        
        return proposed_end
    
    def _detect_juridical_content(self, text: str) -> Dict:
        """Détecte le type de contenu juridique."""
        text_lower = text.lower()
        
        return {
            'has_articles': bool(re.search(r'\barticle\s+\d+', text_lower)),
            'has_considerations': any(marker in text_lower for marker in ['considérant', 'attendu']),
            'has_dispositif': 'par ces motifs' in text_lower or 'dispositif' in text_lower,
            'has_references': bool(re.search(r'(art\.|article)\s*\d+', text_lower)),
            'has_dates': bool(re.search(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', text)),
            'has_legal_terms': sum(1 for marker in self.juridical_markers if marker in text_lower)
        }
    
    def analyze_chunks_with_ai(self, chunks: List[Dict], prompt: str, ai_function, **ai_params) -> List[Dict]:
        """Analyse chaque chunk avec l'IA."""
        print(f"CHUNK_ANALYZER: Analyse de {len(chunks)} chunks avec l'IA")
        
        analyzed_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"CHUNK_ANALYZER: Analyse du chunk {i}/{len(chunks)}")
            
            chunk_prompt = f"""ANALYSE PAR CHUNK - PARTIE {i}/{len(chunks)}

CONTEXTE: Vous analysez la partie {i} d'un document juridique long découpé en {len(chunks)} segments.

CONTENU DU CHUNK {i}:
Position: caractères {chunk['start_pos']} à {chunk['end_pos']}
Taille: {chunk['size']} caractères
Contenu juridique détecté: {chunk['contains_juridical']}

INSTRUCTIONS D'ANALYSE:
{prompt}

CONSIGNES SPÉCIALES POUR LES CHUNKS:
1. Analysez UNIQUEMENT le contenu de ce chunk
2. Identifiez les éléments juridiques clés de cette section
3. Notez si des éléments semblent incomplets (début/fin de phrase coupée)
4. Structurez votre réponse pour faciliter la synthèse finale

TEXTE À ANALYSER:
{chunk['text']}"""

            try:
                analysis = ai_function(
                    text=chunk['text'],
                    prompt=chunk_prompt,
                    **ai_params
                )
                
                chunk_result = chunk.copy()
                chunk_result['analysis'] = analysis
                chunk_result['analysis_timestamp'] = datetime.now().isoformat()
                chunk_result['analysis_success'] = True
                
                analyzed_chunks.append(chunk_result)
                time.sleep(1)
                
            except Exception as e:
                print(f"CHUNK_ANALYZER: Erreur analyse chunk {i}: {e}")
                chunk_result = chunk.copy()
                chunk_result['analysis'] = f"ERREUR: {str(e)}"
                chunk_result['analysis_success'] = False
                analyzed_chunks.append(chunk_result)
        
        return analyzed_chunks
    
    def synthesize_analyses(self, analyzed_chunks: List[Dict], synthesis_prompt: str, ai_function, **ai_params) -> str:
        """Synthétise les analyses de tous les chunks."""
        print("CHUNK_ANALYZER: Synthèse des analyses")
        
        synthesis_content = []
        
        for chunk in analyzed_chunks:
            if chunk['analysis_success']:
                synthesis_content.append(f"""
=== ANALYSE CHUNK {chunk['id']} ===
Position: {chunk['start_pos']}-{chunk['end_pos']} ({chunk['size']} caractères)
Contenu juridique: {chunk['contains_juridical']}

ANALYSE:
{chunk['analysis']}
""")
            else:
                synthesis_content.append(f"""
=== CHUNK {chunk['id']} - ERREUR ===
Position: {chunk['start_pos']}-{chunk['end_pos']}
Erreur: {chunk['analysis']}
""")
        
        full_synthesis_prompt = f"""SYNTHÈSE JURIDIQUE STRUCTURÉE

MISSION: Créez une synthèse juridique structurée à partir des analyses de {len(analyzed_chunks)} chunks d'un document juridique long.

INSTRUCTIONS DE SYNTHÈSE:
{synthesis_prompt}

STRUCTURE ATTENDUE:
1. RÉSUMÉ EXÉCUTIF
2. ÉLÉMENTS JURIDIQUES PRINCIPAUX
3. CHRONOLOGIE/PROCÉDURE
4. POINTS CLÉS PAR THÉMATIQUE
5. CONCLUSIONS ET RECOMMANDATIONS

ANALYSES À SYNTHÉTISER:
{''.join(synthesis_content)}

CONSIGNES:
- Consolidez les informations des différents chunks
- Éliminez les redondances
- Structurez de manière logique et juridique
- Identifiez les liens entre les sections
- Proposez une analyse globale cohérente"""

        try:
            synthesis = ai_function(
                text="SYNTHÈSE DEMANDÉE",
                prompt=full_synthesis_prompt,
                **ai_params
            )
            return synthesis
        except Exception as e:
            return f"ERREUR lors de la synthèse: {str(e)}"
    
    def get_analysis_report(self, chunks: List[Dict], analyzed_chunks: List[Dict]) -> str:
        """Génère un rapport détaillé sur l'analyse par chunks."""
        total_chars = sum(chunk['size'] for chunk in chunks)
        successful_analyses = sum(1 for chunk in analyzed_chunks if chunk['analysis_success'])
        
        report = f"""
=== RAPPORT D'ANALYSE PAR CHUNKS ===

STATISTIQUES GÉNÉRALES:
- Document original: {total_chars:,} caractères
- Nombre de chunks: {len(chunks)}
- Analyses réussies: {successful_analyses}/{len(analyzed_chunks)}
- Taille moyenne par chunk: {total_chars // len(chunks):,} caractères

DÉTAIL DES CHUNKS:
"""
        
        for chunk in analyzed_chunks:
            status = "✅ SUCCÈS" if chunk['analysis_success'] else "❌ ÉCHEC"
            juridical_info = chunk['contains_juridical']
            juridical_score = sum(juridical_info.values()) if isinstance(juridical_info, dict) else 0
            
            report += f"""
Chunk {chunk['id']:2d}: {status}
  - Position: {chunk['start_pos']:,} à {chunk['end_pos']:,} ({chunk['size']:,} caractères)
  - Contenu juridique: {juridical_score} éléments détectés
  - Chevauchements: {chunk['overlap_prev']} | {chunk['overlap_next']}
"""
        
        return report

# =============================================================================
# GESTIONNAIRE DE CONFIGURATION
# =============================================================================

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ollama_config.json")

def load_ollama_config():
    """Charge la configuration Ollama sauvegardée."""
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
                url = config.get('last_ollama_url', 'http://localhost:11434')
                return url
        else:
            return 'http://localhost:11434'
    except Exception as e:
        print(f"CONFIG LOAD ERROR: {e}")
        return 'http://localhost:11434'

def save_ollama_config(ollama_url):
    """Sauvegarde la configuration Ollama."""
    try:
        config_dir = os.path.dirname(CONFIG_FILE_PATH)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        config = {'last_ollama_url': ollama_url}
        
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"CONFIG SAVE ERROR: {e}")
        return False

def initialize_models_list():
    """Initialise la liste des modèles au démarrage."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        else:
            return ["mistral:latest", "llama2:latest", "deepseek-coder:latest"]
            
    except Exception as e:
        return ["mistral:latest", "llama2:latest", "deepseek-coder:latest"]

def build_ui():
    """Construit l'interface utilisateur Gradio avec analyse par chunks intégrée."""
    print("DEBUG: build_ui() appelée")
    print("VERSION: Interface OCR Juridique v7.7-CHUNKS-EMBEDDED")
    
    # Charger la configuration
    saved_ollama_url = load_ollama_config()
    models_list = initialize_models_list()
    store = load_prompt_store()
    
    # Initialiser l'analyseur de chunks
    chunk_analyzer = ChunkAnalyzer()
    
    # UNIQUEMENT VOS PROMPTS
    user_prompt_names = sorted(store.keys()) if store else []
    
    if not user_prompt_names:
        user_prompt_names = ["Aucun prompt trouvé"]
        default_prompt_content = f"AUCUN PROMPT UTILISATEUR TROUVÉ dans {PROMPT_STORE_PATH}\n\nVeuillez ajouter vos prompts personnels."
        selected_prompt = user_prompt_names[0]
    else:
        selected_prompt = user_prompt_names[0] 
        default_prompt_content = store.get(selected_prompt, "")
    
    script_name = os.path.basename(__file__) if '__file__' in globals() else "ocr_legal_tool.py"

    # =============================================================================
    # FONCTIONS CALLBACK AVEC SUPPORT CHUNKS
    # =============================================================================
    
    def ai_call_wrapper(text, prompt, modele, temperature, top_p, max_tokens_out, 
                       provider, ollama_url_val, runpod_endpoint, runpod_token):
        """Wrapper pour les appels IA utilisé par l'analyseur de chunks."""
        try:
            use_runpod = provider == "RunPod.io"
            
            if use_runpod:
                from ai_providers import call_runpod_api
                response = call_runpod_api(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text}
                    ],
                    model=modele,
                    max_tokens=max_tokens_out,
                    temperature=temperature,
                    top_p=top_p,
                    endpoint=runpod_endpoint,
                    token=runpod_token
                )
                return response
            else:
                import requests
                ollama_url = ollama_url_val if provider == "Ollama distant" else "http://localhost:11434"
                
                payload = {
                    "model": modele,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text}
                    ],
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens_out
                    },
                    "stream": False
                }
                
                response = requests.post(
                    f"{ollama_url}/api/chat",
                    json=payload,
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")
                else:
                    return f"ERREUR API Ollama: {response.status_code} - {response.text}"
        except Exception as e:
            return f"ERREUR: {str(e)}"

    def analyze_with_chunks_fn(text1, text2, file_path1, file_path2, modele, profil, max_tokens_out,
                              prompt_text, mode_analysis, temperature, top_p, nettoyer, anonymiser, 
                              processing_mode, provider, ollama_url_val, runpod_endpoint, runpod_token,
                              enable_chunks, chunk_size, chunk_overlap, synthesis_prompt):
        """Analyse avec support des chunks pour documents longs."""
        
        print("ANALYSE AVEC CHUNKS v7.7-EMBEDDED")
        
        # S'assurer qu'on a les textes
        if not text1 and file_path1:
            try:
                message, stats, text1, file_type, anon_report = process_file_to_text(
                    file_path1, nettoyer, anonymiser, False
                )
                if "⛔" in message:
                    text1 = ""
            except:
                text1 = ""
        
        if not text2 and file_path2:
            try:
                message, stats, text2, file_type, anon_report = process_file_to_text(
                    file_path2, nettoyer, anonymiser, False
                )
                if "⛔" in message:
                    text2 = ""
            except:
                text2 = ""
        
        if not text1 and not text2:
            return ("Aucun texte disponible pour l'analyse", "", "", "", "", "", "", "", "", "", "", "", "")
        
        # Préparer le texte complet
        if text1 and text2:
            full_text = f"=== DOCUMENT 1 ===\n{text1}\n\n=== DOCUMENT 2 ===\n{text2}"
        elif text1:
            full_text = text1
        else:
            full_text = text2
        
        # Nettoyer le prompt
        system_prompt = prompt_text.strip()
        if not system_prompt:
            error_msg = "ERREUR: PROMPT VIDE!"
            return (error_msg, "", "", "", "", "", "", "", "", "", "", "", error_msg)
        
        # Décider si utiliser les chunks
        should_use_chunks = enable_chunks and len(full_text) > chunk_size
        
        current_time = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        
        try:
            if should_use_chunks:
                print(f"ANALYSE PAR CHUNKS ACTIVÉE - {len(full_text)} caractères")
                
                # Configurer l'analyseur
                chunk_analyzer.chunk_size = chunk_size
                chunk_analyzer.overlap = chunk_overlap
                
                # Découper en chunks
                chunks = chunk_analyzer.smart_chunk_text(full_text, preserve_structure=True)
                
                # Analyser chaque chunk
                analyzed_chunks = chunk_analyzer.analyze_chunks_with_ai(
                    chunks=chunks,
                    prompt=system_prompt,
                    ai_function=ai_call_wrapper,
                    modele=modele,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens_out=max_tokens_out,
                    provider=provider,
                    ollama_url_val=ollama_url_val,
                    runpod_endpoint=runpod_endpoint,
                    runpod_token=runpod_token
                )
                
                # Créer la synthèse
                if synthesis_prompt.strip():
                    synthesis = chunk_analyzer.synthesize_analyses(
                        analyzed_chunks=analyzed_chunks,
                        synthesis_prompt=synthesis_prompt,
                        ai_function=ai_call_wrapper,
                        modele=modele,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens_out=max_tokens_out,
                        provider=provider,
                        ollama_url_val=ollama_url_val,
                        runpod_endpoint=runpod_endpoint,
                        runpod_token=runpod_token
                    )
                else:
                    synthesis = "AUCUN PROMPT DE SYNTHÈSE FOURNI"
                
                # Rapport d'analyse
                analysis_report = chunk_analyzer.get_analysis_report(chunks, analyzed_chunks)
                
                # Construire le résultat final
                entete_chunks = f"""{'=' * 95}
                                    ANALYSE PAR CHUNKS - v7.7-EMBEDDED
{'=' * 95}

HORODATAGE : {current_time}
MODÈLE : {modele}
MODE CHUNKS : ACTIVÉ
TAILLE CHUNK : {chunk_size:,} caractères
CHEVAUCHEMENT : {chunk_overlap} caractères
NOMBRE DE CHUNKS : {len(chunks)}
TEXTE TOTAL : {len(full_text):,} caractères

{'-' * 95}
                                        SYNTHÈSE JURIDIQUE STRUCTURÉE
{'-' * 95}

{synthesis}

{'-' * 95}
                                        ANALYSES DÉTAILLÉES PAR CHUNK
{'-' * 95}
"""
                
                # Ajouter les analyses de chaque chunk
                chunk_details = []
                for chunk in analyzed_chunks:
                    if chunk['analysis_success']:
                        chunk_details.append(f"""
=== CHUNK {chunk['id']} - ANALYSE ===
Position: {chunk['start_pos']:,} à {chunk['end_pos']:,} ({chunk['size']:,} caractères)

{chunk['analysis']}
""")
                
                resultat_final = entete_chunks + "\n".join(chunk_details) + f"\n\n{'-' * 95}\n{analysis_report}"
                
                debug_info = f"""ANALYSE PAR CHUNKS EXÉCUTÉE v7.7-EMBEDDED
{'=' * 80}

CONFIGURATION:
- Chunks activés: {enable_chunks}
- Taille chunk: {chunk_size:,} caractères
- Nombre de chunks: {len(chunks)}
- Synthèse: {'Oui' if synthesis_prompt.strip() else 'Non'}

PROMPT D'ANALYSE:
{system_prompt[:200]}...

RÉSULTATS:
- Analyses réussies: {sum(1 for c in analyzed_chunks if c['analysis_success'])}/{len(analyzed_chunks)}"""
                
            else:
                print("ANALYSE CLASSIQUE (sans chunks)")
                
                # Analyse classique
                analyse = ai_call_wrapper(
                    text=full_text,
                    prompt=system_prompt,
                    modele=modele,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens_out=max_tokens_out,
                    provider=provider,
                    ollama_url_val=ollama_url_val,
                    runpod_endpoint=runpod_endpoint,
                    runpod_token=runpod_token
                )
                
                resultat_final = f"""{'=' * 95}
                                        ANALYSE CLASSIQUE - v7.7-EMBEDDED
{'=' * 95}

HORODATAGE : {current_time}
MODÈLE : {modele}
MODE CHUNKS : DÉSACTIVÉ
TEXTE TOTAL : {len(full_text):,} caractères

{'-' * 95}
                                        ANALYSE DIRECTE
{'-' * 95}

{analyse}
"""
                
                debug_info = f"""ANALYSE CLASSIQUE EXÉCUTÉE v7.7-EMBEDDED
{'=' * 80}

MODE: Analyse directe
TAILLE DOCUMENT: {len(full_text):,} caractères

PROMPT:
{system_prompt[:200]}..."""
            
            # Stats communes
            text1_length = len(text1) if text1 else 0
            text2_length = len(text2) if text2 else 0
            stats1 = f"{text1_length:,} caractères" if text1 else "Aucun texte"
            stats2 = f"{text2_length:,} caractères" if text2 else "Aucun texte"
            
            return (
                resultat_final, stats1, text1 or "", "", stats2, text2 or "", "",
                text1 or "", text2 or "", file_path1 or "", file_path2 or "", debug_info, ""
            )
            
        except Exception as e:
            print(f"ERREUR lors de l'analyse: {e}")
            error_msg = f"ERREUR TECHNIQUE: {str(e)}"
            return (error_msg, "", "", "", "", "", "", "", "", "", "", error_msg, "")

    # Fonctions callback simplifiées
    def on_test_connection_fn(provider, ollama_url_val, runpod_endpoint, runpod_token):
        result = test_connection(provider, ollama_url_val, runpod_endpoint, runpod_token)
        try:
            url_to_use = ollama_url_val if provider == "Ollama distant" and ollama_url_val else "http://localhost:11434"
            import requests
            response = requests.get(f"{url_to_use}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                if models:
                    return (gr.update(choices=models, value=models[0]), gr.update(value=f"{result} | {len(models)} modèles"))
            return gr.update(), gr.update(value=result)
        except Exception as e:
            return gr.update(), gr.update(value=f"{result} | Erreur: {str(e)}")

    def on_provider_change_fn(provider):
        ollama_visible = provider == "Ollama distant"
        runpod_visible = provider == "RunPod.io"
        
        if ollama_visible:
            current_ollama_url = load_ollama_config()
        else:
            current_ollama_url = ""
        
        status_message = ""
        if provider == "Ollama local":
            status_message = "Utilisation d'Ollama local sur http://localhost:11434"
        elif provider == "Ollama distant":
            status_message = f"URL Ollama distant: {current_ollama_url}"
        elif provider == "RunPod.io":
            status_message = "Configurez votre endpoint et token RunPod"
        
        return (
            gr.update(visible=ollama_visible, value=current_ollama_url if ollama_visible else ""),
            gr.update(visible=runpod_visible, value="" if runpod_visible else ""),
            gr.update(visible=runpod_visible, value="" if runpod_visible else ""),
            gr.update(value=status_message)
        )

    def on_select_prompt_fn(name, store_dict):
        try:
            if name in store_dict:
                text = store_dict.get(name, "")
                return (
                    gr.update(value=text),
                    gr.update(value=f"**PROMPT SÉLECTIONNÉ :** `{name}` ({len(text)} caractères)")
                )
            else:
                return (
                    gr.update(value="Prompt non trouvé."),
                    gr.update(value=f"**ERREUR :** Prompt `{name}` non trouvé !")
                )
        except Exception as e:
            return (
                gr.update(value="Erreur lors du chargement du prompt."),
                gr.update(value=f"**ERREUR :** Exception lors du chargement")
            )

    def process_both_files_fn(file1, file2, nettoyer, anonymiser, force_processing, processing_mode):
        if not file1 and not file2:
            return ("Aucun fichier fourni", "", "", "", "", "", "", "", "", "", "", "", "")
        
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
            combined_status = "\n".join(status_msg) if status_msg else "Aucun fichier traité"
            
            return (
                combined_status, r1[1], r1[2], r1[4], r2[1], r2[2], r2[4],
                r1[2], r2[2], file1 if file1 else "", file2 if file2 else "", "", ""
            )
            
        except Exception as e:
            error_msg = f"Erreur traitement : {str(e)}"
            return (error_msg, "Erreur", "", "", "Erreur", "", "", "", "", "", "", "", "")

    # =============================================================================
    # CONSTRUCTION DE L'INTERFACE GRADIO AVEC CHUNKS INTÉGRÉS
    # =============================================================================
    print("DEBUG: Création de l'interface v7.7-CHUNKS-EMBEDDED...")
    with gr.Blocks(title=f"{script_name} - VOS PROMPTS + CHUNKS v7.7-EMBEDDED") as demo:
        
        gr.Markdown("## OCR + Analyse juridique avec VOS PROMPTS + ANALYSE PAR CHUNKS")
        gr.Markdown("### Version 7.7-CHUNKS-EMBEDDED - Analyse intelligente intégrée")
        gr.Markdown(f"**Vos prompts :** `{PROMPT_STORE_PATH}` | **Nombre de prompts :** {len(user_prompt_names)}")

        # Upload des fichiers
        gr.Markdown("---")
        gr.Markdown("## Upload des fichiers")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**FICHIER 1**")
                input_file1 = gr.File(
                    label="Premier fichier (PDF ou TXT)", 
                    file_types=[".pdf", ".txt", ".text"]
                )
            with gr.Column(scale=1):
                gr.Markdown("**FICHIER 2 (optionnel)**")
                input_file2 = gr.File(
                    label="Deuxième fichier (PDF ou TXT) - optionnel", 
                    file_types=[".pdf", ".txt", ".text"]
                )

        # Configuration commune
        with gr.Row():
            nettoyer = gr.Checkbox(label="Nettoyage avancé", value=True)
            anonymiser = gr.Checkbox(label="Anonymisation automatique", value=False)
            force_processing = gr.Checkbox(label="Forcer nouveau traitement", value=False)

        # Configuration du fournisseur
        gr.Markdown("---")
        gr.Markdown("## Configuration du fournisseur de modèles")
        
        with gr.Row():
            provider_choice = gr.Radio(
                label="Fournisseur", 
                choices=["Ollama local", "Ollama distant", "RunPod.io"], 
                value="Ollama local"
            )
        
        with gr.Row():
            with gr.Column():
                ollama_url = gr.Textbox(
                    label="URL Ollama distant", 
                    value=saved_ollama_url,
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
            test_connection_btn = gr.Button("Tester la connexion", variant="secondary", size="sm")
            connection_status = gr.Markdown("Utilisation d'Ollama local sur http://localhost:11434")

        # Actions principales
        gr.Markdown("---")
        gr.Markdown("## Actions principales")
        with gr.Row():
            process_files_btn = gr.Button("Traiter les fichiers", variant="secondary", size="lg")

        # Interface principale avec onglets
        gr.Markdown("---")
        
        with gr.Tabs():
            # ONGLET 1: RÉSULTATS
            with gr.Tab("RÉSULTATS", elem_id="results_tab"):
                gr.Markdown("## **RÉSULTAT DE VOTRE PROMPT + ANALYSE PAR CHUNKS**")
                
                unified_analysis_box = gr.Textbox(
                    label="Résultat généré par VOTRE prompt (avec chunks si activé)", 
                    lines=45,
                    show_copy_button=True,
                    placeholder="Le résultat de votre prompt personnel apparaîtra ici après analyse...",
                    container=True,
                    show_label=True
                )
                
                # Actions rapides
                with gr.Row():
                    analyze_files_btn = gr.Button("Analyser avec MON PROMPT", variant="primary", size="lg")
                    full_pipeline_btn = gr.Button("TRAITEMENT COMPLET", variant="primary", size="lg")
            
            # ONGLET 2: CONFIGURATION
            with gr.Tab("CONFIGURATION", elem_id="config_tab"):
                gr.Markdown("## Configuration et paramètres")
                
                with gr.Tabs():
                    # VOS PROMPTS
                    with gr.Tab("VOS PROMPTS"):
                        gr.Markdown("### VOS PROMPTS PERSONNELS")

                        with gr.Row():
                            prompt_selector = gr.Dropdown(
                                label="Vos prompts personnels", 
                                choices=user_prompt_names, 
                                value=selected_prompt,
                                info="Sélectionnez le prompt que vous voulez appliquer"
                            )

                        selected_prompt_info = gr.Markdown(
                            f"**PROMPT SÉLECTIONNÉ :** `{selected_prompt}` ({len(default_prompt_content)} caractères)",
                            visible=True
                        )

                        prompt_box = gr.Textbox(
                            label="Contenu de votre prompt sélectionné",
                            value=default_prompt_content,
                            lines=12,
                            interactive=True,
                            info="Ce contenu sera envoyé directement à l'IA",
                            show_copy_button=True
                        )
                    
                    # PARAMÈTRES + CHUNKS
                    with gr.Tab("PARAMÈTRES + CHUNKS"):
                        gr.Markdown("### Configuration du modèle")
                        
                        # Déterminer le modèle par défaut
                        if "mistral:7b-instruct" in models_list:
                            default_model = "mistral:7b-instruct"
                        elif "deepseek-coder:latest" in models_list:
                            default_model = "deepseek-coder:latest"
                        elif "mistral:latest" in models_list:
                            default_model = "mistral:latest"
                        else:
                            default_model = models_list[0] if models_list else "mistral:latest"
                        
                        with gr.Row():
                            modele = gr.Dropdown(
                                label="Modèle IA disponible", 
                                choices=models_list, 
                                value=default_model,
                                info="Sélectionnez le modèle pour l'analyse",
                                interactive=True
                            )
                            refresh_models_btn = gr.Button("Actualiser", variant="secondary", size="sm")
                        
                        gr.Markdown("#### Paramètres de génération")
                        
                        with gr.Row():
                            with gr.Column():
                                profil = gr.Radio(
                                    label="Profil de vitesse", 
                                    choices=["Rapide", "Confort", "Maxi"], 
                                    value="Confort"
                                )
                                mode_analysis = gr.Radio(
                                    label="Mode d'analyse", 
                                    choices=["Standard", "Expert"], 
                                    value="Standard"
                                )
                            with gr.Column():
                                max_tokens_out = gr.Slider(
                                    label="Longueur max de réponse (tokens)", 
                                    minimum=256, 
                                    maximum=8192,
                                    step=256, 
                                    value=4096
                                )
                                temperature = gr.Slider(
                                    label="Créativité (température)", 
                                    minimum=0.0, 
                                    maximum=2.0, 
                                    step=0.1, 
                                    value=0.1
                                )
                                top_p = gr.Slider(
                                    label="Top-p (diversité)", 
                                    minimum=0.0, 
                                    maximum=1.0, 
                                    step=0.05, 
                                    value=0.9
                                )
                                processing_mode = gr.Radio(
                                    label="Mode de traitement", 
                                    choices=["Parallèle", "Séquentiel"], 
                                    value="Parallèle"
                                )
                        
                        # SECTION CHUNKS
                        gr.Markdown("---")
                        gr.Markdown("#### Configuration de l'analyse par chunks")
                        gr.Markdown("**Pour documents longs > 8000 caractères**")
                        
                        enable_chunks = gr.Checkbox(
                            label="Activer l'analyse par chunks", 
                            value=True,
                            info="Découpe automatiquement les longs documents"
                        )
                        
                        with gr.Row():
                            chunk_size = gr.Slider(
                                label="Taille des chunks (caractères)", 
                                minimum=2000, 
                                maximum=15000,
                                step=500, 
                                value=8000
                            )
                            chunk_overlap = gr.Slider(
                                label="Chevauchement entre chunks", 
                                minimum=0, 
                                maximum=1000,
                                step=50, 
                                value=200
                            )
                        
                        synthesis_prompt = gr.Textbox(
                            label="Prompt de synthèse finale (optionnel)",
                            placeholder="Décrivez comment synthétiser les analyses...",
                            lines=4,
                            info="Si fourni, une synthèse finale sera générée"
                        )
                    
                    # DEBUG
                    with gr.Tab("DEBUG"):
                        gr.Markdown("### DEBUG - Informations techniques")
                        
                        debug_prompt_box = gr.Textbox(
                            label="DEBUG - Informations d'exécution", 
                            lines=10, 
                            show_copy_button=True,
                            placeholder="Les informations de debug apparaîtront ici...",
                            interactive=False
                        )
                        
                        chunk_report_box = gr.Textbox(
                            label="RAPPORT CHUNKS", 
                            lines=10, 
                            show_copy_button=True,
                            placeholder="Le rapport des chunks apparaîtra ici...",
                            interactive=False
                        )
                        
                        with gr.Tabs():
                            with gr.Tab("Textes sources"):
                                with gr.Row():
                                    with gr.Column():
                                        text1_stats = gr.Textbox(label="Stats fichier 1", lines=1, interactive=False)
                                        preview1_box = gr.Textbox(label="Texte fichier 1", interactive=False, lines=6)
                                    with gr.Column():
                                        text2_stats = gr.Textbox(label="Stats fichier 2", lines=1, interactive=False)
                                        preview2_box = gr.Textbox(label="Texte fichier 2", interactive=False, lines=6)
                            
                            with gr.Tab("Anonymisation"):
                                with gr.Row():
                                    with gr.Column():
                                        anonymization1_report = gr.Textbox(label="Anonymisation fichier 1", interactive=False, lines=4)
                                    with gr.Column():
                                        anonymization2_report = gr.Textbox(label="Anonymisation fichier 2", interactive=False, lines=4)

        # États
        current_text1 = gr.State(value="")
        current_text2 = gr.State(value="")
        current_file_path1 = gr.State(value="")
        current_file_path2 = gr.State(value="")

        # =============================================================================
        # CONNEXIONS DES ÉVÉNEMENTS
        # =============================================================================

        # Changement de fournisseur
        provider_choice.change(
            fn=on_provider_change_fn,
            inputs=[provider_choice],
            outputs=[ollama_url, runpod_endpoint, runpod_token, connection_status]
        )

        # Test de connexion
        test_connection_btn.click(
            fn=on_test_connection_fn,
            inputs=[provider_choice, ollama_url, runpod_endpoint, runpod_token],
            outputs=[modele, connection_status]
        )

        # Sélection de prompt
        prompt_selector.change(
            fn=on_select_prompt_fn,
            inputs=[prompt_selector, gr.State(value=store)],
            outputs=[prompt_box, selected_prompt_info]
        )

        # Boutons principaux avec support chunks
        analyze_files_btn.click(
            fn=analyze_with_chunks_fn,
            inputs=[current_text1, current_text2, current_file_path1, current_file_path2,
                   modele, profil, max_tokens_out, prompt_box, mode_analysis, temperature, top_p,
                   nettoyer, anonymiser, processing_mode, provider_choice, ollama_url, 
                   runpod_endpoint, runpod_token, enable_chunks, chunk_size, chunk_overlap, synthesis_prompt],
            outputs=[unified_analysis_box, text1_stats, preview1_box,
                    anonymization1_report, text2_stats, preview2_box, anonymization2_report,
                    current_text1, current_text2, current_file_path1, current_file_path2, debug_prompt_box, chunk_report_box]
        )

        full_pipeline_btn.click(
            fn=analyze_with_chunks_fn,
            inputs=[current_text1, current_text2, current_file_path1, current_file_path2,
                   modele, profil, max_tokens_out, prompt_box, mode_analysis, temperature, top_p,
                   nettoyer, anonymiser, processing_mode, provider_choice, ollama_url, 
                   runpod_endpoint, runpod_token, enable_chunks, chunk_size, chunk_overlap, synthesis_prompt],
            outputs=[unified_analysis_box, text1_stats, preview1_box,
                    anonymization1_report, text2_stats, preview2_box, anonymization2_report,
                    current_text1, current_text2, current_file_path1, current_file_path2, debug_prompt_box, chunk_report_box]
        )

        process_files_btn.click(
            fn=process_both_files_fn,
            inputs=[input_file1, input_file2, nettoyer, anonymiser, force_processing, processing_mode],
            outputs=[unified_analysis_box, text1_stats, preview1_box, 
                    anonymization1_report, text2_stats, preview2_box, anonymization2_report,
                    current_text1, current_text2, current_file_path1, current_file_path2, debug_prompt_box, chunk_report_box]
        )

        # Documentation
        gr.Markdown("""
        ---
        ## Version v7.7-CHUNKS-EMBEDDED - Analyse par chunks intégrée
        
        **FONCTIONNALITÉS :**
        - Analyse par chunks pour documents longs
        - Découpage intelligent préservant la structure juridique
        - Synthèse juridique structurée
        - Configuration flexible des paramètres
        - Respect total de vos prompts personnels
        
        **UTILISATION :**
        1. Uploadez vos fichiers
        2. Configurez les paramètres dans "PARAMÈTRES + CHUNKS"
        3. Activez les chunks pour les longs documents
        4. Sélectionnez votre prompt personnel
        5. Lancez l'analyse
        """)
    
    print("DEBUG: Interface créée avec succès")
    return demo

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("Lancement de l'interface Gradio v7.7-CHUNKS-EMBEDDED")
    print("Analyse par chunks intégrée directement dans l'interface")
    demo = build_ui()
    demo.launch()
