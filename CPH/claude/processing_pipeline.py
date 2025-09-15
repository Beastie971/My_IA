#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline de traitement pour OCR Juridique v7.2-SYSTEM-PROMPT-FIXED
Version: 7.2-SYSTEM-PROMPT-FIXED - Séparation claire prompt système/utilisateur
Date: 2025-01-04
"""

import os
import time
import traceback
import re
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract

from config import EXPERT_PROMPT_TEXT, DEFAULT_PROMPT_TEXT, calculate_text_stats
from file_processing import get_file_type, read_text_file, smart_clean, _normalize_unicode
from anonymization import anonymize_text
from cache_manager import get_pdf_hash, load_ocr_cache, save_ocr_cache
from ai_providers import generate_with_ollama, generate_with_runpod

# =============================================================================
# PIPELINE DE TRAITEMENT INCHANGÉ
# =============================================================================

def process_file_to_text(file_path, nettoyer, anonymiser, force_ocr=False):
    """Traite un fichier (PDF ou TXT) et retourne le texte extrait."""
    if not file_path:
        return "❌ Aucun fichier fourni.", "", "", "UNKNOWN", ""
    
    if not os.path.exists(file_path):
        return "❌ Fichier introuvable.", "", "", "UNKNOWN", ""
    
    file_type = get_file_type(file_path)
    anonymization_report = ""
    
    try:
        if file_type == "PDF":
            pdf_hash = get_pdf_hash(file_path)
            ocr_data = None
            
            if pdf_hash and not force_ocr and not anonymiser:
                ocr_data = load_ocr_cache(pdf_hash, nettoyer)
            
            if ocr_data and not anonymiser:
                print("✅ Utilisation du cache OCR existant")
                preview = ocr_data['preview']
                stats = ocr_data['stats']
                total_pages = ocr_data.get('total_pages', '?')
                print(f"Cache utilisé : {total_pages} page(s) déjà traitées")
            else:
                print(f"📄 Conversion PDF : {file_path}")
                images = convert_from_path(file_path)
                raw_pages = []
                
                total_pages = len(images)
                print(f"📄 Traitement de {total_pages} page(s)...")
                
                for i, image in enumerate(images):
                    print(f"🔍 OCR page {i+1}/{total_pages}...")
                    page_text = pytesseract.image_to_string(image, lang="fra")
                    raw_pages.append(page_text or "")

                if nettoyer:
                    print("🧹 Nettoyage du texte...")
                    cleaned_pages = [_normalize_unicode(t) for t in raw_pages]
                    preview = smart_clean("\n".join(cleaned_pages), pages_texts=cleaned_pages)
                else:
                    preview = "\n".join(raw_pages).strip()

                if anonymiser:
                    print("🔒 Anonymisation du texte...")
                    preview, anonymization_report = anonymize_text(preview)

                stats = calculate_text_stats(preview)
                
                if pdf_hash and not anonymiser:
                    cache_data = {
                        'preview': preview,
                        'stats': stats,
                        'total_pages': total_pages,
                        'timestamp': str(os.path.getmtime(file_path))
                    }
                    save_ocr_cache(pdf_hash, nettoyer, cache_data)

            if not preview.strip():
                return "❌ Aucun texte détecté lors de l'OCR.", stats, preview, file_type, anonymization_report
            
            return "✅ OCR terminé. Texte prêt pour analyse.", stats, preview, file_type, anonymization_report

        elif file_type == "TXT":
            print(f"📝 Lecture fichier texte : {file_path}")
            content, read_message = read_text_file(file_path)
            
            if not content:
                return f"❌ {read_message}", "", "", file_type, ""
            
            if nettoyer:
                print("🧹 Nettoyage du texte...")
                content = smart_clean(content)
            
            if anonymiser:
                print("🔒 Anonymisation du texte...")
                content, anonymization_report = anonymize_text(content)
            
            stats = calculate_text_stats(content)
            
            message = f"✅ Fichier texte lu ({read_message}). Prêt pour analyse."
            return message, stats, content, file_type, anonymization_report
        
        else:
            return f"❌ Type de fichier non supporté. Extensions acceptées : .pdf, .txt", "", "", file_type, ""

    except Exception as e:
        traceback.print_exc()
        error_msg = f"❌ Erreur lors du traitement : {str(e)}"
        stats = "Erreur - Impossible de calculer les statistiques"
        return error_msg, stats, "", file_type, ""

def do_analysis_only(user_text, modele, profil, max_tokens_out, system_prompt, mode_analysis, 
                    comparer, source_type="UNKNOWN", anonymiser=False, use_runpod=False, 
                    runpod_endpoint="", runpod_token="", ollama_url="http://localhost:11434"):
    """Effectue uniquement l'analyse du texte avec séparation système/utilisateur."""
    if not user_text or not user_text.strip():
        return "❌ Aucun texte disponible pour l'analyse.", "", "", {}
    
    start_time = time.time()
    
    try:
        text_length = len(user_text)
        estimated_tokens = text_length // 4
        
        print(f"📊 Longueur du texte utilisateur : {text_length:,} caractères (≈{estimated_tokens:,} tokens)")
        print(f"📝 Longueur du prompt système : {len(system_prompt):,} caractères")
        
        if estimated_tokens > 20000:
            print("⚠️ Document très volumineux détecté - recommandation profil 'Maxi'")
        elif estimated_tokens > 10000:
            print("⚠️ Document volumineux - recommandation profil 'Confort' ou 'Maxi'")

        profiles = {
            "Rapide":  {"num_ctx": 8192,  "temperature": 0.2},
            "Confort": {"num_ctx": 16384, "temperature": 0.2},
            "Maxi":    {"num_ctx": 32768, "temperature": 0.2},
        }
        base = profiles.get(profil, profiles["Confort"])
        num_ctx = base["num_ctx"]
        temperature = base["temperature"]
        
        if estimated_tokens > num_ctx * 0.8:
            print(f"⚠️ ATTENTION : Le texte ({estimated_tokens:,} tokens) approche la limite du contexte ({num_ctx:,})")
            print("Considérez utiliser le profil 'Maxi' pour de meilleurs résultats")

        # Utiliser le prompt système tel quel (déjà préparé par l'interface)
        final_system_prompt = system_prompt
        
        # Si mode Expert, utiliser le prompt expert au lieu du prompt fourni
        if (mode_analysis or "").lower().startswith("expert"):
            print("🎓 Mode Expert activé - Utilisation du prompt expert")
            final_system_prompt = EXPERT_PROMPT_TEXT

        print(f"🤖 Génération avec {modele} en mode {mode_analysis}...")
        print(f"🔧 Paramètres : contexte={num_ctx}, max_tokens={max_tokens_out}, temp={temperature}")
        
        # APPEL AVEC SÉPARATION SYSTÈME/UTILISATEUR
        if use_runpod:
            print("☁️ Utilisation de RunPod...")
            analyse = generate_with_runpod(
                modele, final_system_prompt, user_text,
                num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature,
                endpoint=runpod_endpoint, token=runpod_token
            )
        else:
            print("🏠 Utilisation d'Ollama...")
            analyse = generate_with_ollama(
                modele, final_system_prompt, user_text,
                num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature,
                ollama_url=ollama_url
            )

        processing_time = round(time.time() - start_time, 2)

        metadata = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': modele,
            'mode': mode_analysis,
            'profil': profil,
            'num_ctx': num_ctx,
            'max_tokens': max_tokens_out,
            'temperature': temperature,
            'processing_time': processing_time,
            'system_prompt_length': len(final_system_prompt),
            'user_text_length': len(user_text),
            'source_type': source_type,
            'anonymiser': anonymiser,
            'provider': 'RunPod' if use_runpod else 'Ollama'
        }

        provider_info = f"RunPod ({runpod_endpoint})" if use_runpod else f"Ollama ({ollama_url})"
        
        # Pas de params_info ici - c'est géré par l'interface
        
        qc_or_compare = ""
        analyse_alt = ""

        if comparer:
            print("⚖️ Génération comparative...")
            alt_system_prompt = DEFAULT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else EXPERT_PROMPT_TEXT
            
            if use_runpod:
                analyse_alt = generate_with_runpod(
                    modele, alt_system_prompt, user_text,
                    num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature,
                    endpoint=runpod_endpoint, token=runpod_token
                )
            else:
                analyse_alt = generate_with_ollama(
                    modele, alt_system_prompt, user_text,
                    num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature,
                    ollama_url=ollama_url
                )
            
            def _summary_diff(a, b):
                if not a or not b or a.startswith("❌"):
                    return "Comparaison impossible (erreur ou contenu vide)."
                
                words_a = set(w.lower() for w in re.findall(r"\b\w{4,}\b", a))
                words_b = set(w.lower() for w in re.findall(r"\b\w{4,}\b", b))
                
                if not words_a and not words_b:
                    return "Aucun mot significatif détecté."
                
                intersection = len(words_a & words_b)
                union = len(words_a | words_b) or 1
                jaccard = intersection / union
                
                return (f"Similarité lexicale : {jaccard:.2%}\n"
                       f"Mots uniques analyse 1 : {len(words_a)}\n"
                       f"Mots uniques analyse 2 : {len(words_b)}\n"
                       f"Mots communs : {intersection}")
            
            qc_or_compare = _summary_diff(analyse, analyse_alt)

        return analyse, analyse_alt, qc_or_compare, metadata

    except Exception as e:
        traceback.print_exc()
        error_msg = f"❌ Erreur lors de l'analyse : {str(e)}"
        return error_msg, "", "", {}
