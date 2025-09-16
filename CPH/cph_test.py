#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR structuré + Analyse juridique (Ollama) - VERSION COMPLÈTE AVEC ANONYMISATION
"""

import os
import json
import re
import unicodedata
import hashlib
import pickle
import time
import traceback
import argparse
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple, Optional

def check_dependencies():
    missing = []
    try:
        import gradio as gr
    except ImportError:
        missing.append("gradio")
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        from pdf2image import convert_from_path
    except ImportError:
        missing.append("pdf2image")
    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")
    
    if missing:
        print(f"Erreur: Dépendances manquantes: {', '.join(missing)}")
        print("Installez avec: pip install " + " ".join(missing))
        sys.exit(1)

check_dependencies()

import gradio as gr
import requests
from pdf2image import convert_from_path
import pytesseract

DEFAULT_PROMPT_NAME = "Par défaut (analyse rédigée)"
DEFAULT_PROMPT_TEXT = """Tu es juriste spécialisé en droit du travail. À partir du texte fourni, rédige une analyse juridique complète en français juridique, en paragraphes continus (sans listes ni numérotation), visant à faire ressortir les moyens de droit (arguments juridiques) pertinents, tels qu'ils ressortent exclusivement du document.

Exigences impératives :
- Ne JAMAIS inventer ni supposer des faits ou des références. Si une information n'apparaît pas dans le texte, écris : « non précisé dans le document ».
- Si le texte contient une référence manifestement erronée (ex. confusion entre Code du travail et Code civil), signale-le explicitement sans inventer un numéro d'article.
- Anonymisation stricte : remplace les noms par [Mme X], [M. Y], [Société Z] et les montants par [montant].

Structure implicite attendue (sans titres apparents) :
1. Qualification juridique des faits et rappel du contexte procédural (uniquement si mentionné).
2. Exposé des moyens des parties : fondements, arguments, contestations, en citant uniquement ce qui figure dans le texte.
3. Règles de droit applicables : uniquement celles présentes dans le texte ; sinon indiquer « non précisé dans le document ».
4. Discussion : articulation arguments/règles, charge de la preuve, incidences procédurales si mentionnées.
5. Application au cas d'espèce.
6. Conclusion motivée sur la portée des moyens (sans se substituer au juge).

Réponds uniquement par l'analyse rédigée, sans commentaires méta ni hypothèses."""

EXPERT_PROMPT_TEXT = """Tu es un juriste senior en droit du travail. Rédige une analyse approfondie des moyens de droit en français juridique, SANS listes ni numérotation. Mets en évidence : (i) la qualification précise des faits, (ii) l'articulation des moyens principaux et subsidiaires, (iii) le lien exact avec les références textuelles présentes DANS le document UNIQUEMENT (si une référence manque : « non précisé dans le document »), (iv) la charge de la preuve et les incidences procédurales si le texte en fait état, (v) une conclusion motivée. Aucune invention, aucune hypothèse."""

SYSTEM_PROMPT = """Tu es juriste en droit du travail. Rédige une analyse argumentée en français juridique, sans listes à puces, en citant uniquement les fondements mentionnés dans le texte."""

def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    except:
        return os.getcwd()

def _default_store_dir():
    script_dir = _script_dir()
    prompts_dir = os.path.join(script_dir, "prompts")
    try:
        os.makedirs(prompts_dir, exist_ok=True)
        test_path = os.path.join(prompts_dir, ".write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return prompts_dir
    except:
        return script_dir

PROMPT_STORE_DIR = _default_store_dir()
PROMPT_STORE_PATH = os.path.join(PROMPT_STORE_DIR, "prompts_store.json")

def get_pdf_hash(pdf_path: str):
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    try:
        with open(pdf_path, 'rb') as f:
            hash_obj = hashlib.md5()
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except:
        return None

def get_cache_path(pdf_hash: str, nettoyer: bool):
    cache_dir = os.path.join(_script_dir(), "cache_ocr")
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_clean" if nettoyer else "_raw"
    return os.path.join(cache_dir, f"ocr_{pdf_hash}{suffix}.pkl")

def save_ocr_cache(pdf_hash: str, nettoyer: bool, ocr_data: dict):
    try:
        cache_path = get_cache_path(pdf_hash, nettoyer)
        with open(cache_path, 'wb') as f:
            pickle.dump(ocr_data, f)
        return True
    except:
        return False

def load_ocr_cache(pdf_hash: str, nettoyer: bool):
    try:
        cache_path = get_cache_path(pdf_hash, nettoyer)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None

def clear_ocr_cache():
    try:
        cache_dir = os.path.join(_script_dir(), "cache_ocr")
        if not os.path.exists(cache_dir):
            return 0
        count = 0
        for filename in os.listdir(cache_dir):
            if filename.startswith("ocr_") and filename.endswith(".pkl"):
                os.remove(os.path.join(cache_dir, filename))
                count += 1
        return count
    except:
        return 0

def load_prompt_store():
    default_store = {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT}
    if not os.path.exists(PROMPT_STORE_PATH):
        try:
            os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
            with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(default_store, f, ensure_ascii=False, indent=2)
            return default_store
        except:
            return default_store
    
    try:
        with open(PROMPT_STORE_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
        if not isinstance(store, dict):
            return default_store
        if DEFAULT_PROMPT_NAME not in store:
            store[DEFAULT_PROMPT_NAME] = DEFAULT_PROMPT_TEXT
        return store
    except:
        return default_store

def save_prompt_store(store):
    try:
        os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
        with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
        return True, f"Enregistré dans : {PROMPT_STORE_PATH}"
    except Exception as e:
        return False, f"Erreur : {e}"

def sanitize_name(name: str):
    if not name:
        return ""
    name = re.sub(r"\s+", " ", name.strip())
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:100]

def _normalize_unicode(text: str):
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ï¬": "fi", "ï¬‚": "fl", "ï¬€": "ff", "ï¬ƒ": "ffi", "ï¬„": "ffl",
        "'": "'", "'": "'", """: '"', """: '"',
        "–": "-", "—": "-", "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def smart_clean(text: str, pages_texts=None):
    if not text:
        return text
    
    text = _normalize_unicode(text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    page_patterns = [
        r"(?im)^\s*page\s*[:\-]?\s*\d+\s*(?:/|sur|de|of)\s*\d+\s*$",
        r"(?im)^\s*p(?:age)?\.?\s*[:\-]?\s*\d+\s*$",
        r"(?im)^\s*\d+\s*/\s*\d+\s*$",
        r"(?im)^\s*[-–—]+\s*\d+\s*[-–—]+\s*$",
        r"(?im)^\s*\d{1,3}\s*$",
    ]
    
    for pattern in page_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[–—\-]{2,}\s*$", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()

def calculate_ocr_stats(text):
    if not text:
        return "Texte vide"
    lines = text.split('\n')
    words = len(text.split())
    chars = len(text)
    non_empty_lines = len([l for l in lines if l.strip()])
    return f"{chars:,} caractères | {words:,} mots | {non_empty_lines} lignes non vides"

def process_txt_file(txt_path: str, nettoyer: bool):
    if not txt_path or not os.path.exists(txt_path):
        return "Fichier TXT introuvable.", "", ""
    
    try:
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        content = None
        
        for encoding in encodings_to_try:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            return "Erreur : encodage non supporté.", "", ""
        
        if nettoyer:
            content = _normalize_unicode(content)
            content = clean_text_file(content)
        
        stats = calculate_ocr_stats(content)
        return "Fichier TXT chargé.", stats, content
    except Exception as e:
        return f"Erreur TXT : {e}", "", ""

def clean_text_file(text: str):
    if not text:
        return text
    
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    lines = text.split('\n')
    final_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\d{1,4}$', stripped) or re.match(r'^[-=_]{3,}$', stripped):
            continue
        final_lines.append(line)
    
    return '\n'.join(final_lines).strip()

class AnonymizationEngine:
    def __init__(self):
        self.anonymization_map = {}
        self.reverse_map = {}
        self.counters = {'person': 0, 'company': 0, 'address': 0, 'email': 0, 'phone': 0, 'amount': 0, 'date': 0}
    
    def get_or_create_reference(self, original_text: str, entity_type: str, context: str = ""):
        normalized = re.sub(r'\s+', ' ', original_text.strip().lower())
        key = f"{entity_type}:{normalized}"
        
        if key in self.anonymization_map:
            return self.anonymization_map[key]
        
        self.counters[entity_type] += 1
        count = self.counters[entity_type]
        
        if entity_type == 'person':
            if context.lower() in ['mme', 'madame']:
                reference = f"[Mme {chr(87 + count)}]"
            elif context.lower() in ['m', 'monsieur']:
                reference = f"[M. {chr(87 + count)}]"
            else:
                reference = f"[Personne {count}]"
        elif entity_type == 'company':
            reference = f"[Société {chr(64 + count)}]"
        elif entity_type == 'address':
            reference = f"[Adresse {count}]"
        elif entity_type == 'email':
            reference = f"[email{count}@exemple.fr]"
        elif entity_type == 'phone':
            reference = f"[0{count}.XX.XX.XX.XX]"
        elif entity_type == 'amount':
            reference = f"[montant {count}]"
        elif entity_type == 'date':
            reference = f"[date {count}]"
        else:
            reference = f"[{entity_type} {count}]"
        
        self.anonymization_map[key] = reference
        self.reverse_map[reference] = original_text
        return reference
    
    def get_mapping_report(self):
        if not self.anonymization_map:
            return "Aucune anonymisation effectuée."
        
        report = ["=== RAPPORT D'ANONYMISATION ===\n"]
        
        for entity_type in ['person', 'company', 'address', 'email', 'phone', 'amount', 'date']:
            items = [(k, v) for k, v in self.anonymization_map.items() if k.startswith(f"{entity_type}:")]
            if items:
                report.append(f"**{entity_type.upper()}S** :")
                for key, reference in sorted(items, key=lambda x: x[1]):
                    original = key.split(":", 1)[1]
                    report.append(f"  {reference} ← {original}")
                report.append("")
        
        report.append(f"Total: {len(self.anonymization_map)} éléments anonymisés")
        return "\n".join(report)

def anonymize_text(text: str, anonymizer: AnonymizationEngine):
    if not text:
        return text, "Aucun texte à anonymiser."
    
    anonymized_text = text
    
    # Personnes
    person_patterns = [
        r'\b(M\.?|Monsieur|Mme|Madame|Mlle)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\b([A-Z][a-z]+)\s+([A-Z]{2,})',
        r'\b([A-Z]{2,})\s+([A-Z][a-z]+)'
    ]
    
    for pattern in person_patterns:
        matches = list(re.finditer(pattern, anonymized_text))
        for match in reversed(matches):
            full_match = match.group(0)
            if pattern == person_patterns[0]:
                title = match.group(1)
                name = match.group(2)
                reference = anonymizer.get_or_create_reference(name, 'person', title.lower())
            else:
                reference = anonymizer.get_or_create_reference(full_match, 'person')
            anonymized_text = anonymized_text[:match.start()] + reference + anonymized_text[match.end():]
    
    # Sociétés
    company_patterns = [
        r'\b(Société|SARL|SAS|SA|EURL|SCI|SASU)\s+([A-Z][A-Za-z\s&]+)',
        r'\b([A-Z][A-Za-z\s&]+)\s+(SARL|SAS|SA|EURL|SCI|SASU)',
        r'\b(Entreprise|Établissements?|Ets?)\s+([A-Z][A-Za-z\s&]+)'
    ]
    
    for pattern in company_patterns:
        matches = list(re.finditer(pattern, anonymized_text))
        for match in reversed(matches):
            full_match = match.group(0)
            reference = anonymizer.get_or_create_reference(full_match, 'company')
            anonymized_text = anonymized_text[:match.start()] + reference + anonymized_text[match.end():]
    
    # Adresses
    address_patterns = [
        r'\b\d+[,\s]+(?:rue|avenue|boulevard|place|chemin|impasse|allée)\s+[A-Za-z\s\-\']+',
        r'\b\d{5}\s+[A-Z][A-Za-z\s\-\']+'
    ]
    
    for pattern in address_patterns:
        matches = list(re.finditer(pattern, anonymized_text))
        for match in reversed(matches):
            reference = anonymizer.get_or_create_reference(match.group(0), 'address')
            anonymized_text = anonymized_text[:match.start()] + reference + anonymized_text[match.end():]
    
    # Emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = list(re.finditer(email_pattern, anonymized_text))
    for match in reversed(matches):
        reference = anonymizer.get_or_create_reference(match.group(0), 'email')
        anonymized_text = anonymized_text[:match.start()] + reference + anonymized_text[match.end():]
    
    # Téléphones
    phone_patterns = [
        r'\b(?:0[1-9])(?:[\s\.-]?\d{2}){4}\b',
        r'\b(?:\+33|0033)\s?[1-9](?:[\s\.-]?\d{2}){4}\b'
    ]
    
    for pattern in phone_patterns:
        matches = list(re.finditer(pattern, anonymized_text))
        for match in reversed(matches):
            reference = anonymizer.get_or_create_reference(match.group(0), 'phone')
            anonymized_text = anonymized_text[:match.start()] + reference + anonymized_text[match.end():]
    
    # Montants
    amount_patterns = [
        r'\b\d+(?:\s?\d{3})*[,\.]\d{2}\s?€',
        r'\b\d+(?:\s?\d{3})*\s?euros?\b',
        r'\b€\s?\d+(?:\s?\d{3})*[,\.]\d{2}'
    ]
    
    for pattern in amount_patterns:
        matches = list(re.finditer(pattern, anonymized_text, re.IGNORECASE))
        for match in reversed(matches):
            reference = anonymizer.get_or_create_reference(match.group(0), 'amount')
            anonymized_text = anonymized_text[:match.start()] + reference + anonymized_text[match.end():]
    
    # Dates
    date_patterns = [
        r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b',
        r'\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{2,4}\b'
    ]
    
    for pattern in date_patterns:
        matches = list(re.finditer(pattern, anonymized_text, re.IGNORECASE))
        for match in reversed(matches):
            reference = anonymizer.get_or_create_reference(match.group(0), 'date')
            anonymized_text = anonymized_text[:match.start()] + reference + anonymized_text[match.end():]
    
    return anonymized_text, anonymizer.get_mapping_report()

def generate_with_ollama(model: str, prompt_text: str, full_text: str, num_ctx: int, num_predict: int, temperature: float = 0.2, timeout: int = 900):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": f"Texte à analyser :\n{full_text}",
        "system": f"{SYSTEM_PROMPT}\n\n{prompt_text}",
        "stream": True,
        "options": {
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        },
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return "Erreur : Impossible de se connecter à Ollama. Vérifiez qu'Ollama est démarré (ollama serve)."
    except requests.exceptions.Timeout:
        return f"Erreur : Délai dépassé ({timeout}s)."
    except Exception as e:
        return f"Erreur de connexion Ollama : {e}"
    
    if response.status_code != 200:
        return f"Erreur HTTP {response.status_code} : {response.text}"

    parts = []
    try:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
                if obj.get("response"):
                    parts.append(obj["response"])
                if obj.get("error"):
                    return f"Erreur Ollama : {obj['error']}"
            except json.JSONDecodeError:
                continue
    except Exception as e:
        return f"Erreur lors de la lecture du flux : {e}"
    
    result = "".join(parts).strip()
    return result if result else "Aucune réponse reçue."

def do_text_processing(pdf_path=None, txt_path=None, nettoyer=True, force_ocr=False, anonymiser=False):
    if not pdf_path and not txt_path:
        return "Aucun fichier fourni.", "", "", ""
    
    if pdf_path and txt_path:
        return "Choisissez PDF ou TXT, pas les deux.", "", "", ""
    
    if txt_path:
        message, stats, content = process_txt_file(txt_path, nettoyer)
    elif pdf_path:
        message, stats, content = do_ocr_only(pdf_path, nettoyer, force_ocr)
    else:
        return "Erreur inattendue.", "", "", ""
    
    anonymization_report = ""
    if anonymiser and content and not content.startswith("Erreur"):
        anonymizer = AnonymizationEngine()
        content, anonymization_report = anonymize_text(content, anonymizer)
    
    return message, stats, content, anonymization_report

def do_ocr_only(pdf_path, nettoyer, force_ocr=False):
    if not pdf_path or not os.path.exists(pdf_path):
        return "Fichier PDF introuvable.", "", ""
    
    try:
        pdf_hash = get_pdf_hash(pdf_path)
        ocr_data = None
        
        if pdf_hash and not force_ocr:
            ocr_data = load_ocr_cache(pdf_hash, nettoyer)
        
        if ocr_data:
            print("Utilisation cache OCR")
            return "OCR depuis cache.", ocr_data['stats'], ocr_data['preview']
        else:
            print(f"OCR complet : {pdf_path}")
            images = convert_from_path(pdf_path)
            raw_pages = []
            
            for i, image in enumerate(images):
                print(f"OCR page {i+1}/{len(images)}...")
                page_text = pytesseract.image_to_string(image, lang="fra")
                raw_pages.append(page_text or "")

            if nettoyer:
                cleaned_pages = [_normalize_unicode(t) for t in raw_pages]
                preview = smart_clean("\n".join(cleaned_pages), pages_texts=cleaned_pages)
            else:
                preview = "\n".join(raw_pages).strip()

            stats = calculate_ocr_stats(preview)
            
            if pdf_hash:
                cache_data = {'preview': preview, 'stats': stats, 'total_pages': len(images)}
                save_ocr_cache(pdf_hash, nettoyer, cache_data)

            return "OCR terminé.", stats, preview

    except Exception as e:
        return f"Erreur OCR : {e}", "", ""

def do_analysis_only(text_content, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer):
    if not text_content:
        return "Aucun texte pour analyse.", "", "", {}
    
    start_time = time.time()
    
    try:
        profiles = {
            "Rapide":  {"num_ctx": 8192,  "temperature": 0.2},
            "Confort": {"num_ctx": 16384, "temperature": 0.2},
            "Maxi":    {"num_ctx": 32768, "temperature": 0.2},
        }
        base = profiles.get(profil, profiles["Confort"])
        
        main_prompt = EXPERT_PROMPT_TEXT if mode_analysis == "Expert" else prompt_text
        
        analyse = generate_with_ollama(
            modele, main_prompt, text_content,
            num_ctx=base["num_ctx"], num_predict=max_tokens_out, temperature=base["temperature"]
        )

        processing_time = round(time.time() - start_time, 2)
        
        metadata = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': modele,
            'mode': mode_analysis,
            'profil': profil,
            'processing_time': processing_time,
            'prompt': main_prompt
        }

        params_info = f"""
=== PARAMÈTRES D'ANALYSE ===
Modèle : {modele}
Mode : {mode_analysis}
Profil : {profil}
Temps : {processing_time}s
Date : {metadata['timestamp']}
========================

"""

        analyse_avec_params = params_info + analyse
        analyse_alt = ""
        qc_or_compare = ""

        if comparer:
            alt_prompt = DEFAULT_PROMPT_TEXT if mode_analysis == "Expert" else EXPERT_PROMPT_TEXT
            analyse_alt = generate_with_ollama(
                modele, alt_prompt, text_content,
                num_ctx=base["num_ctx"], num_predict=max_tokens_out, temperature=base["temperature"]
            )

        return analyse_avec_params, analyse_alt, qc_or_compare, metadata

    except Exception as e:
        return f"Erreur analyse : {e}", "", "", {}

def save_analysis_result(source_file: str, model: str, analysis_text: str, metadata: dict, format_type: str = "txt"):
    try:
        script_dir = _script_dir()
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{base_name}_{model}_{timestamp}.{format_type}"
        filepath = os.path.join(script_dir, filename)
        
        content = f"""================================================================================
ANALYSE JURIDIQUE AUTOMATISÉE
================================================================================
Document source : {os.path.basename(source_file)}
Date : {metadata.get('timestamp', 'N/A')}
Modèle : {metadata.get('model', 'N/A')}
Mode : {metadata.get('mode', 'N/A')}
================================================================================

{analysis_text}
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True, filename
    except Exception as e:
        return False, f"Erreur sauvegarde : {e}"

def get_ollama_models():
    fallback_models = ["mistral:7b-instruct", "mistral:latest", "deepseek-coder:latest", "llama3:latest"]
    
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        if r.status_code != 200:
            return fallback_models
        
        data = r.json()
        models = data.get("models", [])
        names = [m.get("name") for m in models if m.get("name")]
        
        return names if names else fallback_models
    except:
        return fallback_models

def build_ui():
    models_list = get_ollama_models()
    store = load_prompt_store()
    prompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for nprompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])

    with gr.Blocks(title="OCR Juridique + Ollama") as demo:
        gr.Markdown("## OCR structuré + Analyse juridique (Ollama)")
        gr.Markdown(f"**Fichier des prompts** : `{PROMPT_STORE_PATH}`")

        with gr.Row():
            with gr.Column():
                pdf_file = gr.File(label="Uploader un PDF scanné", file_types=[".pdf"])
                txt_file = gr.File(label="Ou uploader un fichier TXT", file_types=[".txt"])
            with gr.Column():
                nettoyer = gr.Checkbox(label="Nettoyage avancé", value=True)
                anonymiser = gr.Checkbox(label="Anonymisation automatique", value=False)

        with gr.Row():
            force_ocr = gr.Checkbox(label="Forcer nouveau OCR", value=False)
            clear_cache_btn = gr.Button("Vider cache OCR", variant="secondary", size="sm")
            cache_info = gr.Markdown("")

        with gr.Row():
            default_model = "mistral:7b-instruct" if "mistral:7b-instruct" in models_list else models_list[0]
            modele = gr.Dropdown(label="Modèle Ollama", choices=models_list, value=default_model)
            profil = gr.Radio(label="Profil", choices=["Rapide", "Confort", "Maxi"], value="Confort")
            max_tokens_out = gr.Slider(label="Longueur (tokens)", minimum=256, maximum=2048, step=128, value=1280)
        
        with gr.Row():
            mode_analysis = gr.Radio(label="Mode", choices=["Standard", "Expert"], value="Standard")
            comparer = gr.Checkbox(label="Comparer avec l'autre mode", value=False)

        gr.Markdown("### Prompt — gestion persistante")

        with gr.Row():
            prompt_selector = gr.Dropdown(label="Choisir un prompt", choices=prompt_names, value=DEFAULT_PROMPT_NAME)
            prompt_name_box = gr.Textbox(label="Nom du prompt", lines=1, placeholder="Nom...")

        prompt_box = gr.Textbox(
            label="Contenu du prompt (modifiable)",
            value=store.get(DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT),
            lines=12,
            interactive=True
        )

        with gr.Row():
            save_btn = gr.Button("Enregistrer", variant="secondary")
            reset_btn = gr.Button("Réinitialiser", variant="secondary")
            reload_btn = gr.Button("Recharger", variant="secondary")

        info_box = gr.Markdown("")

        with gr.Row():
            process_btn = gr.Button("1. Traiter fichier (OCR/TXT)", variant="secondary")
            analyze_btn = gr.Button("2. Analyser", variant="primary", size="lg")
            full_btn = gr.Button("Traiter + Analyser", variant="primary")

        with gr.Row():
            save_result_btn = gr.Button("Sauvegarder résultat", variant="secondary")
            save_format = gr.Radio(label="Format", choices=["txt", "rtf"], value="txt")
            save_status = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("Analyse (mode choisi)"):
                analysis_box = gr.Textbox(label="Analyse juridique", lines=36, show_copy_button=True)
            with gr.Tab("Analyse (autre mode)"):
                analysis_alt_box = gr.Textbox(label="Analyse comparative", lines=24, show_copy_button=True)
            with gr.Tab("Contrôle qualité"):
                compare_box = gr.Textbox(label="Rapport CQ & comparatif", lines=18, show_copy_button=True)
            with gr.Tab("Anonymisation"):
                anonymization_box = gr.Textbox(label="Rapport d'anonymisation", lines=18, show_copy_button=True)
            with gr.Tab("Texte OCR"):
                ocr_stats = gr.Textbox(label="Statistiques", lines=2, interactive=False)
                preview_box = gr.Textbox(label="Texte OCR extrait", interactive=False, show_copy_button=True, lines=25)

        # États
        prompts_state = gr.State(value=store)
        current_name = gr.State(value=DEFAULT_PROMPT_NAME)
        current_ocr_text = gr.State(value="")
        current_pdf_path = gr.State(value="")
        analysis_metadata = gr.State(value={})

        # Fonctions callbacks
        def clear_cache():
            count = clear_ocr_cache()
            return gr.Markdown.update(value=f"Cache vidé : {count} fichier(s)" if count > 0 else "Cache déjà vide")

        def launch_text_processing(pdf_path, txt_path, nettoyer, anonymiser, force_ocr):
            try:
                message, stats, preview, anon_report = do_text_processing(pdf_path, txt_path, nettoyer, force_ocr, anonymiser)
                source_file = pdf_path if pdf_path else txt_path
                return (message, "", "", anon_report, stats, preview, preview, source_file if source_file else "")
            except Exception as e:
                return f"Erreur traitement : {str(e)}", "", "", "", "Erreur", "", "", ""

        def launch_analysis_only(ocr_text, source_file, modele, profil, max_tokens_out, 
                                prompt_text, mode_analysis, comparer, nettoyer, anonymiser):
            try:
                if not ocr_text and not source_file:
                    return "Aucun texte ni fichier source disponible.", "", "", "", "", "", {}
                
                if not ocr_text and source_file:
                    if source_file.lower().endswith('.pdf'):
                        message, stats, preview = do_ocr_only(source_file, nettoyer, False)
                    elif source_file.lower().endswith('.txt'):
                        message, stats, preview = process_txt_file(source_file, nettoyer)
                    else:
                        return "Type de fichier non supporté.", "", "", "", "", "", {}
                    
                    if "Erreur" in message:
                        return message, "", "", "", stats, preview, preview, {}
                    ocr_text = preview
                    
                    anon_report = ""
                    if anonymiser:
                        anonymizer = AnonymizationEngine()
                        ocr_text, anon_report = anonymize_text(ocr_text, anonymizer)
                
                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    ocr_text, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer
                )
                
                metadata['nettoyer'] = nettoyer
                metadata['anonymiser'] = anonymiser
                metadata['source_file'] = source_file
                stats = calculate_ocr_stats(ocr_text)
                
                return (analyse, analyse_alt, qc_compare, anon_report if 'anon_report' in locals() else "", 
                       stats, ocr_text, ocr_text, metadata)
                
            except Exception as e:
                return f"Erreur analyse : {str(e)}", "", "", "", "Erreur", "", "", {}

        def launch_full_pipeline(pdf_path, txt_path, nettoyer, anonymiser, force_ocr, modele, profil, max_tokens_out, 
                               prompt_text, mode_analysis, comparer):
            try:
                if not pdf_path and not txt_path:
                    return "Aucun fichier fourni.", "", "", "", "", "", "", {}
                
                message, stats, text_content, anon_report = do_text_processing(pdf_path, txt_path, nettoyer, force_ocr, anonymiser)
                if "Erreur" in message:
                    return message, "", "", anon_report, stats, text_content, text_content, {}
                
                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    text_content, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer
                )
                
                metadata['nettoyer'] = nettoyer
                metadata['anonymiser'] = anonymiser
                metadata['source_file'] = pdf_path if pdf_path else txt_path
                
                return (analyse, analyse_alt, qc_compare, anon_report, stats, text_content, text_content, metadata)
                
            except Exception as e:
                return f"Erreur pipeline : {str(e)}", "", "", "", "Erreur", "", "", {}

        def save_analysis_file(source_file, analysis_text, metadata, format_type):
            if not analysis_text or not source_file:
                return "Aucun résultat à sauvegarder"
            
            model = metadata.get('model', 'unknown')
            success, message = save_analysis_result(source_file, model, analysis_text, metadata, format_type)
            
            return f"Résultat sauvegardé : {message}" if success else f"Erreur : {message}"

        def on_file_upload(pdf_path, txt_path, nettoyer):
            if not pdf_path and not txt_path:
                return "", "", "", "", ""
            
            if pdf_path and txt_path:
                return "Choisissez PDF ou TXT, pas les deux.", "", "", "", ""
            
            source_file = pdf_path if pdf_path else txt_path
            
            try:
                if pdf_path:
                    pdf_hash = get_pdf_hash(pdf_path)
                    if pdf_hash:
                        ocr_data = load_ocr_cache(pdf_hash, nettoyer)
                        if ocr_data:
                            return ocr_data['stats'], ocr_data['preview'], ocr_data['preview'], source_file, ""
                
                return "", "", "", source_file, ""
            except:
                return "", "", "", source_file, ""

        def on_select(name, store):
            try:
                name = sanitize_name(name) or DEFAULT_PROMPT_NAME
                if name not in store:
                    name = DEFAULT_PROMPT_NAME
                text = store.get(name, DEFAULT_PROMPT_TEXT)
                return (
                    gr.Textbox.update(value=text),
                    gr.Textbox.update(value=name),
                    name,
                    gr.Markdown.update(value=f"Sélectionné : **{name}**")
                )
            except Exception as e:
                return (
                    gr.Textbox.update(value=DEFAULT_PROMPT_TEXT),
                    gr.Textbox.update(value=DEFAULT_PROMPT_NAME),
                    DEFAULT_PROMPT_NAME,
                    gr.Markdown.update(value=f"Erreur : {e}")
                )

        def on_save(current_name_val, text, store):
            try:
                name = sanitize_name(current_name_val) or DEFAULT_PROMPT_NAME
                if not text.strip():
                    return (
                        gr.Dropdown.update(),
                        gr.Markdown.update(value="Le contenu ne peut pas être vide."),
                        store,
                        current_name_val
                    )
                
                store[name] = text.strip()
                ok, msg = save_prompt_store(store)
                
                if ok:
                    names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
                    return (
                        gr.Dropdown.update(choices=names, value=name),
                        gr.Markdown.update(value=f"Enregistré : {msg}"),
                        store,
                        name
                    )
                else:
                    return (
                        gr.Dropdown.update(),
                        gr.Markdown.update(value=f"{msg}"),
                        store,
                        current_name_val
                    )
            except Exception as e:
                return (
                    gr.Dropdown.update(),
                    gr.Markdown.update(value=f"Erreur : {e}"),
                    store,
                    current_name_val
                )

        # Connexions
        pdf_file.change(
            fn=on_file_upload,
            inputs=[pdf_file, txt_file, nettoyer],
            outputs=[ocr_stats, preview_box, current_ocr_text, current_pdf_path, anonymization_box]
        )
        
        txt_file.change(
            fn=on_file_upload,
            inputs=[pdf_file, txt_file, nettoyer],
            outputs=[ocr_stats, preview_box, current_ocr_text, current_pdf_path, anonymization_box]
        )
        
        prompt_selector.change(
            fn=on_select,
            inputs=[prompt_selector, prompts_state],
            outputs=[prompt_box, prompt_name_box, current_name, info_box]
        )

        save_btn.click(
            fn=on_save,
            inputs=[current_name, prompt_box, prompts_state],
            outputs=[prompt_selector, info_box, prompts_state, current_name]
        )

        process_btn.click(
            fn=launch_text_processing,
            inputs=[pdf_file, txt_file, nettoyer, anonymiser, force_ocr],
            outputs=[analysis_box, analysis_alt_box, compare_box, anonymization_box, ocr_stats, preview_box, current_ocr_text, current_pdf_path]
        )

        analyze_btn.click(
            fn=launch_analysis_only,
            inputs=[current_ocr_text, current_pdf_path, modele, profil, max_tokens_out, 
                   prompt_box, mode_analysis, comparer, nettoyer, anonymiser],
            outputs=[analysis_box, analysis_alt_box, compare_box, anonymization_box, ocr_stats, preview_box, current_ocr_text, analysis_metadata]
        )

        full_btn.click(
            fn=launch_full_pipeline,
            inputs=[pdf_file, txt_file, nettoyer, anonymiser, force_ocr, modele, profil, max_tokens_out, 
                   prompt_box, mode_analysis, comparer],
            outputs=[analysis_box, analysis_alt_box, compare_box, anonymization_box, ocr_stats, preview_box, current_ocr_text, analysis_metadata]
        )

        save_result_btn.click(
            fn=save_analysis_file,
            inputs=[current_pdf_path, analysis_box, analysis_metadata, save_format],
            outputs=[save_status]
        )

        clear_cache_btn.click(fn=clear_cache, outputs=[cache_info])

        gr.Markdown("""
        ### Guide d'utilisation

        **Types de fichiers supportés :**
        - **PDF scannés** : OCR automatique avec cache intelligent
        - **Fichiers TXT** : Traitement direct avec nettoyage optionnel

        **Anonymisation automatique :**
        - **Personnes** : M./Mme + Nom → [M. X], [Mme Y]
        - **Sociétés** : SARL/SAS + Nom → [Société A], [Société B]
        - **Adresses** : Rue + Code postal → [Adresse 1], [Adresse 2]
        - **Contacts** : emails, téléphones → [email1@exemple.fr], [01.XX.XX.XX.XX]
        - **Montants** : euros, sommes → [montant 1], [montant 2]
        - **Dates** : formats variés → [date 1], [date 2]

        **Modèles recommandés (selon RAM) :**
        - **8 GB ou moins** : mistral:7b-instruct, deepseek-coder:latest
        - **12+ GB** : llama3:latest, mistral:latest

        **Prérequis** : Ollama démarré (`ollama serve`)
        """)

    return demo

def main():
    parser = argparse.ArgumentParser(description="OCR + Analyse juridique avec Ollama")
    parser.add_argument('--list-models', action='store_true', help='Lister les modèles Ollama disponibles')
    parser.add_argument('--host', default='127.0.0.1', help='Adresse d\'écoute')
    parser.add_argument('--port', type=int, help='Port d\'écoute')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Modèles Ollama disponibles:")
        try:
            models = get_ollama_models()
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
        except Exception as e:
            print(f"Erreur: {e}")
        return
    
    print("Démarrage de l'interface OCR Juridique...")
    print(f"Répertoire des prompts : {PROMPT_STORE_DIR}")
    
    try:
        models = get_ollama_models()
        print(f"Modèles disponibles : {len(models)}")
        
        app = build_ui()
        
        launch_kwargs = {
            'server_name': args.host,
            'share': False,
            'inbrowser': True
        }
        
        if args.port:
            launch_kwargs['server_port'] = args.port
            
        app.launch(**launch_kwargs)
        
    except Exception as e:
        print(f"Erreur : {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
