#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR structuré + Analyse juridique (Ollama) - VERSION COMPLÈTE
Support PDF (OCR avec cache) et TXT avec anonymisation
Author: Assistant Claude
Version: 1.0
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

# Vérification des dépendances
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

# Configuration
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
        return prompts_dir
    except:
        return script_dir

PROMPT_STORE_DIR = _default_store_dir()
PROMPT_STORE_PATH = os.path.join(PROMPT_STORE_DIR, "prompts_store.json")

# Gestion des fichiers
def read_text_file(file_path: str) -> Tuple[str, str]:
    if not file_path or not os.path.exists(file_path):
        return "", "Fichier non trouvé"
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return content.strip(), f"Fichier lu avec encodage {encoding}"
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return "", f"Erreur lecture : {str(e)}"
    return "", "Impossible de décoder le fichier"

def get_file_type(file_path: str) -> str:
    if not file_path:
        return "UNKNOWN"
    ext = file_path.lower()
    if ext.endswith('.pdf'):
        return "PDF"
    elif ext.endswith(('.txt', '.text')):
        return "TXT"
    else:
        return "UNKNOWN"

# Anonymisation
class AnonymizationManager:
    def __init__(self):
        self.person_counter = 0
        self.company_counter = 0
        self.place_counter = 0
        self.replacements = {}
    
    def get_unique_reference(self, entity_type: str, original: str) -> str:
        if original in self.replacements:
            return self.replacements[original]
        
        if entity_type == "person":
            self.person_counter += 1
            ref = f"[Personne-{self.person_counter}]"
        elif entity_type == "company":
            self.company_counter += 1
            ref = f"[Société-{self.company_counter}]"
        elif entity_type == "place":
            self.place_counter += 1
            ref = f"[Lieu-{self.place_counter}]"
        else:
            ref = "[Entité]"
        
        self.replacements[original] = ref
        return ref
    
    def get_mapping_report(self) -> str:
        if not self.replacements:
            return "Aucune anonymisation effectuée."
        
        report = ["=== RAPPORT D'ANONYMISATION ===\n"]
        
        persons = {k: v for k, v in self.replacements.items() if "[Personne-" in v}
        companies = {k: v for k, v in self.replacements.items() if "[Société-" in v}
        places = {k: v for k, v in self.replacements.items() if "[Lieu-" in v}
        
        if persons:
            report.append("PERSONNES :")
            for original, replacement in sorted(persons.items()):
                report.append(f"  {original} → {replacement}")
            report.append("")
        
        if companies:
            report.append("SOCIÉTÉS :")
            for original, replacement in sorted(companies.items()):
                report.append(f"  {original} → {replacement}")
            report.append("")
        
        if places:
            report.append("LIEUX :")
            for original, replacement in sorted(places.items()):
                report.append(f"  {original} → {replacement}")
            report.append("")
        
        report.append(f"Total : {len(self.replacements)} entité(s) anonymisée(s)")
        return "\n".join(report)

def anonymize_text(text: str) -> tuple:
    if not text or not text.strip():
        return text, ""
    
    anonymizer = AnonymizationManager()
    result = text
    
    # Prénoms français courants
    french_firstnames = [
        "Pierre", "Jean", "Michel", "André", "Philippe", "Alain", "Bernard", "Claude", "Daniel", "Jacques",
        "François", "Henri", "Louis", "Marcel", "Paul", "Robert", "Roger", "Serge", "Christian", "Gérard",
        "Maurice", "Raymond", "René", "Guy", "Antoine", "Julien", "Nicolas", "Olivier", "Pascal", "Patrick",
        "Stéphane", "Thierry", "Vincent", "Xavier", "Yves", "Alexandre", "Christophe", "David", "Frédéric",
        "Laurent", "Sébastien", "Éric", "Fabrice", "Guillaume", "Jérôme", "Ludovic", "Mathieu", "Maxime",
        "Thomas", "Adrien", "Arthur", "Hugo", "Lucas", "Nathan", "Raphaël", "Gabriel", "Léo", "Adam",
        "Marie", "Monique", "Françoise", "Isabelle", "Catherine", "Sylvie", "Anne", "Christine", "Martine",
        "Brigitte", "Jacqueline", "Nathalie", "Chantal", "Nicole", "Véronique", "Dominique", "Christiane",
        "Patricia", "Céline", "Corinne", "Sandrine", "Valérie", "Karine", "Stéphanie", "Sophie", "Laurence",
        "Julie", "Carole", "Caroline", "Élisabeth", "Hélène", "Agnès", "Pascale", "Mireille", "Danielle",
        "Sylviane", "Florence", "Virginie", "Aurélie", "Émilie", "Mélanie", "Sarah", "Amélie", "Claire",
        "Charlotte", "Léa", "Manon", "Emma", "Chloé", "Camille", "Océane", "Marie-Christine", "Anne-Marie"
    ]
    
    # Noms de famille français courants
    french_surnames = [
        "Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy", "Moreau",
        "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier",
        "Morel", "Girard", "André", "Lefèvre", "Mercier", "Dupont", "Lambert", "Bonnet", "François", "Martinez",
        "Legrand", "Garnier", "Faure", "Rousseau", "Blanc", "Guerin", "Muller", "Henry", "Roussel", "Nicolas",
        "Perrin", "Morin", "Mathieu", "Clement", "Gauthier", "Dumont", "Lopez", "Fontaine", "Chevalier", "Robin",
        "Masson", "Sanchez", "Gerard", "Nguyen", "Boyer", "Denis", "Lemaire", "Duval", "Gautier", "Hernandez"
    ]
    
    # Patterns de civilité
    civility_patterns = [
        r"\b(?:M\.|Mr\.|Monsieur)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Mme|Madame)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Mlle|Mademoiselle)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Dr|Docteur)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b",
        r"\b(?:Me|Maître)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b"
    ]
    
    for pattern in civility_patterns:
        matches = re.finditer(pattern, result, re.IGNORECASE)
        for match in reversed(list(matches)):
            full_name = match.group(1)
            title = result[match.start():match.start(1)].strip()
            replacement = anonymizer.get_unique_reference("person", full_name)
            result = result[:match.start()] + title + " " + replacement + result[match.end():]
    
    # Anonymisation des prénoms avec contexte
    for firstname in french_firstnames:
        pattern = r'\b(' + re.escape(firstname) + r')\b'
        matches = list(re.finditer(pattern, result, re.IGNORECASE))
        for match in reversed(matches):
            original = match.group(1)
            start_pos = max(0, match.start() - 20)
            end_pos = min(len(result), match.end() + 20)
            context = result[start_pos:end_pos].lower()
            
            if any(indicator in context for indicator in [
                'monsieur', 'madame', 'mademoiselle', 'appelant', 'défendeur', 
                'demandeur', 'salarié', 'employé', 'directeur', 'gérant'
            ]):
                replacement = anonymizer.get_unique_reference("person", original)
                result = result[:match.start()] + replacement + result[match.end():]
    
    # Noms de famille
    for surname in french_surnames:
        pattern = r'\b(' + re.escape(surname) + r')\b'
        matches = list(re.finditer(pattern, result))
        for match in reversed(matches):
            if match.group(1)[0].isupper():
                original = match.group(1)
                replacement = anonymizer.get_unique_reference("person", original)
                result = result[:match.start()] + replacement + result[match.end():]
    
    # Patterns de sociétés
    company_patterns = [
        r'\b([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\s+(?:SARL|SAS|SA|EURL|SNC|SCI|SASU)\b',
        r'\b(?:SARL|SAS|SA|EURL|SNC|SCI|SASU)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+(?:\s+[A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ]+)*)\b',
        r'\b(?:Société|Entreprise|Établissements?)\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][A-Za-zàâäçéèêëîïôöùûüÿñ\s]+)\b',
        r'\b([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][A-Za-zàâäçéèêëîïôöùûüÿñ\s&]+)\s+(?:et\s+(?:Fils|Associés|Cie))\b'
    ]
    
    for pattern in company_patterns:
        matches = re.finditer(pattern, result, re.IGNORECASE)
        for match in reversed(list(matches)):
            company_name = match.group(1).strip()
            if len(company_name) > 2:
                replacement = anonymizer.get_unique_reference("company", company_name)
                result = result.replace(match.group(0), match.group(0).replace(company_name, replacement))
    
    # Adresses
    address_patterns = [
        r'\b\d+,?\s+(?:rue|avenue|boulevard|place|impasse|allée|chemin|route)\s+([A-Za-zàâäçéèêëîïôöùûüÿñ\s\-\']+)\b',
        r'\b(?:rue|avenue|boulevard|place|impasse|allée|chemin|route)\s+([A-Za-zàâäçéèêëîïôöùûüÿñ\s\-\']+)\b'
    ]
    
    for pattern in address_patterns:
        matches = re.finditer(pattern, result, re.IGNORECASE)
        for match in reversed(list(matches)):
            street_name = match.group(1).strip()
            if len(street_name) > 3:
                replacement = anonymizer.get_unique_reference("place", street_name)
                result = result.replace(match.group(0), match.group(0).replace(street_name, replacement))
    
    # Codes postaux et villes
    postal_pattern = r'\b(\d{5})\s+([A-ZÀÂÄÇÉÈÊËÏÎÔÙÛÜŸÑ][a-zàâäçéèêëîïôöùûüÿñ\s\-]+)\b'
    matches = re.finditer(postal_pattern, result)
    for match in reversed(list(matches)):
        city = match.group(2).strip()
        if len(city) > 2:
            city_replacement = anonymizer.get_unique_reference("place", city)
            postal_replacement = "[Code-Postal]"
            result = result[:match.start()] + postal_replacement + " " + city_replacement + result[match.end():]
    
    # Téléphones, emails, montants
    result = re.sub(r'\b0[1-9](?:[.\-\s]?\d{2}){4}\b', '[Téléphone]', result)
    result = re.sub(r'\b(?:\+33\s?[1-9]|0[1-9])(?:[.\-\s]?\d{2}){4}\b', '[Téléphone]', result)
    result = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[Email]', result)
    result = re.sub(r'\b\d+(?:\.\d{3})*,\d{2}\s*(?:euros?|€)\b', '[Montant]', result, flags=re.IGNORECASE)
    result = re.sub(r'\b\d+(?:[.,]\d{2,3})*\s*(?:euros?|€)\b', '[Montant]', result, flags=re.IGNORECASE)
    
    return result, anonymizer.get_mapping_report()

# Cache OCR
def get_pdf_hash(pdf_path: str) -> Optional[str]:
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

def get_cache_path(pdf_hash: str, nettoyer: bool) -> str:
    cache_dir = os.path.join(_script_dir(), "cache_ocr")
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_clean" if nettoyer else "_raw"
    return os.path.join(cache_dir, f"ocr_{pdf_hash}{suffix}.pkl")

def save_ocr_cache(pdf_hash: str, nettoyer: bool, ocr_data: dict) -> bool:
    try:
        cache_path = get_cache_path(pdf_hash, nettoyer)
        with open(cache_path, 'wb') as f:
            pickle.dump(ocr_data, f)
        print(f"OCR mis en cache : {cache_path}")
        return True
    except Exception as e:
        print(f"Erreur sauvegarde cache : {e}")
        return False

def load_ocr_cache(pdf_hash: str, nettoyer: bool) -> Optional[dict]:
    try:
        cache_path = get_cache_path(pdf_hash, nettoyer)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"OCR chargé depuis le cache : {cache_path}")
            return data
    except Exception as e:
        print(f"Erreur chargement cache : {e}")
    return None

def clear_ocr_cache() -> int:
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
    except Exception as e:
        print(f"Erreur nettoyage cache : {e}")
        return 0

# Nettoyage du texte
def _normalize_unicode(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "'": "'", "'": "'", """: '"', """: '"',
        "–": "-", "—": "-", "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def smart_clean(text: str, pages_texts=None) -> str:
    if not text:
        return text

    text = _normalize_unicode(text)

    if pages_texts:
        pages_lines = [t.splitlines() for t in pages_texts]
        text = "\n".join("\n".join(lines) for lines in pages_lines)

    # Réparer les césures
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Patterns de numéros de page
    page_patterns = [
        r"(?im)^\s*page\s*[:\-]?\s*\d+\s*(?:/|sur|de|of)\s*\d+\s*$",
        r"(?im)^\s*p(?:age)?\.?\s*[:\-]?\s*\d+\s*$",
        r"(?im)^\s*\d+\s*/\s*\d+\s*$",
        r"(?im)^\s*[-–—]+\s*\d+\s*[-–—]+\s*$",
        r"(?im)^\s*\d{1,3}\s*$",
        r"(?im)^\s*page\s+n[°º]\s*\d+\s*$",
        r"(?im)^\s*page\s*[-–—]+\s*\d+\s*[-–—]+\s*$",
    ]
    
    for pattern in page_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # Nettoyage général
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[–—\-]{2,}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-–—\s]*\s*$", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"(?<![.!?:;])\n(?!\n)(?=[a-zàâäçéèêëîïôöùûüÿñ\(\,\;\:\.\-])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    text = text.strip()
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r'^[\d\s\-–—\.]{1,10}$', stripped):
            cleaned_lines.append(line)
        elif not stripped:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def calculate_text_stats(text):
    if not text:
        return "Texte vide"
    
    lines = text.split('\n')
    words = len(text.split())
    chars = len(text)
    non_empty_lines = len([l for l in lines if l.strip()])
    
    return f"{chars:,} caractères | {words:,} mots | {non_empty_lines} lignes non vides | {len(lines)} lignes totales"

# Gestion des prompts
def load_prompt_store() -> Dict[str, str]:
    default_store = {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT}
    
    if not os.path.exists(PROMPT_STORE_PATH):
        try:
            os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
            with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
                json.dump(default_store, f, ensure_ascii=False, indent=2)
            return default_store
        except Exception as e:
            print(f"Attention: Impossible de créer le store initial: {e}")
            return default_store
    
    try:
        with open(PROMPT_STORE_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
        
        if not isinstance(store, dict) or not store:
            return default_store
            
        if DEFAULT_PROMPT_NAME not in store:
            store[DEFAULT_PROMPT_NAME] = DEFAULT_PROMPT_TEXT
            
        return store
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Erreur chargement store: {e}")
        return default_store

def save_prompt_store(store: Dict[str, str]) -> Tuple[bool, str]:
    try:
        if not isinstance(store, dict):
            return False, "Erreur: store n'est pas un dictionnaire"
        
        os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
        json_data = json.dumps(store, ensure_ascii=False, indent=2)
        
        with open(PROMPT_STORE_PATH, "w", encoding="utf-8") as f:
            f.write(json_data)
        
        return True, f"Enregistré dans : `{PROMPT_STORE_PATH}`"
        
    except Exception as e:
        return False, f"Échec d'enregistrement : {e}"

# API Ollama
def generate_with_ollama(model: str, prompt_text: str, full_text: str,
                         num_ctx: int, num_predict: int, temperature: float = 0.2,
                         timeout: int = 900) -> str:
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": f"Texte à analyser :\n{full_text}",
        "system": prompt_text,  # Le prompt utilisateur est maintenant passé directement en system
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
        return "❌ Erreur : Impossible de se connecter à Ollama. Vérifiez qu'Ollama est démarré (ollama serve)."
    except requests.exceptions.Timeout:
        return f"❌ Erreur : Délai dépassé ({timeout}s)."
    except Exception as e:
        return f"❌ Erreur de connexion Ollama : {e}"
    
    if response.status_code != 200:
        error_text = response.text
        if "system memory" in error_text.lower() and "available" in error_text.lower():
            return f"❌ MÉMOIRE INSUFFISANTE : Le modèle nécessite plus de RAM que disponible.\n\n" \
                   f"Solutions :\n" \
                   f"1. Utilisez un modèle plus léger (mistral:7b-instruct ou deepseek-coder:latest)\n" \
                   f"2. Fermez d'autres applications pour libérer de la RAM\n" \
                   f"3. Redémarrez Ollama : 'ollama serve'\n\n" \
                   f"Erreur complète : {error_text}"
        return f"❌ Erreur HTTP {response.status_code} : {error_text}"

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
                    return f"❌ Erreur Ollama : {obj['error']}"
            except json.JSONDecodeError:
                continue
    except Exception as e:
        return f"❌ Erreur lors de la lecture du flux : {e}"
    
    result = "".join(parts).strip()
    return result if result else "❌ Aucune réponse reçue (flux vide)."

def get_ollama_models():
    fallback_models = [
        "mistral:7b-instruct",
        "mistral:latest", 
        "deepseek-coder:latest",
        "llama3:latest",
        "llama3.1:8b-instruct-q5_K_M"
    ]
    
    try:
        print("🔍 Tentative de récupération des modèles Ollama...")
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if r.status_code != 200:
            print(f"❌ Erreur API Ollama - Status {r.status_code}: {r.text}")
            return fallback_models
        
        data = r.json()
        models = data.get("models", [])
        names = []
        
        for m in models:
            name = m.get("name")
            if name:
                names.append(name)
        
        if not names:
            print("⚠️ Aucun modèle trouvé dans la réponse, utilisation des modèles de fallback")
            return fallback_models
        
        print(f"✅ Total: {len(names)} modèle(s) récupéré(s) depuis l'API")
        return names
        
    except requests.exceptions.ConnectionError:
        print("❌ Ollama non accessible (connexion refusée)")
        print("Vérifiez qu'Ollama est démarré avec: ollama serve")
        return fallback_models
    except Exception as e:
        print(f"❌ Erreur récupération modèles: {e}")
        return fallback_models

# Pipeline de traitement
def process_file_to_text(file_path, nettoyer, anonymiser, force_ocr=False):
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

def do_analysis_only(text_content, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer, source_type="UNKNOWN", anonymiser=False):
    if not text_content or not text_content.strip():
        return "❌ Aucun texte disponible pour l'analyse.", "", "", {}
    
    start_time = time.time()
    
    try:
        text_length = len(text_content)
        estimated_tokens = text_length // 4
        
        print(f"📊 Longueur du texte : {text_length:,} caractères (≈{estimated_tokens:,} tokens)")
        
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

        main_prompt = EXPERT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else prompt_text

        print(f"🤖 Génération avec {modele} en mode {mode_analysis}...")
        
        analyse = generate_with_ollama(
            modele, main_prompt, text_content,
            num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature
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
            'prompt': main_prompt,
            'source_type': source_type,
            'anonymiser': anonymiser
        }

        params_info = f"""
=== PARAMÈTRES D'ANALYSE ===
Type de source : {source_type}
Modèle : {modele}
Mode : {mode_analysis}
Profil : {profil} (contexte: {num_ctx}, température: {temperature})
Longueur max : {max_tokens_out} tokens
Temps de traitement : {processing_time}s
Anonymisation : {'Oui' if anonymiser else 'Non'}
Date : {metadata['timestamp']}
========================

"""

        analyse_avec_params = params_info + analyse

        qc_or_compare = ""
        analyse_alt = ""

        if comparer:
            print("⚖️ Génération comparative...")
            alt_prompt = DEFAULT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else EXPERT_PROMPT_TEXT
            analyse_alt = generate_with_ollama(
                modele, alt_prompt, text_content,
                num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature
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

        return analyse_avec_params, analyse_alt, qc_or_compare, metadata

    except Exception as e:
        traceback.print_exc()
        error_msg = f"❌ Erreur lors de l'analyse : {str(e)}"
        return error_msg, "", "", {}

# Interface utilisateur
def build_ui():
    models_list = get_ollama_models()
    store = load_prompt_store()
    prompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
    
    script_name = os.path.basename(__file__) if '__file__' in globals() else "ocr_legal_tool.py"

    with gr.Blocks(title=f"{script_name} - OCR Juridique + Ollama") as demo:
        gr.Markdown("## OCR structuré + Analyse juridique (Ollama) - Avec support TXT et Anonymisation")
        gr.Markdown(f"**Fichier des prompts** : `{PROMPT_STORE_PATH}`")

        with gr.Row():
            input_file = gr.File(label="Uploader un fichier (PDF ou TXT)", file_types=[".pdf", ".txt", ".text"])
            with gr.Column():
                nettoyer = gr.Checkbox(label="Nettoyage avancé", value=True)
                anonymiser = gr.Checkbox(label="Anonymisation automatique", value=False)

        with gr.Row():
            force_processing = gr.Checkbox(label="Forcer nouveau traitement (ignorer cache PDF)", value=False)
            clear_cache_btn = gr.Button("Vider le cache OCR", variant="secondary", size="sm")
            cache_info = gr.Markdown("")

        with gr.Row():
            if "mistral:7b-instruct" in models_list:
                default_model = "mistral:7b-instruct"
            elif "deepseek-coder:latest" in models_list:
                default_model = "deepseek-coder:latest"
            elif "mistral:latest" in models_list:
                default_model = "mistral:latest"
            else:
                default_model = models_list[0] if models_list else "mistral:latest"
                
            modele = gr.Dropdown(label="Modèle Ollama", choices=models_list, value=default_model)
            profil = gr.Radio(label="Profil", choices=["Rapide", "Confort", "Maxi"], value="Confort")
            max_tokens_out = gr.Slider(label="Longueur (tokens)", minimum=256, maximum=2048, step=128, value=1280)
        
        with gr.Row():
            mode_analysis = gr.Radio(label="Mode", choices=["Standard", "Expert"], value="Standard")
            comparer = gr.Checkbox(label="Comparer avec l'autre mode", value=False)

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
                anonymization_report_box = gr.Textbox(label="Rapport d'anonymisation", interactive=False, show_copy_button=True, lines=25, placeholder="Le rapport d'anonymisation apparaîtra ici si l'anonymisation est activée...")

        # États
        current_text = gr.State(value="")
        current_file_path = gr.State(value="")
        analysis_metadata = gr.State(value={})

        # Callbacks
        def clear_cache():
            count = clear_ocr_cache()
            if count > 0:
                return gr.Markdown.update(value=f"Cache vidé : {count} fichier(s) supprimé(s)")
            else:
                return gr.Markdown.update(value="Cache déjà vide")

        def launch_processing_only(file_path, nettoyer, anonymiser, force_processing):
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
                                provider, runpod_endpoint, runpod_token):
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
                
                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    text_content, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer, 
                    source_type, anonymiser, use_runpod, runpod_endpoint, runpod_token
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
                               prompt_text, mode_analysis, comparer, provider, runpod_endpoint, runpod_token):
            try:
                if not file_path:
                    return "❌ Aucun fichier fourni.", "", "", "", "", "", "", {}
                
                message, stats, text_content, file_type, anon_report = process_file_to_text(file_path, nettoyer, anonymiser, force_processing)
                if "❌" in message:
                    return message, "", "", stats, text_content, anon_report, text_content, {}
                
                use_runpod = provider == "RunPod.io"
                
                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    text_content, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer, 
                    file_type, anonymiser, use_runpod, runpod_endpoint, runpod_token
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
                
            except Exception as e:
                traceback.print_exc()
                return f"❌ Erreur pipeline : {str(e)}", "", "", "Erreur", "", "", "", {}

        def on_file_upload(file_path, nettoyer, anonymiser):
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
            try:
                if name not in store:
                    name = DEFAULT_PROMPT_NAME
                text = store.get(name, DEFAULT_PROMPT_TEXT)
                return gr.Textbox.update(value=text)
            except Exception as e:
                return gr.Textbox.update(value=DEFAULT_PROMPT_TEXT)

        # Connexions
        input_file.change(
            fn=on_file_upload,
            inputs=[input_file, nettoyer, anonymiser],
            outputs=[text_stats, preview_box, anonymization_report_box, current_text, current_file_path]
        )
        
        prompt_selector.change(
            fn=on_select,
            inputs=[prompt_selector, gr.State(value=store)],
            outputs=[prompt_box]
        )

        process_btn.click(
            fn=launch_processing_only,
            inputs=[input_file, nettoyer, anonymiser, force_processing],
            outputs=[analysis_box, analysis_alt_box, compare_box, text_stats, preview_box, anonymization_report_box, current_text, current_file_path]
        )

        analyze_btn.click(
            fn=launch_analysis_only,
            inputs=[current_text, current_file_path, modele, profil, max_tokens_out, 
                   prompt_box, mode_analysis, comparer, nettoyer, anonymiser],
            outputs=[analysis_box, analysis_alt_box, compare_box, text_stats, preview_box, anonymization_report_box, current_text, analysis_metadata]
        )

        full_btn.click(
            fn=launch_full_pipeline,
            inputs=[input_file, nettoyer, anonymiser, force_processing, modele, profil, max_tokens_out, 
                   prompt_box, mode_analysis, comparer],
            outputs=[analysis_box, analysis_alt_box, compare_box, text_stats, preview_box, anonymization_report_box, current_text, analysis_metadata]
        )

        clear_cache_btn.click(
            fn=clear_cache,
            outputs=[cache_info]
        )

        gr.Markdown("""
        ### Guide d'utilisation

        **Flux de travail recommandé :**
        1. **Uploader un fichier** (PDF ou TXT) - Le cache se charge automatiquement si disponible
        2. **Traiter fichier** - Traite le PDF (OCR) ou lit le TXT directement
        3. **Analyser** - Lance l'analyse juridique sur le texte disponible

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

        **Modèles recommandés (selon RAM) :**
        - **8 GB ou moins** : mistral:7b-instruct, deepseek-coder:latest
        - **12+ GB** : llama3:latest, mistral:latest
        - **16+ GB** : llama3.1:8b-instruct-q5_K_M

        **Profils d'inférence :**
        - **Rapide** : Documents < 15 pages (8k contexte)
        - **Confort** : Documents 15-30 pages (16k contexte)  
        - **Maxi** : Documents 30+ pages (32k contexte)

        **Prérequis** : Ollama démarré (`ollama serve`)  
        **Répertoires** : Cache `./cache_ocr/` | Prompts `./prompts/`
        
        **Note sur l'anonymisation** : Le cache OCR n'est pas utilisé quand l'anonymisation est activée pour éviter les conflits entre versions anonymisées et non-anonymisées.
        """)

    return demo

def main():
    parser = argparse.ArgumentParser(description="OCR + Analyse juridique avec Ollama (PDF/TXT + Anonymisation)")
    parser.add_argument('--list-models', action='store_true', help='Lister les modèles Ollama disponibles')
    parser.add_argument('--host', default='127.0.0.1', help='Adresse d\'écoute (défaut: 127.0.0.1)')
    parser.add_argument('--port', type=int, help='Port d\'écoute (défaut: auto)')
    
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
    
    print("🚀 Démarrage de l'interface OCR Juridique (PDF/TXT + Anonymisation)...")
    print(f"📁 Répertoire des prompts : {PROMPT_STORE_DIR}")
    
    if not os.path.exists(PROMPT_STORE_DIR):
        os.makedirs(PROMPT_STORE_DIR, exist_ok=True)
    
    try:
        models = get_ollama_models()
        print(f"🤖 Modèles disponibles : {len(models)}")
        
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
        print(f"❌ Erreur : {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
