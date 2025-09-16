#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PERSO_cph_5.py — OCR + Analyse juridique (Ollama) avec:
- URL Ollama modifiable + persistante (settings.json)
- Rafraîchissement de la liste des modèles
- OCR avec cache et nettoyage avancé
- Prompts persistants (JSON) et compositeur de prompts (fichiers prompt*.txt)
- Contrôle qualité des références juridiques
- CLI: --api-url, --list-models, --host, --port
"""

import os
import sys
import re
import json
import time
import pickle
import hashlib
import traceback
import argparse
import unicodedata
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional, Set, List

# -------------------------
# Vérification dépendances
# -------------------------
def check_dependencies():
    missing = []
    try:
        import gradio as gr  # noqa
    except ImportError:
        missing.append("gradio")
    try:
        import requests  # noqa
    except ImportError:
        missing.append("requests")
    try:
        from pdf2image import convert_from_path  # noqa
    except ImportError:
        missing.append("pdf2image")
    try:
        import pytesseract  # noqa
    except ImportError:
        missing.append("pytesseract")
    if missing:
        print(f"Erreur: dépendances manquantes: {', '.join(missing)}")
        print("Installez avec: pip install " + " ".join(missing))
        sys.exit(1)

check_dependencies()
import gradio as gr
import requests
from pdf2image import convert_from_path
import pytesseract


# -------------------------
# Répertoires / fichiers
# -------------------------
def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()

def _default_store_dir() -> str:
    base = _script_dir()
    prompts_dir = os.path.join(base, "prompts")
    try:
        os.makedirs(prompts_dir, exist_ok=True)
        # test d'écriture
        probe = os.path.join(prompts_dir, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return prompts_dir
    except Exception as e:
        print(f"Attention: Impossible de créer ./prompts : {e}")
        return base

PROMPT_STORE_DIR = _default_store_dir()
PROMPT_STORE_PATH = os.path.join(PROMPT_STORE_DIR, "prompts_store.json")
PROMPT_FILES_DIR = PROMPT_STORE_DIR
SETTINGS_PATH = os.path.join(PROMPT_STORE_DIR, "settings.json")

# -------------------------
# I/O atomique texte
# -------------------------
def atomic_write_text(path: str, data: str) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

# -------------------------
# Persistance URL Ollama
# -------------------------
def _load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Avertissement: impossible de charger {SETTINGS_PATH}: {e}")
    return {}

def _save_settings(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        atomic_write_text(SETTINGS_PATH, json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Avertissement: impossible d'enregistrer {SETTINGS_PATH}: {e}")

def get_saved_api_url(default: str = "http://localhost:11434") -> str:
    s = _load_settings()
    url = s.get("ollama_api_url", default) or default
    return url.strip().rstrip("/")

def set_saved_api_url(url: str):
    s = _load_settings()
    s["ollama_api_url"] = (url or "").strip().rstrip("/")
    _save_settings(s)

# -------------------------
# Prompts persistants (JSON)
# -------------------------
DEFAULT_PROMPT_NAME = "Par défaut (analyse rédigée)"
DEFAULT_PROMPT_TEXT = """Tu es juriste spécialisé en droit du travail. À partir du texte fourni, rédige une analyse juridique complète en français juridique, en paragraphes continus (sans listes ni numérotation), visant à faire ressortir les moyens de droit pertinents, tels qu'ils ressortent exclusivement du document.
Exigences impératives :
- Ne JAMAIS inventer ni supposer des faits ou des références. Si une information n'apparaît pas dans le texte, écris : « non précisé dans le document ».
- Si le texte contient une référence manifestement erronée (ex. confusion de codes), signale-le sans inventer un numéro d'article.
- Anonymisation stricte : remplace les noms par [Mme X], [M. Y], [Société Z] et les montants par [montant].
Réponds uniquement par l'analyse rédigée, sans commentaires méta ni hypothèses."""
EXPERT_PROMPT_TEXT = """Tu es un juriste senior en droit du travail. Analyse les moyens de droit en français juridique, SANS listes ni numérotation. Mets en évidence : (i) qualification précise des faits, (ii) articulation des moyens principaux et subsidiaires, (iii) lien exact avec les références présentes DANS le document UNIQUEMENT (sinon « non précisé dans le document »), (iv) charge de la preuve et incidences procédurales si mentionnées, (v) conclusion motivée. Aucune invention, aucune hypothèse."""
SYSTEM_PROMPT = """Tu es juriste en droit du travail. Rédige une analyse argumentée en français juridique, sans listes à puces, en citant uniquement les fondements mentionnés dans le texte."""

def load_prompt_store() -> Dict[str, str]:
    default_store = {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT}
    if not os.path.exists(PROMPT_STORE_PATH):
        try:
            os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
            atomic_write_text(PROMPT_STORE_PATH, json.dumps(default_store, ensure_ascii=False, indent=2))
            return default_store
        except Exception as e:
            print(f"Attention: création store initial impossible: {e}")
            return default_store
    try:
        with open(PROMPT_STORE_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
        if not isinstance(store, dict) or not store:
            return default_store
        if DEFAULT_PROMPT_NAME not in store:
            store[DEFAULT_PROMPT_NAME] = DEFAULT_PROMPT_TEXT
            save_prompt_store(store)
        return store
    except Exception as e:
        print(f"Erreur chargement store: {e}")
        return default_store

def save_prompt_store(store: Dict[str, str]) -> Tuple[bool, str]:
    try:
        if not isinstance(store, dict):
            return False, "store n'est pas un dictionnaire"
        atomic_write_text(PROMPT_STORE_PATH, json.dumps(store, ensure_ascii=False, indent=2))
        return True, f"Enregistré dans : `{PROMPT_STORE_PATH}`"
    except Exception as e:
        return False, f"Échec d'enregistrement : {e}"

def sanitize_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"\s+", " ", name.strip())
    name = re.sub(r'[<>:"/\\\?\*]', "_", name)
    return name[:100]

# -------------------------
# Gestion fichiers prompt*.txt
# -------------------------
def get_prompt_files(directory: str) -> List[str]:
    try:
        if not os.path.exists(directory):
            return []
        out = []
        for fn in os.listdir(directory):
            if fn.startswith("prompt") and fn.endswith(".txt"):
                p = os.path.join(directory, fn)
                if os.path.isfile(p):
                    out.append(fn)
        return sorted(out)
    except Exception as e:
        print(f"Erreur lecture répertoire prompts : {e}")
        return []

def read_prompt_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        return f"Erreur lecture fichier {filepath}: {e}"

def concatenate_prompts(selected_files: list, directory: str) -> str:
    if not selected_files:
        return "Aucun fichier sélectionné."
    parts = ["=== PROMPT GÉNÉRAL COMPOSÉ ===\n"]
    for fn in selected_files:
        parts.append(f"--- {fn} ---")
        parts.append(read_prompt_file(os.path.join(directory, fn)))
        parts.append("")
    parts.append("=== FIN PROMPT COMPOSÉ ===")
    return "\n".join(parts)

# -------------------------
# Anonymisation texte
# -------------------------
class TextAnonymizer:
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.replacement_map: Dict[str, str] = {}
        self.counters = {'person': 0, 'company': 0, 'amount': 0, 'date': 0, 'address': 0}

    def anonymize_text(self, text: str) -> str:
        if not text:
            return text
        t = text
        t = self._anonymize_amounts(t)
        t = self._anonymize_dates(t)
        t = self._anonymize_names(t)
        t = self._anonymize_companies(t)
        t = self._anonymize_addresses(t)
        t = self._anonymize_phone_numbers(t)
        t = self._anonymize_emails(t)
        return t

    def _anonymize_amounts(self, text: str) -> str:
        patterns = [
            r'\b\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})?\s*(?:€|eur|euros?)\b',
            r'\b(?:€|eur|euros?)\s*\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})?\b',
            r'\b\d+(?:[.,]\d{2})?\s*(?:centimes?)\b',
            r'\b(?:la\s+somme\s+de|le\s+montant\s+de|pour\s+un\s+montant\s+de)\s+\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})?\s*(?:€|eur|euros?)\b',
        ]
        for p in patterns:
            text = re.sub(p, '[MONTANT_ANONYME]', text, flags=re.IGNORECASE)
        return text

    def _anonymize_dates(self, text: str) -> str:
        months = r'janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre'
        patterns = [
            r'\b\d{1,2}[/. -]\d{1,2}[/. -]\d{2,4}\b',
            rf'\b\d{1,2}\s+(?:{months})\s+\d{2,4}\b',
            rf'\b(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+\d{{1,2}}\s+(?:{months})\s+\d{{2,4}}\b',
            r'\b\d{2,4}[/. -]\d{1,2}[/. -]\d{1,2}\b',
        ]
        for p in patterns:
            text = re.sub(p, '[DATE_ANONYME]', text, flags=re.IGNORECASE)
        return text

    def _anonymize_names(self, text: str) -> str:
        name_patterns = [
            r'\b(?:M\.|Mme|Monsieur|Madame|Mlle|Mademoiselle)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-]+)*\b',
            r'\b[A-Z]{2,}\s+[A-Z][a-zÀ-ÖØ-öø-ÿ\-]+\b',
            r'\bde\s+[A-Z][a-zÀ-ÖØ-öø-ÿ\-]+\b',
            r'\b[A-Z][a-zÀ-ÖØ-öø-ÿ\-]+\s+né[e]?\s+[A-Z][a-zÀ-ÖØ-öø-ÿ\-]+\b',
        ]
        for pattern in name_patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                full = m.group(0)
                if full not in self.replacement_map:
                    self.counters['person'] += 1
                    self.replacement_map[full] = f"[PERSONNE_{self.counters['person']}]"
                text = text.replace(full, self.replacement_map[full])
        return text

    def _anonymize_companies(self, text: str) -> str:
        company_patterns = [
            r'\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ&\s\-]+\s+(?:SA|SARL|SAS|SASU|SNC|SCA|EURL|GIE|SCOP|SEM)\b',
            r'\b(?:SA|SARL|SAS|SASU|SNC|SCA|EURL|GIE|SCOP|SEM)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ&\s\-]+\b',
            r'\b(?:Société|Entreprise|Compagnie|Établissements?)\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ&\s\-]+\b',
            r'\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ&\s\-]+\s+(?:&\s*Cie|et\s+Fils|France|International|Groupe?)\b',
        ]
        for pattern in company_patterns:
            for m in re.finditer(pattern, text):
                full = m.group(0)
                if full not in self.replacement_map:
                    self.counters['company'] += 1
                    self.replacement_map[full] = f"[SOCIETE_{self.counters['company']}]"
                text = text.replace(full, self.replacement_map[full])
        return text

    def _anonymize_addresses(self, text: str) -> str:
        address_patterns = [
            r'\b\d{1,4}[\s,]+(?:rue|avenue|boulevard|bd|place|impasse|allée|allee|chemin|route)\s+[A-Za-zÀ-ÖØ-öø-ÿ\'\-\s]+\b',
            r'\b\d{5}\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-\s]+\b',
            r'\b(?:domicilié[e]?\s+)?(?:au|à|en)\s+\d{1,4}[\s,]+(?:rue|avenue|boulevard|bd|place|impasse|allée|allee|chemin|route)\s+[A-Za-zÀ-ÖØ-öø-ÿ\'\-\s]+\b',
        ]
        for p in address_patterns:
            text = re.sub(p, '[ADRESSE_ANONYME]', text, flags=re.IGNORECASE)
        return text

    def _anonymize_phone_numbers(self, text: str) -> str:
        phone_patterns = [
           r'\b0[1-9{4}\b',
            r'\b\+33[.\-\s]?1-9{4}\b',
        ]
        for p in phone_patterns:
            text = re.sub(p, '[TELEPHONE_ANONYME]', text)
        return text

    def _anonymize_emails(self, text: str) -> str:
        email_pattern = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
        return re.sub(email_pattern, '[EMAIL_ANONYME]', text)

    def get_anonymization_report(self) -> str:
        total = sum(self.counters.values()) + len(self.replacement_map)
        if total == 0:
            return "Aucune donnée sensible détectée dans le texte."
        parts = ["=== RAPPORT D'ANONYMISATION ==="]
        labels = {'person': 'Personnes', 'company': 'Sociétés', 'amount': 'Montants', 'date': 'Dates', 'address': 'Adresses'}
        parts.append(f"Total d'éléments anonymisés : {total}")
        for k, c in self.counters.items():
            if c > 0:
                parts.append(f"- {labels.get(k, k)} : {c}")
        if self.replacement_map:
            parts.append(f"- Noms/entités mappées : {len(self.replacement_map)}")
        parts.append("=" * 35)
        return "\n".join(parts)

def anonymize_legal_text(text: str, strict_mode: bool = True) -> Tuple[str, str]:
    a = TextAnonymizer(strict_mode=strict_mode)
    out = a.anonymize_text(text)
    return out, a.get_anonymization_report()

# -------------------------
# Nettoyage OCR
# -------------------------
def _normalize_unicode(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "–": "-", "—": "-", "…": "...",
        "“": '"', "”": '"', "’": "'",
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def _strip_headers_footers(pages_lines: List[List[str]]) -> List[List[str]]:
    if not pages_lines or len(pages_lines) < 2:
        return pages_lines
    counter = Counter()
    for lines in pages_lines:
        if not lines:
            continue
        sample = lines[:3] + (lines[-3:] if len(lines) > 6 else [])
        for line in sample:
            cl = line.strip()
            if len(cl) >= 3 and not cl.isdigit():
                counter.update([cl])
    min_occ = max(2, len(pages_lines) // 3)
    repeated = {l for l, c in counter.items() if c >= min_occ}
    cleaned = []
    for lines in pages_lines:
        cleaned.append([ln for ln in lines if ln.strip() not in repeated])
    return cleaned

def smart_clean(text: str, pages_texts: Optional[List[str]] = None) -> str:
    if not text:
        return text
    text = _normalize_unicode(text)
    if pages_texts:
        pages_lines = [t.splitlines() for t in pages_texts]
        pages_lines = _strip_headers_footers(pages_lines)
        text = "\n".join("\n".join(lines) for lines in pages_lines)

    # Raccord mots coupés
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Suppression en-têtes/pieds & bruits
    page_patterns = [
        r"(?im)^\s*page\s*[:\-]?\s*\d+\s*(?:/|sur|de|of)\s*\d+\s*$",
        r"(?im)^\s*p(?:age)?\.?\s*[:\-]?\s*\d+\s*$",
        r"(?im)^\s*\d+\s*/\s*\d+\s*$",
        r"(?im)^\s*[-–—]+\s*\d+\s*[-–—]+\s*$",
        r"(?im)^\s*\d{1,3}\s*$",
        r"(?im)^\s*page\s+n[°º]\s*\d+\s*$",
        r"(?im)^\s*page\s*[-–—]+\s*\d+\s*[-–—]+\s*$",
    ]
    for p in page_patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE)

    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[–—\-]{2,}\s*$", "", text)
    text = re.sub(r"(?m)^[\-\–—\s]*\s*$", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"(?<![.\!?:;])\n(?!\n)(?=[a-zàâäçéèêëîïôöùûüÿñ\(\,;:\.\-])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Nettoyage lignes orphelines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        s = line.strip()
        if s and not re.match(r'^[\d\s\-\–—\.]{1,10}$', s):
            cleaned_lines.append(line)
        elif not s:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def calculate_ocr_stats(text: str) -> str:
    if not text:
        return "Texte vide"
    lines = text.split("\n")
    words = len(text.split())
    chars = len(text)
    non_empty = len([l for l in lines if l.strip()])
    return f"{chars:,} caractères \n {words:,} mots \n {non_empty} lignes non vides \n {len(lines)} lignes totales"

# -------------------------
# Cache OCR
# -------------------------
def get_pdf_hash(pdf_path: str) -> Optional[str]:
    if not pdf_path or not os.path.exists(pdf_path):
        return None
    try:
        h = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"Erreur calcul hash PDF : {e}")
        return None

def _cache_dir() -> str:
    d = os.path.join(_script_dir(), "cache_ocr")
    os.makedirs(d, exist_ok=True)
    return d

def _cache_path(pdf_hash: str, nettoyer: bool) -> str:
    suffix = "_clean" if nettoyer else "_raw"
    return os.path.join(_cache_dir(), f"ocr_{pdf_hash}{suffix}.pkl")

def save_ocr_cache(pdf_hash: str, nettoyer: bool, data: dict) -> bool:
    try:
        with open(_cache_path(pdf_hash, nettoyer), "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde cache : {e}")
        return False

def load_ocr_cache(pdf_hash: str, nettoyer: bool) -> Optional[dict]:
    try:
        p = _cache_path(pdf_hash, nettoyer)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Erreur chargement cache : {e}")
    return None

def clear_ocr_cache() -> int:
    try:
        d = _cache_dir()
        if not os.path.exists(d):
            return 0
        c = 0
        for fn in os.listdir(d):
            if fn.startswith("ocr_") and fn.endswith(".pkl"):
                os.remove(os.path.join(d, fn))
                c += 1
        return c
    except Exception as e:
        print(f"Erreur nettoyage cache : {e}")
        return 0

# -------------------------
# Contrôle Qualité
# -------------------------
def _qc_extract_references(text: str) -> Set[str]:
    if not text:
        return set()
    refs: Set[str] = set()
    t = text.lower()
    for m in re.finditer(r"\b(art\.|article)\s+([lrd]\s*[.\-]?\s*\d+[0-9\-\._]*)", t):
        refs.add(m.group(0).strip())
    for m in re.finditer(r"\bcode\s+du\s+(travail|procédure\s+civile|civil)\b", t):
        refs.add(m.group(0).strip())
    return refs

def qc_check_references(ocr_text: str, analysis_text: str) -> str:
    if not analysis_text:
        return "Contrôle qualité : contenu d'analyse vide."
    ocr_refs = _qc_extract_references(ocr_text or "")
    out_refs = _qc_extract_references(analysis_text or "")
    if not out_refs:
        return "Contrôle qualité : aucune référence détectée dans l'analyse."
    missing = sorted(r for r in out_refs if r not in ocr_refs)
    if not missing:
        return f"Contrôle qualité : toutes les références citées ({len(out_refs)}) apparaissent dans le texte OCR."
    rep = ["Contrôle qualité : références citées par l'analyse MAIS absentes du texte OCR :"]
    rep += [f" • {r}" for r in missing]
    rep.append(f"\nStatistiques : {len(out_refs)} références totales, {len(missing)} manquantes")
    rep.append("\nRecommandation : remplacer par « non précisé dans le document » ou reformuler.")
    return "\n".join(rep)

# -------------------------
# API Ollama
# -------------------------
def generate_with_ollama(model: str, prompt_text: str, full_text: str,
                         num_ctx: int, num_predict: int, temperature: float = 0.2,
                         timeout: int = 900, api_base: Optional[str] = None) -> str:
    base = (api_base or get_saved_api_url() or "http://localhost:11434").strip().rstrip("/")
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": f"Texte à analyser :\n{full_text}",
        "system": f"{SYSTEM_PROMPT}\n\n{prompt_text}",
        "stream": True,
        "options": {"num_ctx": int(num_ctx), "num_predict": int(num_predict), "temperature": float(temperature)},
    }
    try:
        r = requests.post(url, json=payload, stream=True, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return "❌ Erreur : Impossible de se connecter à Ollama. Vérifiez qu'Ollama est démarré (ollama serve)."
    except requests.exceptions.Timeout:
        return f"❌ Erreur : Délai dépassé ({timeout}s)."
    except Exception as e:
        return f"❌ Erreur de connexion Ollama : {e}"

    if r.status_code != 200:
        err = r.text
        if "system memory" in err.lower() and "available" in err.lower():
            return ("❌ MÉMOIRE INSUFFISANTE : Le modèle nécessite plus de RAM.\n"
                    "Solutions :\n"
                    "1. Utilisez un modèle plus léger (mistral:7b-instruct, deepseek-coder:latest)\n"
                    "2. Fermez d'autres applications\n"
                    "3. Redémarrez Ollama : 'ollama serve'\n\n"
                    f"Erreur complète : {err}")
        return f"❌ Erreur HTTP {r.status_code} : {err}"

    parts: List[str] = []
    try:
        for line in r.iter_lines():
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

    out = "".join(parts).strip()
    return out if out else "❌ Aucune réponse reçue (flux vide)."


def get_ollama_models(url: str) -> list:
    """
    Interroge l'API Ollama pour récupérer la liste des modèles disponibles.
    Retourne une liste de noms de modèles ou une liste vide en cas d'erreur.
    """
    try:
        response = requests.get(f"{url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        print(f"Erreur lors de la récupération des modèles Ollama : {e}")
        return []




# -------------------------
# Sauvegarde résultats
# -------------------------
def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\\?\*]', '_', name or "")
    name = re.sub(r'\s+', '_', name)
    return name[:100]

def save_analysis_result(pdf_name: str, model: str, analysis_text: str,
                         metadata: dict, format_type: str = "txt") -> Tuple[bool, str]:
    try:
        base_dir = _script_dir()
        base_name = sanitize_filename(os.path.splitext(os.path.basename(pdf_name))[0])
        model_clean = sanitize_filename(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"result_{base_name}_{model_clean}_{timestamp}.{format_type}"
        out_path = os.path.join(base_dir, out_name)

        content_parts = [
            "=" * 80,
            "ANALYSE JURIDIQUE AUTOMATISÉE",
            "=" * 80,
            f"Document source : {os.path.basename(pdf_name)}",
            f"Date de traitement : {metadata.get('timestamp', 'N/A')}",
            f"Modèle utilisé : {metadata.get('model', 'N/A')}",
            f"Mode d'analyse : {metadata.get('mode', 'N/A')}",
            f"Profil d'inférence : {metadata.get('profil', 'N/A')}",
            f"Contexte (tokens) : {metadata.get('num_ctx', 'N/A')}",
            f"Longueur max sortie : {metadata.get('max_tokens', 'N/A')}",
            f"Température : {metadata.get('temperature', 'N/A')}",
            f"Temps de traitement : {metadata.get('processing_time', 'N/A')}s",
            f"Nettoyage avancé : {'Oui' if metadata.get('nettoyer', False) else 'Non'}",
            "",
            "PROMPT UTILISÉ :",
            "-" * 40,
        ]
        ptxt = metadata.get("prompt", "")
        if len(ptxt) > 500:
            ptxt = ptxt[:500] + "... [tronqué]"
        content_parts.append(ptxt)
        content_parts += ["", "RÉSULTAT DE L'ANALYSE :", "=" * 40, analysis_text]
        content = "\n".join(content_parts)

        if format_type.lower() == "rtf":
            rtf = "{\\rtf1\\ansi\\deff0 {\\fonttbl {\\f0 Times New Roman;}} \\f0\\fs24 " + content.replace("\n", "\\par ") + "}"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(rtf)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)
        return True, f"Résultat sauvegardé : {out_name}"
    except Exception as e:
        return False, f"Erreur sauvegarde : {e}"

# -------------------------
# Pipelines OCR / Analyse
# -------------------------
def do_ocr_only(pdf_path: str, nettoyer: bool, force_ocr: bool = False) -> Tuple[str, str, str]:
    if not pdf_path:
        return "❌ Aucun fichier PDF fourni.", "", ""
    if not os.path.exists(pdf_path):
        return "❌ Fichier PDF introuvable.", "", ""
    try:
        pdf_hash = get_pdf_hash(pdf_path)
        if pdf_hash and not force_ocr:
            cached = load_ocr_cache(pdf_hash, nettoyer)
            if cached:
                print("✅ OCR depuis cache.")
                return "✅ OCR terminé. Texte prêt pour analyse.", cached["stats"], cached["preview"]

        print(f"🔄 Conversion PDF : {pdf_path}")
        images = convert_from_path(pdf_path)
        raw_pages: List[str] = []
        total_pages = len(images)
        print(f"📄 {total_pages} page(s) à traiter…")
        for i, image in enumerate(images, 1):
            print(f"🔍 OCR page {i}/{total_pages}…")
            page_text = pytesseract.image_to_string(image, lang="fra")
            raw_pages.append(page_text or "")

        if nettoyer:
            print("🧹 Nettoyage du texte OCR…")
            cleaned_pages = [_normalize_unicode(t) for t in raw_pages]
            preview = smart_clean("\n".join(cleaned_pages), pages_texts=cleaned_pages)
        else:
            preview = "\n".join(raw_pages).strip()

        stats = calculate_ocr_stats(preview)
        if pdf_hash:
            save_ocr_cache(pdf_hash, nettoyer, {
                "preview": preview, "stats": stats,
                "total_pages": total_pages, "timestamp": str(os.path.getmtime(pdf_path))
            })

        if not preview.strip():
            return "❌ Aucun texte détecté lors de l'OCR.", stats, preview
        return "✅ OCR terminé. Texte prêt pour analyse.", stats, preview
    except Exception as e:
        traceback.print_exc()
        return f"❌ Erreur lors de l'OCR : {e}", "Erreur - Impossible de calculer les statistiques", ""

def do_analysis_only(text_ocr: str, modele: str, profil: str, max_tokens_out: int,
                     prompt_text: str, mode_analysis: str, comparer: bool,
                     api_base: Optional[str] = None) -> Tuple[str, str, str, dict]:
    if not text_ocr or not text_ocr.strip():
        return "❌ Aucun texte OCR disponible pour l'analyse.", "", "", {}
    start = time.time()
    try:
        text_len = len(text_ocr)
        est_tokens = text_len // 4
        print(f"📊 Longueur du texte : {text_len:,} caractères (≈{est_tokens:,} tokens)")
        if est_tokens > 20000:
            print("⚠️ Très volumineux : profil 'Maxi' recommandé.")
        elif est_tokens > 10000:
            print("⚠️ Volumineux : profil 'Confort' ou 'Maxi' recommandé.")

        profiles = {
            "Rapide": {"num_ctx": 8192, "temperature": 0.2},
            "Confort": {"num_ctx": 16384, "temperature": 0.2},
            "Maxi": {"num_ctx": 32768, "temperature": 0.2},
        }
        base = profiles.get(profil, profiles["Confort"])
        num_ctx = base["num_ctx"]
        temperature = base["temperature"]

        if est_tokens > int(num_ctx * 0.8):
            print(f"⚠️ ATTENTION : {est_tokens:,} tokens vs contexte {num_ctx:,}.")

        main_prompt = EXPERT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else prompt_text
        print(f"🧠 Génération avec {modele} (mode {mode_analysis})…")
        analyse = generate_with_ollama(
            modele, main_prompt, text_ocr,
            num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature,
            api_base=api_base
        )

        processing_time = round(time.time() - start, 2)
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": modele,
            "mode": mode_analysis,
            "profil": profil,
            "num_ctx": num_ctx,
            "max_tokens": max_tokens_out,
            "temperature": temperature,
            "processing_time": processing_time,
            "prompt": main_prompt
        }

        params_info = (
            f"\n=== PARAMÈTRES D'ANALYSE ===\n"
            f"Modèle : {modele}\n"
            f"Mode : {mode_analysis}\n"
            f"Profil : {profil} (contexte: {num_ctx}, température: {temperature})\n"
            f"Longueur max : {max_tokens_out} tokens\n"
            f"Temps de traitement : {processing_time}s\n"
            f"Date : {metadata['timestamp']}\n"
            f"========================\n"
        )
        analyse_with_params = params_info + analyse

        qc_or_compare = ""
        analyse_alt = ""
        if (mode_analysis or "").lower().startswith("expert"):
            print("🔍 Contrôle qualité…")
            qc_or_compare = qc_check_references(text_ocr, analyse)

        if comparer:
            print("🪞 Génération comparative…")
            alt_prompt = DEFAULT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else EXPERT_PROMPT_TEXT
            analyse_alt = generate_with_ollama(
                modele, alt_prompt, text_ocr,
                num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature,
                api_base=api_base
            )

            def _summary_diff(a: str, b: str) -> str:
                if not a or not b or a.startswith("❌") or b.startswith("❌"):
                    return "Comparaison impossible (erreur ou contenu vide)."
                wa = set(w.lower() for w in re.findall(r"\b\w{4,}\b", a))
                wb = set(w.lower() for w in re.findall(r"\b\w{4,}\b", b))
                if not wa and not wb:
                    return "Aucun mot significatif détecté."
                inter = len(wa & wb)
                union = len(wa | wb) or 1
                jac = inter / union
                return (f"Similarité lexicale : {jac:.2%}\n"
                        f"Mots uniques analyse 1 : {len(wa)}\n"
                        f"Mots uniques analyse 2 : {len(wb)}\n"
                        f"Mots communs : {inter}")
            comp = _summary_diff(analyse, analyse_alt)
            qc_or_compare = (qc_or_compare + "\n\n" + comp).strip() if qc_or_compare else comp

        return analyse_with_params, analyse_alt, qc_or_compare, metadata
    except Exception as e:
        traceback.print_exc()
        return f"❌ Erreur lors de l'analyse : {e}", "", "", {}

# -------------------------
# UI Gradio
# -------------------------
def build_ui(initial_api_url: Optional[str] = None):
    initial_api_url = (initial_api_url or get_saved_api_url()).strip().rstrip("/")
    models_list = get_ollama_models(initial_api_url)
    store = load_prompt_store()
    prompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
    script_name = os.path.basename(__file__) if "__file__" in globals() else "app.py"

    with gr.Blocks(title=f"{script_name} - OCR Juridique + Ollama") as demo:
        # Barre URL + refresh
        with gr.Row():
            api_url_input = gr.Textbox(label="URL API Ollama", value=initial_api_url, interactive=True)
            refresh_models_btn = gr.Button("Rafraîchir modèles", variant="secondary")
        api_info = gr.Markdown("")

        gr.Markdown("## OCR structuré + Analyse juridique (Ollama)")
        gr.Markdown(f"**Fichier des prompts** : `{PROMPT_STORE_PATH}`")

        # Fichier + options OCR
        with gr.Row():
            pdf_file = gr.File(label="Uploader un PDF scanné", file_types=[".pdf"])
            nettoyer = gr.Checkbox(label="Nettoyage avancé", value=True)
        with gr.Row():
            force_ocr = gr.Checkbox(label="Forcer nouveau OCR (ignorer le cache)", value=False)
            clear_cache_btn = gr.Button("Vider le cache OCR", variant="secondary")
            cache_info = gr.Markdown("")

        # Modèle + profil
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

        # Prompts (JSON)
        gr.Markdown("### Prompt — gestion persistante (JSON)")
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
        info_box = gr.Markdown("")

        # Compositeur fichiers prompt*.txt
        gr.Markdown("### Compositeur de prompts (fichiers prompt*.txt)")
        files_checklist = gr.CheckboxGroup(
            label="Fichiers prompt*.txt disponibles dans ./prompts/",
            choices=get_prompt_files(PROMPT_FILES_DIR),
            value=[]
        )
        with gr.Row():
            refresh_files_btn = gr.Button("Actualiser fichiers", variant="secondary")
            compose_btn = gr.Button("Composer prompt général", variant="primary")
            copy_to_main_btn = gr.Button("Copier vers prompt principal", variant="secondary")
        composed_prompt_box = gr.Textbox(
            label="Prompt général composé",
            lines=15,
            interactive=False,
            show_copy_button=True,
            placeholder="Le prompt composé apparaîtra ici..."
        )

        # Actions principales
        with gr.Row():
            ocr_btn = gr.Button("1. OCR seulement", variant="secondary")
            analyze_btn = gr.Button("2. Analyser", variant="primary")
            full_btn = gr.Button("OCR + Analyse", variant="primary")

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
            with gr.Tab("Texte OCR"):
                ocr_stats = gr.Textbox(label="Statistiques", lines=2, interactive=False)
                preview_box = gr.Textbox(label="Texte OCR extrait", interactive=False, show_copy_button=True, lines=25)

        # States
        prompts_state = gr.State(value=store)
        current_name = gr.State(value=DEFAULT_PROMPT_NAME)
        current_ocr_text = gr.State(value="")
        current_pdf_path = gr.State(value="")
        analysis_metadata = gr.State(value={})
        api_url_state = gr.State(value=initial_api_url)

        # === Callbacks ===
        def clear_cache():
            count = clear_ocr_cache()
            return gr.Markdown.update(value=("Cache vidé : " + str(count) + " fichier(s) supprimé(s)") if count > 0 else "Cache déjà vide")

        def on_pdf_upload(pdf_path, nettoyer):
            if not pdf_path:
                return "", "", "", ""
            try:
                pdf_hash = get_pdf_hash(pdf_path)
                if pdf_hash:
                    ocr_data = load_ocr_cache(pdf_hash, nettoyer)
                    if ocr_data:
                        return ocr_data['stats'], ocr_data['preview'], ocr_data['preview'], pdf_path
                return "", "", "", pdf_path
            except Exception:
                return "", "", "", pdf_path

        def launch_ocr_only(pdf_path, nettoyer, force_ocr):
            try:
                message, stats, preview = do_ocr_only(pdf_path, nettoyer, force_ocr)
                return (message, "", "", stats, preview, preview, pdf_path if pdf_path else "")
            except Exception as e:
                traceback.print_exc()
                return f"❌ Erreur OCR : {str(e)}", "", "", "Erreur", "", "", ""

        def launch_analysis_only(ocr_text, pdf_path, modele, profil, max_tokens_out,
                                prompt_text, mode_analysis, comparer, nettoyer, api_url):
            try:
                if not ocr_text and not pdf_path:
                    return "❌ Aucun texte OCR ni fichier PDF disponible.", "", "", "", "", "", {}
                if not ocr_text and pdf_path:
                    message, stats, preview = do_ocr_only(pdf_path, nettoyer, False)
                    if "❌" in message:
                        return message, "", "", stats, preview, preview, {}
                    ocr_text = preview

                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    ocr_text, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer,
                    api_base=api_url
                )
                metadata['nettoyer'] = nettoyer
                stats = calculate_ocr_stats(ocr_text)
                return analyse, analyse_alt, qc_compare, stats, ocr_text, ocr_text, metadata
            except Exception as e:
                traceback.print_exc()
                return f"❌ Erreur analyse : {str(e)}", "", "", "Erreur", "", "", {}

        def launch_full_pipeline(pdf_path, nettoyer, force_ocr, modele, profil, max_tokens_out,
                                 prompt_text, mode_analysis, comparer, api_url):
            try:
                if not pdf_path:
                    return "❌ Aucun fichier PDF fourni.", "", "", "", "", "", {}
                message, stats, ocr_text = do_ocr_only(pdf_path, nettoyer, force_ocr)
                if "❌" in message:
                    return message, "", "", stats, ocr_text, ocr_text, {}
                analyse, analyse_alt, qc_compare, metadata = do_analysis_only(
                    ocr_text, modele, profil, max_tokens_out, prompt_text, mode_analysis, comparer,
                    api_base=api_url
                )
                metadata['nettoyer'] = nettoyer
                return analyse, analyse_alt, qc_compare, stats, ocr_text, ocr_text, metadata
            except Exception as e:
                traceback.print_exc()
                return f"❌ Erreur pipeline : {str(e)}", "", "", "Erreur", "", "", {}

        def save_analysis_file(pdf_path, analysis_text, metadata, format_type):
            if not analysis_text or not pdf_path:
                return "❌ Aucun résultat à sauvegarder"
            model = metadata.get('model', 'unknown')
            ok, msg = save_analysis_result(pdf_path, model, analysis_text, metadata, format_type)
            return f"✅ {msg}" if ok else f"❌ {msg}"

        def on_select(name, store):
            try:
                name = sanitize_name(name) or DEFAULT_PROMPT_NAME
                if name not in store:
                    name = DEFAULT_PROMPT_NAME
                text = store.get(name, DEFAULT_PROMPT_TEXT)
                return gr.Textbox.update(value=text), gr.Textbox.update(value=name), name, gr.Markdown.update(value=f"Sélectionné : **{name}**")
            except Exception as e:
                return gr.Textbox.update(value=DEFAULT_PROMPT_TEXT), gr.Textbox.update(value=DEFAULT_PROMPT_NAME), DEFAULT_PROMPT_NAME, gr.Markdown.update(value=f"Erreur : {e}")

        def on_save(current_name_val, text, store):
            try:
                name = sanitize_name(current_name_val) or DEFAULT_PROMPT_NAME
                if not text.strip():
                    return gr.Dropdown.update(), gr.Markdown.update(value="Le contenu ne peut pas être vide."), store, current_name_val
                store[name] = text.strip()
                ok, msg = save_prompt_store(store)
                if ok:
                    names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
                    return gr.Dropdown.update(choices=names, value=name), gr.Markdown.update(value=f"✅ Enregistré : {msg}"), store, name
                else:
                    return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ {msg}"), store, current_name_val
            except Exception as e:
                return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Erreur : {e}"), store, current_name_val

        def refresh_prompt_files():
            files = get_prompt_files(PROMPT_FILES_DIR)
            return gr.CheckboxGroup.update(choices=files, value=[])

        def compose_selected_prompts(selected_files):
            if not selected_files:
                return "Aucun fichier sélectionné."
            return concatenate_prompts(selected_files, PROMPT_FILES_DIR)

        def copy_composed_to_main(composed_text):
            if not composed_text or "Aucun fichier" in composed_text:
                return "", "Aucun prompt composé à copier."
            lines = composed_text.split('\n')
            clean = []
            for line in lines:
                if line.startswith("=== ") or line.startswith("--- "):
                    continue
                clean.append(line)
            clean_prompt = '\n'.join(clean).strip()
            if not clean_prompt:
                return "", "Prompt vide après nettoyage."
            return clean_prompt, f"✅ Prompt composé copié ({len(clean_prompt)} caractères)."

        def on_api_url_change(new_url):
            url = (new_url or "").strip().rstrip("/")
            if not url:
                restored = get_saved_api_url()
                return gr.State.update(value=restored), gr.Markdown.update(value="❌ URL vide : retour à la valeur sauvegardée.")
            set_saved_api_url(url)
            return gr.State.update(value=url), gr.Markdown.update(value=f"✅ URL mise à jour : **{url}**")

        def do_refresh_models(api_url, current_model):
            url = (api_url or get_saved_api_url()).strip().rstrip("/")
            models = get_ollama_models(url)
            if not models:
                return (
                    gr.Dropdown.update(),
                    gr.Markdown.update(value=f"❌ Aucun modèle trouvé à **{url}**"),
                    gr.State.update(value=url)
                )
            # Choisir un modèle par défaut
            if current_model in models:
                new_value = current_model
            elif "mistral:7b-instruct" in models:
                new_value = "mistral:7b-instruct"
            elif "deepseek-coder:latest" in models:
                new_value = "deepseek-coder:latest"
            elif "mistral:latest" in models:
                new_value = "mistral:latest"
            else:
                new_value = models[0]
            return (
                gr.Dropdown.update(choices=models, value=new_value),
                gr.Markdown.update(value=f"🔁 Modèles actualisés depuis **{url}** ({len(models)})"),
                gr.State.update(value=url)
            )



        # Connexions
        pdf_file.change(fn=on_pdf_upload, inputs=[pdf_file, nettoyer], outputs=[ocr_stats, preview_box, current_ocr_text, current_pdf_path])
        ocr_btn.click(fn=launch_ocr_only, inputs=[pdf_file, nettoyer, force_ocr], outputs=[analysis_box, analysis_alt_box, compare_box, ocr_stats, preview_box, current_ocr_text, current_pdf_path])
        analyze_btn.click(fn=launch_analysis_only, inputs=[current_ocr_text, current_pdf_path, modele, profil, max_tokens_out, prompt_box, mode_analysis, comparer, nettoyer, api_url_input], outputs=[analysis_box, analysis_alt_box, compare_box, ocr_stats, preview_box, current_ocr_text, analysis_metadata])
        full_btn.click(fn=launch_full_pipeline, inputs=[pdf_file, nettoyer, force_ocr, modele, profil, max_tokens_out, prompt_box, mode_analysis, comparer, api_url_input], outputs=[analysis_box, analysis_alt_box, compare_box, ocr_stats, preview_box, current_ocr_text, analysis_metadata])
        clear_cache_btn.click(fn=clear_cache, outputs=[cache_info])

        prompt_selector.change(fn=on_select, inputs=[prompt_selector, prompts_state], outputs=[prompt_box, prompt_name_box, current_name, info_box])
        save_btn.click(fn=on_save, inputs=[current_name, prompt_box, prompts_state], outputs=[prompt_selector, info_box, prompts_state, current_name])

        refresh_files_btn.click(fn=refresh_prompt_files, outputs=[files_checklist])
        compose_btn.click(fn=compose_selected_prompts, inputs=[files_checklist], outputs=[composed_prompt_box])
        copy_to_main_btn.click(fn=copy_composed_to_main, inputs=[composed_prompt_box], outputs=[prompt_box, info_box])

        save_result_btn.click(fn=save_analysis_file, inputs=[current_pdf_path, analysis_box, analysis_metadata, save_format], outputs=[save_status])

        api_url_input.submit(fn=on_api_url_change, inputs=[api_url_input], outputs=[api_url_state, api_info])
        #refresh_models_btn.click(fn=do_refresh_models, inputs=[api_url_input, modele], outputs=[modele, api_info, api_url_state])
        refresh_models_btn.click( fn=do_refresh_models, inputs=[api_url_input, modele], outputs=[modele, api_info, api_url_state])

        gr.Markdown("""\
### Guide d'utilisation
**Flux recommandé :**
1. Uploader un PDF (le cache OCR est chargé si disponible)
2. Cliquer **OCR seulement** (met en cache si absent)
3. Cliquer **Analyser** (ou **OCR + Analyse**)
4. **Sauvegarder** le résultat

**Modèles recommandés (selon RAM) :**
- 8 GB ou moins : `mistral:7b-instruct`, `deepseek-coder:latest`
- 12+ GB : `llama3:latest`, `mistral:latest`
- 16+ GB : `llama3.1:8b-instruct-q5_K_M`

**Prérequis** : Tesseract + Poppler installés ; Ollama démarré (`ollama serve`)
**Répertoires** : Cache `./cache_ocr/` — Prompts `./prompts/`
""")
    return demo

# -------------------------
# Entrée principale (CLI)
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="OCR + Analyse juridique avec Ollama")
    parser.add_argument('--list-models', action='store_true', help='Lister les modèles Ollama disponibles')
    parser.add_argument('--gui', action='store_true', help="Lancer l'interface graphique (défaut)")
    parser.add_argument('--host', default='127.0.0.1', help="Adresse d'écoute (défaut: 127.0.0.1)")
    parser.add_argument('--port', type=int, help="Port d'écoute (défaut: auto)")
    parser.add_argument('--api-url', default=None, help="URL de l'API Ollama (ex: http://localhost:11434)")
    args = parser.parse_args()

    if args.list_models:
        print("Modèles Ollama disponibles :")
        try:
            models = get_ollama_models(args.api_url)
            for i, model in enumerate(models, 1):
                print(f" {i}. {model}")
        except Exception as e:
            print(f"Erreur: {e}")
        return

    print("🚀 Démarrage de l'interface OCR Juridique…")
    print(f"📁 Répertoire des prompts : {PROMPT_STORE_DIR}")
    try:
        initial_url = (args.api_url or get_saved_api_url()).strip().rstrip("/")
        _ = get_ollama_models(initial_url)  # ping initial (logs console)
        app = build_ui(initial_api_url=initial_url)
        launch_kwargs = {'server_name': args.host, 'share': False, 'inbrowser': True}
        if args.port:
            launch_kwargs['server_port'] = args.port
        app.launch(**launch_kwargs)
    except Exception as e:
        print(f"❌ Erreur : {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

