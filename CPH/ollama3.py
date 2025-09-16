#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR structuré + Analyse juridique (Ollama) - VERSION AVEC CACHE
- Nettoyage amélioré des numéros de page
- Gestion prompts JSON corrigée
- Gestion d'erreurs renforcée
- OCR indépendant de l'analyse automatique
- Système de cache OCR intelligent
"""

import os
import json
import re
import unicodedata
import hashlib
import pickle
from collections import Counter
from typing import Dict, Tuple

import gradio as gr
import requests
from pdf2image import convert_from_path
import pytesseract
import traceback

# =======================
#  PARAMÈTRES & PROMPTS
# =======================

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

def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def _default_store_dir() -> str:
    # Utiliser ./prompts au niveau du script
    script_dir = _script_dir()
    prompts_dir = os.path.join(script_dir, "prompts")
    
    try:
        os.makedirs(prompts_dir, exist_ok=True)
        # Test d'écriture
        test_path = os.path.join(prompts_dir, ".write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return prompts_dir
    except Exception as e:
        print(f"Erreur création répertoire ./prompts : {e}")
        # Fallback vers le répertoire du script
        return script_dir

PROMPT_STORE_DIR = _default_store_dir()
PROMPT_STORE_PATH = os.path.join(PROMPT_STORE_DIR, "prompts_store.json")
PROMPT_FILES_DIR = PROMPT_STORE_DIR

# =================================
#  SYSTÈME DE CACHE OCR
# =================================

def get_pdf_hash(pdf_path: str) -> str:
    """Calcule un hash du PDF pour identifier les fichiers de manière unique"""
    try:
        with open(pdf_path, 'rb') as f:
            hash_obj = hashlib.md5()
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"Erreur calcul hash PDF : {e}")
        return None

def get_cache_path(pdf_hash: str, nettoyer: bool) -> str:
    """Génère le chemin du fichier cache"""
    cache_dir = os.path.join(PROMPT_STORE_DIR, "cache_ocr")
    os.makedirs(cache_dir, exist_ok=True)
    
    suffix = "_clean" if nettoyer else "_raw"
    return os.path.join(cache_dir, f"ocr_{pdf_hash}{suffix}.pkl")

def save_ocr_cache(pdf_hash: str, nettoyer: bool, ocr_data: dict) -> bool:
    """Sauvegarde le résultat OCR en cache"""
    try:
        cache_path = get_cache_path(pdf_hash, nettoyer)
        with open(cache_path, 'wb') as f:
            pickle.dump(ocr_data, f)
        print(f"OCR mis en cache : {cache_path}")
        return True
    except Exception as e:
        print(f"Erreur sauvegarde cache : {e}")
        return False

def load_ocr_cache(pdf_hash: str, nettoyer: bool) -> dict:
    """Charge le résultat OCR depuis le cache"""
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
    """Nettoie le cache OCR et retourne le nombre de fichiers supprimés"""
    try:
        cache_dir = os.path.join(PROMPT_STORE_DIR, "cache_ocr")
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

# =================================
#  GESTIONNAIRE DE FICHIERS PROMPTS TXT
# =================================

def get_prompt_files(directory: str) -> list:
    """Récupère la liste des fichiers prompt*.txt dans le répertoire"""
    try:
        if not os.path.exists(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            if filename.startswith("prompt") and filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    files.append(filename)
        
        return sorted(files)
    except Exception as e:
        print(f"Erreur lecture répertoire prompts : {e}")
        return []

def read_prompt_file(filepath: str) -> str:
    """Lit le contenu d'un fichier prompt"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return f"Erreur lecture fichier {filepath}: {e}"

def concatenate_prompts(selected_files: list, directory: str) -> str:
    """Concatène les prompts sélectionnés"""
    if not selected_files:
        return "Aucun fichier sélectionné."
    
    result_parts = []
    result_parts.append("=== PROMPT GÉNÉRAL COMPOSÉ ===\n")
    
    for filename in selected_files:
        filepath = os.path.join(directory, filename)
        content = read_prompt_file(filepath)
        
        result_parts.append(f"--- {filename} ---")
        result_parts.append(content)
        result_parts.append("")
    
    result_parts.append("=== FIN PROMPT COMPOSÉ ===")
    return "\n".join(result_parts)

# =================================
#  GESTIONNAIRE DE PROMPTS (JSON)
# =================================

def atomic_write_text(path: str, data: str) -> None:
    """Écriture atomique avec gestion d'erreur améliorée"""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass
        raise

def load_prompt_store() -> Dict[str, str]:
    """Charge le fichier JSON des prompts"""
    default_store = {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT}
    
    if not os.path.exists(PROMPT_STORE_PATH):
        try:
            os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
            atomic_write_text(PROMPT_STORE_PATH, json.dumps(default_store, ensure_ascii=False, indent=2))
            return default_store
        except Exception as e:
            print(f"Erreur création store initial: {e}")
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
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Erreur chargement store: {e}")
        return default_store

def save_prompt_store(store: Dict[str, str]) -> Tuple[bool, str]:
    """Sauvegarde le store"""
    try:
        if not isinstance(store, dict):
            return False, "Erreur: store n'est pas un dictionnaire"
        
        os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
        json_data = json.dumps(store, ensure_ascii=False, indent=2)
        atomic_write_text(PROMPT_STORE_PATH, json_data)
        
        return True, f"Enregistré dans : `{PROMPT_STORE_PATH}`"
        
    except Exception as e:
        return False, f"Échec d'enregistrement : {e}"

def sanitize_name(name: str) -> str:
    """Nettoie et valide un nom de prompt"""
    if not name:
        return ""
    name = re.sub(r"\s+", " ", name.strip())
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:100]

# ================
#  OCR & NETTOYAGE
# ================

def _normalize_unicode(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ï¬": "fi", "ï¬‚": "fl", "ï¬€": "ff", "ï¬ƒ": "ffi", "ï¬„": "ffl",
        "'": "'", "'": "'", """: '"', """: '"',
        "—": "-", "–": "-", "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def _strip_headers_footers(pages_lines):
    """Détecte/supprime en-têtes/pieds répétés"""
    if not pages_lines or len(pages_lines) < 2:
        return pages_lines
        
    counter = Counter()
    
    for lines in pages_lines:
        if not lines:
            continue
        sample = lines[:3] + (lines[-3:] if len(lines) > 6 else [])
        
        for line in sample:
            clean_line = line.strip()
            if len(clean_line) >= 3 and not clean_line.isdigit():
                counter.update([clean_line])
    
    min_occurrences = max(2, len(pages_lines) // 3)
    repeated = {l for l, c in counter.items() if c >= min_occurrences}
    
    cleaned_pages = []
    for lines in pages_lines:
        new_lines = [line for line in lines if line.strip() not in repeated]
        cleaned_pages.append(new_lines)
    
    return cleaned_pages

def smart_clean(text: str, pages_texts=None) -> str:
    """Nettoyage amélioré des artéfacts OCR"""
    if not text:
        return text

    text = _normalize_unicode(text)

    if pages_texts:
        pages_lines = [t.splitlines() for t in pages_texts]
        pages_lines = _strip_headers_footers(pages_lines)
        text = "\n".join("\n".join(lines) for lines in pages_lines)

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    page_patterns = [
        r"(?im)^\s*page\s*[:\-]?\s*\d+\s*(?:/|sur|de|of)\s*\d+\s*$",
        r"(?im)^\s*p(?:age)?\.?\s*[:\-]?\s*\d+\s*$",
        r"(?im)^\s*\d+\s*/\s*\d+\s*$",
        r"(?im)^\s*[-—–]+\s*\d+\s*[-—–]+\s*$",
        r"(?im)^\s*\d{1,3}\s*$",
        r"(?im)^\s*page\s+n[°º]\s*\d+\s*$",
        r"(?im)^\s*page\s*[-—–]+\s*\d+\s*[-—–]+\s*$",
    ]
    
    for pattern in page_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[—–\-]{2,}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[-—–\s]*\s*$", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"(?<![.!?:;])\n(?!\n)(?=[a-zàâäçéèêëîïôöùûüÿñ\(\,\;\:\.\-])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    text = text.strip()
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r'^[\d\s\-—–\.]{1,10}$', stripped):
            cleaned_lines.append(line)
        elif not stripped:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def calculate_ocr_stats(text):
    """Calcule les statistiques du texte OCR"""
    if not text:
        return "Texte vide"
    
    lines = text.split('\n')
    words = len(text.split())
    chars = len(text)
    non_empty_lines = len([l for l in lines if l.strip()])
    
    return f"{chars:,} caractères | {words:,} mots | {non_empty_lines} lignes non vides | {len(lines)} lignes totales"

# ================
#  CONTRÔLE QUALITÉ
# ================

def _qc_extract_references(text: str):
    """Extrait des références juridiques"""
    if not text:
        return set()
    refs = set()
    t = text.lower()
    
    for m in re.finditer(r"\b(art\.|article)\s+([lrd]\s*[\.\-]?\s*\d+[0-9\-\._]*)", t):
        refs.add(m.group(0).strip())
    
    for m in re.finditer(r"\bcode\s+du\s+(travail|procédure\s+civile|civil)\b", t):
        refs.add(m.group(0).strip())
    
    return refs

def qc_check_references(ocr_text: str, analysis_text: str) -> str:
    """Contrôle qualité des références"""
    if not analysis_text:
        return "Contrôle qualité : contenu d'analyse vide."
    
    ocr_refs = _qc_extract_references(ocr_text or "")
    out_refs = _qc_extract_references(analysis_text or "")
    
    if not out_refs:
        return "Contrôle qualité : aucune référence détectée dans l'analyse."
    
    missing = sorted(r for r in out_refs if r not in ocr_refs)
    
    if not missing:
        return f"Contrôle qualité : toutes les références citées ({len(out_refs)}) apparaissent dans le texte OCR."
    
    report = ["Contrôle qualité : références citées par l'analyse MAIS absentes du texte OCR :"]
    for r in missing:
        report.append(f"  • {r}")
    report.append(f"\nStatistiques : {len(out_refs)} références totales, {len(missing)} manquantes")
    report.append("\nRecommandation : remplacer par « non précisé dans le document » ou reformuler.")
    
    return "\n".join(report)

# ================
#  OLLAMA (API)
# ================

def generate_with_ollama(model: str, prompt_text: str, full_text: str,
                         num_ctx: int, num_predict: int, temperature: float = 0.2,
                         timeout: int = 900) -> str:
    """Génération avec Ollama"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": f"{prompt_text}\nTexte :\n{full_text}",
        "stream": True,
        "options": {
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        },
        "system": (
            "Tu es juriste en droit du travail. Rédige une analyse argumentée en français juridique, "
            "sans listes à puces, en citant uniquement les fondements mentionnés dans le texte."
        ),
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
        error_text = response.text
        if "system memory" in error_text.lower() and "available" in error_text.lower():
            return f"MÉMOIRE INSUFFISANTE : Le modèle nécessite plus de RAM que disponible.\n\n" \
                   f"Solutions :\n" \
                   f"1. Utilisez un modèle plus léger (mistral:7b-instruct ou deepseek-coder:latest)\n" \
                   f"2. Fermez d'autres applications pour libérer de la RAM\n" \
                   f"3. Redémarrez Ollama : 'ollama serve'\n\n" \
                   f"Erreur complète : {error_text}"
        return f"Erreur HTTP {response.status_code} : {error_text}"

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
    return result if result else "Aucune réponse reçue (flux vide)."

# ====================
#  PIPELINE ANALYSE
# ====================

def ocr_and_analyze(pdf_path, nettoyer, modele, profil, max_tokens_out,
                    prompt_text, mode_analysis, comparer, auto_analyze=True, force_ocr=False):
    """Pipeline principal avec cache OCR"""
    preview = ""
    
    if not pdf_path:
        return "Aucun fichier PDF fourni.", "", "", "", ""
    
    try:
        pdf_hash = get_pdf_hash(pdf_path)
        ocr_data = None
        
        if pdf_hash and not force_ocr:
            ocr_data = load_ocr_cache(pdf_hash, nettoyer)
        
        if ocr_data:
            print("Utilisation du cache OCR existant")
            preview = ocr_data['preview']
            stats = ocr_data['stats']
            total_pages = ocr_data.get('total_pages', '?')
            print(f"Cache utilisé : {total_pages} page(s) déjà traitées")
        else:
            print(f"Conversion PDF : {pdf_path}")
            images = convert_from_path(pdf_path)
            raw_pages = []
            
            total_pages = len(images)
            print(f"Traitement de {total_pages} page(s)...")
            
            for i, image in enumerate(images):
                print(f"OCR page {i+1}/{total_pages}...")
                page_text = pytesseract.image_to_string(image, lang="fra")
                raw_pages.append(page_text or "")

            if nettoyer:
                print("Nettoyage du texte...")
                cleaned_pages = [_normalize_unicode(t) for t in raw_pages]
                preview = smart_clean("\n".join(cleaned_pages), pages_texts=cleaned_pages)
            else:
                preview = "\n".join(raw_pages).strip()

            stats = calculate_ocr_stats(preview)
            
            if pdf_hash:
                cache_data = {
                    'preview': preview,
                    'stats': stats,
                    'total_pages': total_pages,
                    'timestamp': str(os.path.getmtime(pdf_path))
                }
                save_ocr_cache(pdf_hash, nettoyer, cache_data)

        if not preview.strip():
            return "Aucun texte détecté lors de l'OCR.", "", "", stats, preview

        if not auto_analyze:
            return "OCR et nettoyage terminés. Texte prêt pour analyse.", "", "", stats, preview

        text_length = len(preview)
        estimated_tokens = text_length // 4
        
        print(f"Longueur du texte : {text_length:,} caractères (≈{estimated_tokens:,} tokens)")
        
        if estimated_tokens > 20000:
            print("Document très volumineux détecté - recommandation profil 'Maxi'")
        elif estimated_tokens > 10000:
            print("Document volumineux - recommandation profil 'Confort' ou 'Maxi'")

        profiles = {
            "Rapide":  {"num_ctx": 8192,  "temperature": 0.2},
            "Confort": {"num_ctx": 16384, "temperature": 0.2},
            "Maxi":    {"num_ctx": 32768, "temperature": 0.2},
        }
        base = profiles.get(profil, profiles["Confort"])
        num_ctx = base["num_ctx"]
        temperature = base["temperature"]
        
        if estimated_tokens > num_ctx * 0.8:
            print(f"ATTENTION : Le texte ({estimated_tokens:,} tokens) approche la limite du contexte ({num_ctx:,})")
            print("Considérez utiliser le profil 'Maxi' pour de meilleurs résultats")

        main_prompt = EXPERT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else prompt_text

        print(f"Génération avec {modele} en mode {mode_analysis} et prompt : {main_prompt}")
        
        print("
===== PROMPT COMPLET TRANSMIS À OLLAMA =====")
print(main_prompt + '
Texte :
' + preview)
analyse = generate_with_ollama(
            modele, main_prompt, preview,
            num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature
        )

        qc_or_compare = ""
        analyse_alt = ""

        if (mode_analysis or "").lower().startswith("expert"):
            print("Contrôle qualité...")
            qc_or_compare = qc_check_references(preview, analyse)

        if comparer:
            print("Génération comparative...")
            alt_prompt = DEFAULT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else EXPERT_PROMPT_TEXT
            analyse_alt = generate_with_ollama(
                modele, alt_prompt, preview,
                num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature
            )
            
            def _summary_diff(a, b):
                if not a or not b or a.startswith("Erreur"):
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
            
            comp = _summary_diff(analyse, analyse_alt)
            qc_or_compare = (qc_or_compare + "\n\n" + comp).strip() if qc_or_compare else comp

        return analyse, analyse_alt, qc_or_compare, stats, preview

    except Exception as e:
        traceback.print_exc()
        error_msg = f"Erreur lors du traitement : {str(e)}"
        stats = "Erreur - Impossible de calculer les statistiques"
        return error_msg, "", "", stats, preview

# ==================
#  MODELES OLLAMA
# ==================

def get_ollama_models():
    """Récupère la liste des modèles avec priorité aux plus légers"""
    fallback_models = [
        "mistral:7b-instruct",
        "mistral:latest", 
        "deepseek-coder:latest",
        "llama3:latest",
        "llama3.1:8b-instruct-q5_K_M"
    ]
    
    try:
        print("Tentative de récupération des modèles Ollama...")
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if r.status_code != 200:
            print(f"Erreur API Ollama - Status {r.status_code}: {r.text}")
            return fallback_models
        
        data = r.json()
        print(f"Réponse Ollama brute: {data}")
        
        models = data.get("models", [])
        names = []
        
        for m in models:
            print(f"Modèle analysé: {m}")
            name = m.get("name")
            if name:
                names.append(name)
                print(f"Modèle ajouté: {name}")
        
        if not names:
            print("Aucun modèle trouvé dans la réponse, utilisation des modèles de fallback")
            return fallback_models
        
        print(f"Total: {len(names)} modèle(s) récupéré(s) depuis l'API")
        return names
        
    except requests.exceptions.ConnectionError:
        print("Ollama non accessible (connexion refusée)")
        print("Vérifiez qu'Ollama est démarré avec: ollama serve")
        return fallback_models
    except Exception as e:
        print(f"Erreur récupération modèles: {e}")
        return fallback_models

# ===========
#  UI GRADIO
# ===========

def build_ui():
    models_list = get_ollama_models()
    store = load_prompt_store()
    prompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
    
    script_name = os.path.basename(__file__) if '__file__' in globals() else "ollama_juridique.py"

    with gr.Blocks(title=f"{script_name} - OCR Juridique + Ollama") as demo:
        gr.Markdown("## OCR structuré + Analyse juridique (Ollama)")
        gr.Markdown(f"**Fichier des prompts** : `{PROMPT_STORE_PATH}`")

        with gr.Row():
            pdf_file = gr.File(label="Uploader un PDF scanné", file_types=[".pdf"])
            nettoyer = gr.Checkbox(label="Nettoyage avancé", value=True)
            auto_analyze = gr.Checkbox(label="Lancer l'analyse automatiquement", value=True)

        with gr.Row():
            force_ocr = gr.Checkbox(label="Forcer nouveau OCR (ignorer le cache)", value=False)
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
                default_model = models_list[0]
                
            modele = gr.Dropdown(label="Modèle Ollama", choices=models_list, value=default_model)
            profil = gr.Radio(label="Profil", choices=["Rapide", "Confort", "Maxi"], value="Confort")
            max_tokens_out = gr.Slider(label="Longueur (tokens)", minimum=256, maximum=2048, step=128, value=1280)
        
        with gr.Row():
            mode_analysis = gr.Radio(label="Mode", choices=["Standard", "Expert"], value="Standard")
            comparer = gr.Checkbox(label="Comparer avec l'autre mode", value=False)

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
            save_as_btn = gr.Button("Enregistrer sous...", variant="secondary")
            rename_btn = gr.Button("Renommer", variant="secondary")
            delete_btn = gr.Button("Supprimer", variant="secondary")
            reset_btn = gr.Button("Réinitialiser", variant="secondary")
            reload_btn = gr.Button("Recharger", variant="secondary")

        info_box = gr.Markdown("")

        gr.Markdown("### Compositeur de prompts (fichiers prompt*.txt)")
        
        prompt_files = get_prompt_files(PROMPT_FILES_DIR)
        
        with gr.Row():
            files_checklist = gr.CheckboxGroup(
                label="Fichiers prompt*.txt disponibles dans ./prompts/",
                choices=prompt_files,
                value=[]
            )
            refresh_files_btn = gr.Button("Actualiser fichiers", variant="secondary")
        
        with gr.Row():
            compose_btn = gr.Button("Composer prompt général", variant="primary")
            copy_to_main_btn = gr.Button("Copier vers prompt principal", variant="secondary")
        
        composed_prompt_box = gr.Textbox(
            label="Prompt général composé",
            lines=15,
            interactive=False,
            show_copy_button=True,
            placeholder="Le prompt composé apparaîtra ici..."
        )

        run_btn = gr.Button("Lancer l'analyse", variant="primary", size="lg")

        with gr.Row():
            ocr_only_btn = gr.Button("OCR seulement", variant="secondary", visible=False)

        def toggle_ocr_button_visibility(auto_analyze_value):
            return gr.Button.update(visible=not auto_analyze_value)

        auto_analyze.change(
            fn=toggle_ocr_button_visibility,
            inputs=[auto_analyze],
            outputs=[ocr_only_btn]
        )

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

        # États
        prompts_state = gr.State(value=store)
        current_name = gr.State(value=DEFAULT_PROMPT_NAME)

        # Fonctions callbacks
        def clear_cache():
            count = clear_ocr_cache()
            if count > 0:
                return gr.Markdown.update(value=f"Cache vidé : {count} fichier(s) supprimé(s)")
            else:
                return gr.Markdown.update(value="Cache déjà vide")

        def launch_analysis(pdf_path, nettoyer, modele, profil, max_tokens_out, 
                           prompt_text, mode_analysis, comparer, auto_analyze, force_ocr):
            try:
                return ocr_and_analyze(pdf_path, nettoyer, modele, profil, max_tokens_out, 
                                      prompt_text, mode_analysis, comparer, auto_analyze, force_ocr)
            except Exception as e:
                traceback.print_exc()
                return f"Erreur : {str(e)}", "", "", "Erreur", ""

        def launch_ocr_only(pdf_path, nettoyer, force_ocr):
            try:
                return ocr_and_analyze(pdf_path, nettoyer, "", "", 0, "", "", False, False, force_ocr)
            except Exception as e:
                traceback.print_exc()
                return f"Erreur OCR : {str(e)}", "", "", "Erreur", ""

        # Gestion prompts JSON
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
                        gr.Markdown.update(value=msg),
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

        def on_reload():
            try:
                store = load_prompt_store()
                names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
                text = store.get(DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT)
                return (
                    store,
                    gr.Dropdown.update(choices=names, value=DEFAULT_PROMPT_NAME),
                    gr.Textbox.update(value=text),
                    gr.Textbox.update(value=DEFAULT_PROMPT_NAME),
                    gr.Markdown.update(value=f"Store rechargé ({len(store)} prompt(s))"),
                    DEFAULT_PROMPT_NAME
                )
            except Exception as e:
                return (
                    {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT},
                    gr.Dropdown.update(choices=[DEFAULT_PROMPT_NAME], value=DEFAULT_PROMPT_NAME),
                    gr.Textbox.update(value=DEFAULT_PROMPT_TEXT),
                    gr.Textbox.update(value=DEFAULT_PROMPT_NAME),
                    gr.Markdown.update(value=f"Erreur : {e}"),
                    DEFAULT_PROMPT_NAME
                )

        # Gestion fichiers prompts
        def refresh_prompt_files():
            files = get_prompt_files(PROMPT_FILES_DIR)
            return gr.CheckboxGroup.update(choices=files, value=[])

        def compose_selected_prompts(selected_files):
            if not selected_files:
                return "Aucun fichier sélectionné."
            return concatenate_prompts(selected_files, PROMPT_FILES_DIR)

        def copy_composed_to_main(composed_text):
            if not composed_text or "Aucun fichier" in composed_text:
                return (
                    gr.Textbox.update(),
                    gr.Markdown.update(value="Aucun prompt composé à copier.")
                )
            
            lines = composed_text.split('\n')
            clean_lines = []
            skip = False
            
            for line in lines:
                if line.startswith("=== ") or line.startswith("--- "):
                    skip = not line.startswith("--- ")
                    if line.startswith("--- "):
                        clean_lines.append(f"# {line}")
                elif not skip and line.strip():
                    clean_lines.append(line)
            
            clean_prompt = '\n'.join(clean_lines).strip()
            
            return (
                gr.Textbox.update(value=clean_prompt),
                gr.Markdown.update(value="Prompt composé copié.")
            )

        # Connexions
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

        reload_btn.click(
            fn=on_reload,
            outputs=[prompts_state, prompt_selector, prompt_box, prompt_name_box, info_box, current_name]
        )

        refresh_files_btn.click(
            fn=refresh_prompt_files,
            outputs=[files_checklist]
        )

        compose_btn.click(
            fn=compose_selected_prompts,
            inputs=[files_checklist],
            outputs=[composed_prompt_box]
        )

        copy_to_main_btn.click(
            fn=copy_composed_to_main,
            inputs=[composed_prompt_box],
            outputs=[prompt_box, info_box]
        )

        run_btn.click(
            fn=launch_analysis,
            inputs=[pdf_file, nettoyer, modele, profil, max_tokens_out, prompt_box, mode_analysis, comparer, auto_analyze, force_ocr],
            outputs=[analysis_box, analysis_alt_box, compare_box, ocr_stats, preview_box]
        )

        ocr_only_btn.click(
            fn=launch_ocr_only,
            inputs=[pdf_file, nettoyer, force_ocr],
            outputs=[analysis_box, analysis_alt_box, compare_box, ocr_stats, preview_box]
        )

        clear_cache_btn.click(
            fn=clear_cache,
            outputs=[cache_info]
        )

        gr.Markdown("""
        ### Conseils d'utilisation
        
        **OCR et Cache :**
        - OCR intégral : Traite toutes les pages du PDF
        - Cache intelligent : L'OCR est automatiquement mis en cache
        - Forcer nouveau OCR : Cochez pour ignorer le cache
        - Vider le cache : Supprime tous les fichiers OCR mis en cache
        
        **Gestion des prompts :**
        - Prompts JSON : Création/modification/sauvegarde via l'interface
        - Fichiers prompt*.txt : Placez vos fichiers dans ./prompts/
        - Composition : Sélectionnez plusieurs fichiers pour créer un prompt combiné
        
        **Recommandations modèles (selon RAM) :**
        - 9 GB ou moins : mistral:7b-instruct, deepseek-coder:latest
        - 12+ GB : llama3:latest, mistral:latest
        - 16+ GB : llama3.1:8b-instruct-q5_K_M
        
        **Profils :**
        - Rapide : < 15 pages (8k contexte)
        - Confort : 15-30 pages (16k contexte)
        - Maxi : 30+ pages (32k contexte)
        
        Répertoire : `./prompts/`
        Prérequis : Ollama démarré (`ollama serve`)
        """)

    return demo

if __name__ == "__main__":
    print("Démarrage de l'interface OCR Juridique...")
    print(f"Répertoire des prompts : {PROMPT_STORE_DIR}")
    
    if not os.path.exists(PROMPT_STORE_DIR):
        try:
            os.makedirs(PROMPT_STORE_DIR)
            print(f"Répertoire ./prompts/ créé : {PROMPT_STORE_DIR}")
        except Exception as e:
            print(f"Erreur création répertoire : {e}")
    
    try:
        models = get_ollama_models()
        print(f"Modèles Ollama disponibles : {len(models)}")
        for model in models:
            print(f"  - {model}")
        
        app = build_ui()
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False
        )
    except Exception as e:
        print(f"Erreur au démarrage : {e}")
        traceback.print_exc()
