#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR structuré + Analyse juridique (Ollama) - VERSION CORRIGÉE
- Nettoyage amélioré des numéros de page
- Gestion prompts JSON corrigée
- Gestion d'erreurs renforcée
- OCR indépendant de l'analyse automatique
"""

import os
import json
import re
import unicodedata
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
    # 1) XDG_CONFIG_HOME/ollama_juridique
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    target = os.path.join(xdg, "ollama_juridique")
    try:
        os.makedirs(target, exist_ok=True)
        # Test d'écriture
        test_path = os.path.join(target, ".write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return target
    except Exception:
        # 2) fallback : dossier du script
        sd = _script_dir()
        try:
            test_path = os.path.join(sd, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
            return sd
        except Exception:
            # 3) dernier recours : CWD
            return os.getcwd()

PROMPT_STORE_DIR = _default_store_dir()
PROMPT_STORE_PATH = os.path.join(PROMPT_STORE_DIR, "prompts_store.json")

# =================================
#  GESTIONNAIRE DE PROMPTS (JSON) - CORRIGÉ
# =================================

def atomic_write_text(path: str, data: str) -> None:
    """Écriture atomique avec gestion d'erreur améliorée"""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        # Nettoyage du fichier temporaire en cas d'erreur
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass
        raise

def load_prompt_store() -> Dict[str, str]:
    """
    Charge le fichier JSON des prompts. S'il n'existe pas, crée un store
    avec un prompt par défaut. FONCTION CORRIGÉE.
    """
    default_store = {DEFAULT_PROMPT_NAME: DEFAULT_PROMPT_TEXT}
    
    if not os.path.exists(PROMPT_STORE_PATH):
        try:
            # Créer le répertoire si nécessaire
            os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
            atomic_write_text(PROMPT_STORE_PATH, json.dumps(default_store, ensure_ascii=False, indent=2))
            return default_store
        except Exception as e:
            print(f"Erreur création store initial: {e}")
            return default_store
    
    try:
        with open(PROMPT_STORE_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
        
        # Vérifications de cohérence
        if not isinstance(store, dict):
            print("Store invalide (pas un dict), réinitialisation")
            return default_store
        
        if not store:
            print("Store vide, réinitialisation")
            return default_store
            
        # S'assurer que le prompt par défaut existe
        if DEFAULT_PROMPT_NAME not in store:
            store[DEFAULT_PROMPT_NAME] = DEFAULT_PROMPT_TEXT
            save_prompt_store(store)
            
        return store
        
    except json.JSONDecodeError as e:
        print(f"Erreur JSON dans le store: {e}")
        return default_store
    except Exception as e:
        print(f"Erreur chargement store: {e}")
        return default_store

def save_prompt_store(store: Dict[str, str]) -> Tuple[bool, str]:
    """
    Sauvegarde le store avec gestion d'erreur améliorée. FONCTION CORRIGÉE.
    """
    try:
        # Validation du store avant sauvegarde
        if not isinstance(store, dict):
            return False, "Erreur: store n'est pas un dictionnaire"
        
        # S'assurer que le répertoire existe
        os.makedirs(os.path.dirname(PROMPT_STORE_PATH), exist_ok=True)
        
        # Écriture atomique
        json_data = json.dumps(store, ensure_ascii=False, indent=2)
        atomic_write_text(PROMPT_STORE_PATH, json_data)
        
        return True, f"Enregistré dans : `{PROMPT_STORE_PATH}`"
        
    except PermissionError:
        return False, f"Erreur de permissions pour : `{PROMPT_STORE_PATH}`"
    except OSError as e:
        return False, f"Erreur système : {e}"
    except Exception as e:
        return False, f"Échec d'enregistrement : {e}"

def sanitize_name(name: str) -> str:
    """Nettoie et valide un nom de prompt"""
    if not name:
        return ""
    # Nettoyer les espaces multiples et caractères problématiques
    name = re.sub(r"\s+", " ", name.strip())
    # Supprimer les caractères problématiques pour les noms de fichiers
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:100]  # Limiter la longueur

# ================
#  OCR & NETTOYAGE - AMÉLIORÉ
# ================

def _normalize_unicode(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    # Ligatures courantes & apostrophes typographiques
    replacements = {
        "ï¬": "fi", "ï¬‚": "fl", "ï¬€": "ff", "ï¬ƒ": "ffi", "ï¬„": "ffl",
        "'": "'", "'": "'", """: '"', """: '"',
        "—": "-", "–": "-", "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def _strip_headers_footers(pages_lines):
    """
    Détecte/supprime en-têtes/pieds répétés avec seuil ajustable
    """
    if not pages_lines or len(pages_lines) < 2:
        return pages_lines
        
    counter = Counter()
    
    # Analyser les lignes potentiellement répétitives
    for lines in pages_lines:
        if not lines:
            continue
        sample = []
        # Prendre plus de lignes en début/fin pour mieux détecter
        sample.extend(lines[:3])  # 3 premières lignes
        if len(lines) > 6:
            sample.extend(lines[-3:])  # 3 dernières lignes
        
        # Ne considérer que les lignes avec du contenu significatif
        for line in sample:
            clean_line = line.strip()
            if len(clean_line) >= 3 and not clean_line.isdigit():
                counter.update([clean_line])
    
    # Seuil adaptatif : plus de pages = seuil plus bas
    min_occurrences = max(2, len(pages_lines) // 3)
    repeated = {l for l, c in counter.items() if c >= min_occurrences}
    
    # Nettoyer les pages
    cleaned_pages = []
    for lines in pages_lines:
        new_lines = []
        for line in lines:
            if line.strip() not in repeated:
                new_lines.append(line)
        cleaned_pages.append(new_lines)
    
    return cleaned_pages

def smart_clean(text: str, pages_texts=None) -> str:
    """
    Nettoyage amélioré des artéfacts OCR avec meilleure détection des numéros de page
    """
    if not text:
        return text

    text = _normalize_unicode(text)

    # Supprimer les en-têtes/pieds répétés si pages disponibles
    if pages_texts:
        pages_lines = [t.splitlines() for t in pages_texts]
        pages_lines = _strip_headers_footers(pages_lines)
        text = "\n".join("\n".join(lines) for lines in pages_lines)

    # Césures interlignes
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # AMÉLIORATION : Patterns étendus pour les numéros de page
    page_patterns = [
        # "Page 3/50", "Page 3 sur 50", "Page 3 de 50"
        r"(?im)^\s*page\s*[:\-]?\s*\d+\s*(?:/|sur|de|of)\s*\d+\s*$",
        # "Page 7", "p. 7", "P. 7"
        r"(?im)^\s*p(?:age)?\.?\s*[:\-]?\s*\d+\s*$",
        # "3/12", "7/45" (ligne complète)
        r"(?im)^\s*\d+\s*/\s*\d+\s*$",
        # "- 3 -", "--- 7 ---"
        r"(?im)^\s*[-—–]+\s*\d+\s*[-—–]+\s*$",
        # Numéros seuls en début/fin de ligne
        r"(?im)^\s*\d{1,3}\s*$",
        # "Page n° 7", "Page nº 7"
        r"(?im)^\s*page\s+n[°º]\s*\d+\s*$",
        # Formats avec tirets "Page - 3 -"
        r"(?im)^\s*page\s*[-—–]+\s*\d+\s*[-—–]+\s*$",
    ]
    
    for pattern in page_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # Lignes contenant uniquement des chiffres/numéros isolés (plus agressif)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    
    # Tirets seuls / ornements
    text = re.sub(r"(?m)^\s*[—–\-]{2,}\s*$", "", text)
    
    # Lignes vides multiples ou avec seulement des espaces/tirets
    text = re.sub(r"(?m)^\s*[-—–\s]*\s*$", "", text)

    # Espaces en fin de ligne
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    
    # Espaces multiples → espace simple
    text = re.sub(r" {2,}", " ", text)

    # Fusion intelligente des lignes (préserver les vrais paragraphes)
    # Ne pas fusionner si la ligne précédente se termine par : . ! ? : ;
    text = re.sub(r"(?<![.!?:;])\n(?!\n)(?=[a-zàâäçéèêëîïôöùûüÿñ\(\,\;\:\.\-])", " ", text)

    # Compactage des lignes vides (max 2 consécutives)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Nettoyage final
    text = text.strip()
    
    # Post-traitement : supprimer les lignes qui ne contiennent que des numéros après nettoyage
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Ignorer les lignes qui ne sont que des chiffres ou des ponctuations simples
        if stripped and not re.match(r'^[\d\s\-—–\.]{1,10}$', stripped):
            cleaned_lines.append(line)
        elif not stripped:  # Conserver les lignes vides pour la structure
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

# =====================
#  CONTRÔLE QUALITÉ (QC)
# =====================

def _qc_extract_references(text: str):
    """
    Extrait des références juridiques (heuristique améliorée).
    """
    if not text:
        return set()
    refs = set()
    t = text.lower()
    
    # Articles
    for m in re.finditer(r"\b(art\.|article)\s+([lrd]\s*[\.\-]?\s*\d+[0-9\-\._]*)", t):
        refs.add(m.group(0).strip())
    
    # Codes
    for m in re.finditer(r"\bcode\s+du\s+(travail|procédure\s+civile|civil)\b", t):
        refs.add(m.group(0).strip())
    
    return refs

def qc_check_references(ocr_text: str, analysis_text: str) -> str:
    """Contrôle qualité amélioré des références"""
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
    report.append("\nRecommandation : remplacer par « non précisé dans le document » ou reformuler sans citer.")
    
    return "\n".join(report)

# ================
#  OLLAMA (API)
# ================

def generate_with_ollama(model: str, prompt_text: str, full_text: str,
                         num_ctx: int, num_predict: int, temperature: float = 0.2,
                         timeout: int = 900) -> str:
    """Génération avec Ollama - gestion d'erreur améliorée"""
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
        return f"Erreur : Délai dépassé ({timeout}s). Le modèle met trop de temps à répondre."
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
    return result if result else "Aucune réponse reçue (flux vide)."

# ====================
#  PIPELINE ANALYSE - CORRIGÉ
# ====================

def ocr_and_analyze(pdf_path, nettoyer, modele, profil, max_tokens_out,
                    prompt_text, mode_analysis, comparer, auto_analyze=True):
    """Pipeline principal avec OCR toujours actif et analyse conditionnelle"""
    preview = ""
    
    if not pdf_path:
        return "Aucun fichier PDF fourni.", "", "", "", ""
    
    try:
        # OCR des 3 premières pages (TOUJOURS FAIT)
        print(f"Conversion PDF : {pdf_path}")
        images = convert_from_path(pdf_path)
        raw_pages = []
        
        for i, image in enumerate(images[:3]):
            print(f"OCR page {i+1}/3...")
            page_text = pytesseract.image_to_string(image, lang="fra")
            raw_pages.append(page_text or "")

        # Nettoyage (TOUJOURS FAIT SI ACTIVÉ)
        if nettoyer:
            print("Nettoyage du texte...")
            cleaned_pages = [_normalize_unicode(t) for t in raw_pages]
            preview = smart_clean("\n".join(cleaned_pages), pages_texts=cleaned_pages)
        else:
            preview = "\n".join(raw_pages).strip()

        # Calculer les statistiques OCR
        stats = calculate_ocr_stats(preview)

        if not preview.strip():
            return "Aucun texte détecté lors de l'OCR.", "", "", stats, preview

        # SI L'ANALYSE AUTO N'EST PAS ACTIVÉE, S'ARRÊTER ICI
        if not auto_analyze:
            return "OCR et nettoyage terminés. Texte prêt pour analyse.", "", "", stats, preview

        # Configuration du profil pour l'analyse
        profiles = {
            "Rapide":  {"num_ctx": 4096,  "temperature": 0.2},
            "Confort": {"num_ctx": 8192,  "temperature": 0.2},
            "Maxi":    {"num_ctx": 12288, "temperature": 0.2},
        }
        base = profiles.get(profil, profiles["Confort"])
        num_ctx = base["num_ctx"]
        temperature = base["temperature"]

        # Sélection du prompt principal
        main_prompt = EXPERT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else prompt_text

        print(f"Génération avec {modele} en mode {mode_analysis}...")
        
        # Analyse principale
        analyse = generate_with_ollama(
            modele, main_prompt, preview,
            num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature
        )

        qc_or_compare = ""
        analyse_alt = ""

        # Contrôle qualité en mode Expert
        if (mode_analysis or "").lower().startswith("expert"):
            print("Contrôle qualité...")
            qc_or_compare = qc_check_references(preview, analyse)

        # Comparaison optionnelle
        if comparer:
            print("Génération comparative...")
            alt_prompt = DEFAULT_PROMPT_TEXT if (mode_analysis or "").lower().startswith("expert") else EXPERT_PROMPT_TEXT
            analyse_alt = generate_with_ollama(
                modele, alt_prompt, preview,
                num_ctx=num_ctx, num_predict=max_tokens_out, temperature=temperature
            )
            
            # Statistiques de comparaison
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
    """Récupère la liste des modèles avec gestion d'erreur"""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        if r.status_code != 200:
            print(f"Erreur API Ollama : {r.status_code}")
            return ["llama3.1:8b-instruct-q5_K_M", "mistral"]
        
        data = r.json()
        models = data.get("models", [])
        names = []
        
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                names.append(name)
        
        return names if names else ["llama3.1:8b-instruct-q5_K_M", "mistral"]
        
    except requests.exceptions.ConnectionError:
        print("Ollama non accessible, utilisation des modèles par défaut")
        return ["llama3.1:8b-instruct-q5_K_M", "mistral"]
    except Exception as e:
        print(f"Erreur récupération modèles : {e}")
        return ["llama3.1:8b-instruct-q5_K_M", "mistral"]

# ===========
#  UI GRADIO - CORRIGÉE
# ===========

def build_ui():
    models_list = get_ollama_models()
    store = load_prompt_store()
    prompt_names = [DEFAULT_PROMPT_NAME] + sorted([n for n in store.keys() if n != DEFAULT_PROMPT_NAME])
    
    # Nom dynamique basé sur le fichier du script
    script_name = os.path.basename(__file__) if '__file__' in globals() else "ollama_juridique.py"

    with gr.Blocks(title=f"{script_name} - OCR Juridique + Ollama") as demo:
        gr.Markdown("## OCR structuré + Analyse juridique (Ollama)")
        gr.Markdown(f"**Fichier des prompts** : `{PROMPT_STORE_PATH}`")

        # Ligne 1 : dépôt + nettoyage + analyse auto
        with gr.Row():
            pdf_file = gr.File(label="Uploader un PDF scanné", file_types=[".pdf"])
            nettoyer = gr.Checkbox(label="Nettoyage avancé (en-têtes, césures, numéros, pages)", value=True)
            auto_analyze = gr.Checkbox(label="Lancer l'analyse automatiquement", value=True)

        # Ligne 2 : modèle + profil + longueur + mode
        with gr.Row():
            modele = gr.Dropdown(label="Modèle Ollama", choices=models_list, value=models_list[0])
            profil = gr.Radio(label="Profil d'inférence", choices=["Rapide", "Confort", "Maxi"], value="Confort")
            max_tokens_out = gr.Slider(label="Longueur de sortie (tokens)", minimum=256, maximum=2048, step=128, value=1280)
        
        with gr.Row():
            mode_analysis = gr.Radio(label="Mode d'analyse", choices=["Standard", "Expert"], value="Standard")
            comparer = gr.Checkbox(label="Comparer avec l'autre mode", value=False)

        gr.Markdown("### Prompt — gestion persistante (JSON)")

        with gr.Row():
            prompt_selector = gr.Dropdown(
                label="Choisir un prompt",
                choices=prompt_names,
                value=DEFAULT_PROMPT_NAME
            )
            prompt_name_box = gr.Textbox(
                label="Nom du prompt (Enregistrer sous / Renommer)", 
                lines=1,
                placeholder="Entrez un nom pour le prompt..."
            )

        prompt_box = gr.Textbox(
            label="Contenu du prompt (modifiable)",
            value=store.get(DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEXT),
            lines=12,
            interactive=True,
            placeholder="Tapez votre prompt ici..."
        )

        with gr.Row():
            save_btn = gr.Button("Enregistrer", variant="secondary")
            save_as_btn = gr.Button("Enregistrer sous...", variant="secondary")
            rename_btn = gr.Button("Renommer", variant="secondary")
            delete_btn = gr.Button("Supprimer", variant="secondary")
            reset_btn = gr.Button("Réinitialiser", variant="secondary")
            reload_btn = gr.Button("Recharger", variant="secondary")

        info_box = gr.Markdown("")  # messages de statut

        run_btn = gr.Button("Lancer l'analyse", variant="primary", size="lg")

        # Bouton OCR seul (visible seulement si analyse auto désactivée)
        with gr.Row():
            ocr_only_btn = gr
